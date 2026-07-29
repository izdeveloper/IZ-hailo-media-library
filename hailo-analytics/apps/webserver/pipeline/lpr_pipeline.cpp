#include "lpr_pipeline.hpp"
#include "lpr_pipeline_builder.hpp"
#include "common/common.hpp"
#include "hailo_analytics/pipeline/sinks/app_sink_stage.hpp"
#include "hailo_analytics/pipeline/routing/tee_stage.hpp"
#include "hailo_analytics/pipeline/ai/postprocess_stage.hpp"

#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <glib.h>
#include <fstream>
#include <regex>
#include <thread>
#include <sys/mman.h>

using namespace hailo_analytics::pipeline;
using namespace hailo_analytics::pipeline::sinks;
using namespace hailo_analytics::pipeline::routing;
using namespace hailo_analytics::pipeline::ai;
using namespace webserver::pipeline;
using namespace webserver::resources;

static const std::string CONFIG_JSON_PATH = "/home/root/apps/license_plate_recognition/resources/event_config_ip.json";

// Reads target external IP dynamically on every event dispatch
static std::string get_target_inex_url() {
    std::ifstream file(CONFIG_JSON_PATH);
    std::string ip = "127.0.0.1";
    std::string port = "8080";

    if (file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        std::regex ip_regex("\"http_endpoint_ip\"\\s*:\\s*\"([^\"]+)\"");
        std::regex port_regex("\"http_endpoint_port\"\\s*:\\s*(\\d+)");
        std::smatch match;

        if (std::regex_search(content, match, ip_regex) && match.size() > 1) ip = match[1].str();
        if (std::regex_search(content, match, port_regex) && match.size() > 1) port = match[1].str();
    }

    return "http://" + ip + ":" + port + "/api/v1/lpr-events?cmd=uploadevent&api_version=1.7";
}

LprPipeline::LprPipeline(webserver::resources::ResourceRepository &resources, MediaLibrary &media_library,
                         RTPConverterStage &webrtc_stage, Architecture platform)
    : BasePipeline(resources, media_library, webrtc_stage, platform, ProfileType::Daylight, {ProfileType::Daylight}) {}

std::string LprPipeline::pipeline_name() const { return "LPR"; }
std::string LprPipeline::get_profile_name_by_type(ProfileType type) const { return "Daylight_FaceLandmarks"; }
ProfileType LprPipeline::get_profile_type_by_name(const std::string &name) const { return ProfileType::Daylight; }

void LprPipeline::build_pipeline() {
    WEBSERVER_LOG_INFO("Building LPR pipeline");

    // 1. WebRTC Sink Stage
    auto webrtc_sink_stage = AppSinkStageBuild::create()
        .set_stage_name("webrtc_sink")
        .set_queue_size_opt(1)
        .set_leaky_opt(false)
        .set_process_func([&](hailo_analytics::pipeline::BufferPtr buf) { m_webrtc_stage.process(buf); })
        .buildptr();

    // 2. AI Sub-pipelines
    auto tiling_pipeline = lpr_app::build_tiling_pipeline("tiling_pipeline", lpr_app::TrackingMode::BALANCED).value();
    auto veh_attrs_pipeline = lpr_app::build_vehicle_attributes_pipeline("vehicle_attributes_pipeline").value();
    auto cls_pipeline = lpr_app::build_classification_pipeline("classification_pipeline").value();
    auto ocr_pipeline = lpr_app::build_ocr_pipeline("ocr_pipeline").value();
         
    // 3. Postprocess stage (.so)
    auto event_engine_stage = PostprocessStageBuild::create()
        .set_stage_name("lpr_event_engine_post")
        .set_so_path("/usr/lib/hailo-post-processes/liblpr_event_analytics_post.so")
        .set_function_name_opt("filter")
        .set_queue_size_opt(5)
        .set_leaky_opt(false)
        .buildptr();
    
    // --- DYNAMIC AI STREAM DIMENSIONS ---
    // Query exact dimensions of the AI stream to prevent Segmentation Faults in OpenCV
    int ai_width = 1920;
    int ai_height = 1080;
    auto output_streams = m_app_resources->media_library.m_frontend->get_outputs_streams();
    if (output_streams.has_value()) {
        for (const auto &stream : output_streams.value()) {
            if (stream.id == "sink2") { // AI path is connected to sink2
                ai_width = stream.width;
                ai_height = stream.height;
                break;
            }
        }
    }

    // 4. Event Sink Stage (Catches frame buffers tagged by the .so)
    auto lpr_event_sink = AppSinkStageBuild::create()
        .set_stage_name("lpr_event_sink")
        .set_queue_size_opt(5)
        .set_leaky_opt(true) // Enable leaky mode so slow AI drops don't block the video pipeline
        .set_process_func([this, ai_width, ai_height](hailo_analytics::pipeline::BufferPtr buf) {
            if (!buf) return;

            HailoROIPtr roi = buf->get_roi();
            if (!roi) return;

            std::string event_json_str = "";

            // The .so file attaches the classification directly to the Main ROI
            for (const auto& obj : roi->get_objects()) {
                if (obj->get_type() == HAILO_CLASSIFICATION) {
                    auto cls = std::dynamic_pointer_cast<HailoClassification>(obj);
                    if (cls && cls->get_classification_type() == "inex_event") {
                        event_json_str = cls->get_label();
                        break; 
                    }
                }
            }

            // If an event tag was found, extract raw memory FAST (< 1ms) and release the Hailo buffer
            if (!event_json_str.empty()) {
                std::cout << "\n[LPR_SINK] Found INEX event tag. Fast-copying frame..." << std::endl;

                std::vector<uint8_t> y_copy;
                std::vector<uint8_t> uv_copy;
                bool is_split_plane = false;

                auto ml_buf = buf->get_buffer(); 
                if (ml_buf) {
                    int fd_y = ml_buf->get_plane_fd(0);
                    int fd_uv = ml_buf->get_plane_fd(1);
                    
                    if (fd_y >= 0) {
                        struct dma_buf_sync sync = { 0 };
                        sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
                        ioctl(fd_y, DMA_BUF_IOCTL_SYNC, &sync);
                        if (fd_uv >= 0) ioctl(fd_uv, DMA_BUF_IOCTL_SYNC, &sync);

                        // Map Y Plane
                        off_t size_y = lseek(fd_y, 0, SEEK_END); lseek(fd_y, 0, SEEK_SET);
                        size_t map_size_y = (size_y > 0) ? size_y : (ai_width * ai_height);
                        void* data_y = mmap(NULL, map_size_y, PROT_READ, MAP_SHARED, fd_y, 0);

                        // Map UV Plane
                        void* data_uv = MAP_FAILED;
                        size_t map_size_uv = 0;
                        if (fd_uv >= 0) {
                            off_t size_uv = lseek(fd_uv, 0, SEEK_END); lseek(fd_uv, 0, SEEK_SET);
                            map_size_uv = (size_uv > 0) ? size_uv : (ai_width * ai_height / 2);
                            data_uv = mmap(NULL, map_size_uv, PROT_READ, MAP_SHARED, fd_uv, 0);
                        }

                        // Fast copy raw bytes to std::vector
                        if (data_y != MAP_FAILED) {
                            if (fd_uv >= 0 && data_uv != MAP_FAILED) {
                                is_split_plane = true;
                                y_copy.assign((uint8_t*)data_y, (uint8_t*)data_y + (ai_width * ai_height));
                                uv_copy.assign((uint8_t*)data_uv, (uint8_t*)data_uv + (ai_width * ai_height / 2));
                                munmap(data_uv, map_size_uv);
                            } else {
                                y_copy.assign((uint8_t*)data_y, (uint8_t*)data_y + (ai_width * ai_height * 3 / 2));
                            }
                            munmap(data_y, map_size_y);
                        }

                        sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
                        ioctl(fd_y, DMA_BUF_IOCTL_SYNC, &sync);
                        if (fd_uv >= 0) ioctl(fd_uv, DMA_BUF_IOCTL_SYNC, &sync);
                    }
                }

                // BufferPtr `buf` goes out of scope here and immediately frees the DMA buffer back to the pool!

                // Offload CPU-heavy tasks (OpenCV conversion, JPEG encoding, Base64, JSON, and cURL) to a background thread
                std::thread([this, event_json_str, y_copy = std::move(y_copy), uv_copy = std::move(uv_copy), is_split_plane, ai_width, ai_height]() {
                    std::string base64_image = "";

                    if (!y_copy.empty()) {
                        try {
                            cv::Mat bgr;
                            if (is_split_plane && !uv_copy.empty()) {
                                cv::Mat y(ai_height, ai_width, CV_8UC1, (void*)y_copy.data());
                                cv::Mat uv(ai_height / 2, ai_width / 2, CV_8UC2, (void*)uv_copy.data());
                                cv::cvtColorTwoPlane(y, uv, bgr, cv::COLOR_YUV2BGR_NV12);
                            } else {
                                cv::Mat yuv(ai_height * 3 / 2, ai_width, CV_8UC1, (void*)y_copy.data());
                                cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
                            }

                            std::vector<uchar> buf_jpg;
                            cv::imencode(".jpg", bgr, buf_jpg);

                            gchar* b64_char = g_base64_encode(buf_jpg.data(), buf_jpg.size());
                            base64_image = std::string(b64_char);
                            g_free(b64_char);

                            std::cout << "[LPR_SINK] Image encoded in background thread! Base64 Size: " << base64_image.size() << " bytes." << std::endl;
                        } catch (const std::exception& e) {
                            std::cout << "[LPR_SINK] ERROR: OpenCV conversion failed in worker thread: " << e.what() << std::endl;
                        }
                    }

                    nlohmann::json inex_payload = nlohmann::json::parse(event_json_str);
                    inex_payload["images"] = nlohmann::json::array({
                        {
                            {"image_id", 0},
                            {"image_guid", inex_payload.value("transaction_guid", "unknown")},
                            {"image_encoding", "jpg"},
                            {"image_data", base64_image}
                        }
                    });

                    // Update UI locally
                    {
                        std::lock_guard<std::mutex> lock(m_events_mutex);
                        m_lpr_events.insert(m_lpr_events.begin(), inex_payload);
                        if (m_lpr_events.size() > 50) m_lpr_events.pop_back();
                    }

                    // Dispatch to target INEX server
                    std::string target_url = get_target_inex_url();
                    std::string json_str = inex_payload.dump();
                    
                    std::cout << "[LPR_SINK] Dispatching INEX Event to: " << target_url << std::endl;
                    
                    std::string command = "curl --max-time 3 -X POST -H \"Content-Type: application/json\" -d '" 
                                        + json_str + "' \"" + target_url + "\" > /dev/null 2>&1";

                    int status = std::system(command.c_str());
                    if (status != 0) {
                        std::cout << "[LPR_SINK] WARNING: Failed to post INEX event to external server." << std::endl;
                    }
                }).detach();
            }
        })
        .buildptr();

    // 5. Assemble full pipeline
    m_app_resources->pipeline = PipelineBuilder()
        // Vision Path
        .add_stage("frontend", configure_frontend(), StageType::SOURCE)
        .add_stage("encoder", configure_encoder_and_osd(DEFAULT_STREAM_4K_NAME))
        .add_stage("vision_tee", std::make_shared<TeeStage>("vision_tee", 2, false, false))
        .add_stage("webrtc_sink", webrtc_sink_stage, StageType::SINK)
        
        // AI Path
        .add_stage(tiling_pipeline)
        .add_stage(veh_attrs_pipeline)
        .add_stage(cls_pipeline)
        .add_stage(ocr_pipeline)
        .add_stage(event_engine_stage)
        .add_stage("lpr_event_sink", lpr_event_sink, StageType::SINK)
        
        // Connect Vision Path
        .connect_frontend("frontend", DEFAULT_STREAM_4K_NAME, "encoder")
        .connect("encoder", "vision_tee")
        .connect("vision_tee", "webrtc_sink")
        
        // Connect AI Path
        .connect_frontend("frontend", "sink2", "tiling_pipeline")
        .connect("tiling_pipeline", "vehicle_attributes_pipeline")
        .connect("vehicle_attributes_pipeline", "classification_pipeline")
        .connect("classification_pipeline", "ocr_pipeline")
        .connect("ocr_pipeline", "lpr_event_engine_post")
        .connect("lpr_event_engine_post", "lpr_event_sink")
        .build("LPRPipeline", true);
}

void LprPipeline::start() {
    build_pipeline();
    BasePipeline::start();
}

void LprPipeline::register_endpoints() {
    BasePipeline::register_endpoints();
    WEBSERVER_LOG_INFO("Registering LPR event endpoints");

    // Serve local event log to web interface
    m_resources.m_srv.Get("/api/v1/lpr-events", std::function<nlohmann::json()>([this]() {
        std::lock_guard<std::mutex> lock(m_events_mutex);
        return nlohmann::json(m_lpr_events); 
    }));
}

void LprPipeline::unregister_endpoints() {
    m_resources.m_srv.Unregister("/api/v1/lpr-events");
    BasePipeline::unregister_endpoints();
}