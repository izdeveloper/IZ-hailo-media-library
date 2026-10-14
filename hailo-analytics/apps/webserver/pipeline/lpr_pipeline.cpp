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
#include <mutex>
#include <queue>
#include <deque>
#include <condition_variable>
#include <sys/mman.h>
#include <cstdio>
#include <httplib.h>

using namespace hailo_analytics::pipeline;
using namespace hailo_analytics::pipeline::sinks;
using namespace hailo_analytics::pipeline::routing;
using namespace hailo_analytics::pipeline::ai;
using namespace webserver::pipeline;
using namespace webserver::resources;

static const std::string CONFIG_JSON_PATH = "/home/root/apps/license_plate_recognition/resources/event_config_ip.json";
static const std::string DB_FILE_PATH = "/home/root/apps/webserver/resources/configs/events_db.json";
static std::mutex g_db_mutex;

// In-Memory cache to make 1-second live polling lightning fast (0 disk reads)
static std::deque<nlohmann::json> g_recent_events;
static std::unordered_map<int, std::string> g_temp_candidate_images;
static std::mutex g_candidate_mutex;

static size_t g_max_saved_events = 1000;

static void save_event_to_db_safely(const nlohmann::json &new_event)
{
    std::lock_guard<std::mutex> lock(g_db_mutex);

    // 1. Update in-memory cache for the live UI polling
    g_recent_events.push_back(new_event);
    if (g_recent_events.size() > 20)
    {
        g_recent_events.pop_front();
    }

    // 2. Read existing events from disk
    nlohmann::json db = nlohmann::json::array();
    std::ifstream infile(DB_FILE_PATH);
    if (infile.is_open())
    {
        try
        {
            infile >> db;
        }
        catch (...)
        {
        }
        infile.close();
    }

    // 3. Append the new event and enforce the server-side limit
    db.push_back(new_event);
    while (db.size() > g_max_saved_events)
    {
        db.erase(0); // Drop the oldest events
    }

    // 4. Save back to disk
    std::ofstream outfile(DB_FILE_PATH);
    outfile << db.dump();
}

// Persistent AppSinkStage wrapper that prevents deinit() from wiping process callbacks on stream restart
class PersistentAppSinkStage : public AppSinkStage
{
private:
    std::function<void(hailo_analytics::pipeline::BufferPtr)> m_func;

public:
    PersistentAppSinkStage(std::string name, size_t queue_size = 1, bool leaky = false, bool trace = true)
        : AppSinkStage(name, queue_size, leaky, trace) {}

    void set_process_function(std::function<void(hailo_analytics::pipeline::BufferPtr)> func)
    {
        m_func = func;
        configure(m_func);
    }

    AppStatus init() override
    {
        if (m_func)
        {
            configure(m_func); // Restore process function before AppSinkStage::init() checks it
        }
        return AppSinkStage::init();
    }
};

// =========================================================================
// Bounded Worker Queue
// =========================================================================
class EventUploader
{
public:
    static EventUploader &get_instance()
    {
        static EventUploader instance;
        return instance;
    }

    void enqueue_task(std::function<void()> task)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_queue.size() >= MAX_QUEUE_SIZE)
            {
                m_queue.pop();
                std::cerr << "[LPR_SINK] WARNING: Queue full. Dropping oldest event." << std::endl;
            }
            m_queue.push(std::move(task));
        }
        m_cv.notify_one();
    }

private:
    EventUploader() : m_stop(false)
    {
        m_worker = std::thread([this]()
                               {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_cv.wait(lock, [this] { return m_stop || !m_queue.empty(); });
                    if (m_stop && m_queue.empty()) return;
                    task = std::move(m_queue.front());
                    m_queue.pop();
                }
                try {
                    task();
                } catch (const std::exception& e) {
                    std::cerr << "[LPR_SINK] Task Exception: " << e.what() << std::endl;
                }
            } });
    }

    ~EventUploader()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        if (m_worker.joinable())
            m_worker.join();
    }

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<std::function<void()>> m_queue;
    std::thread m_worker;
    bool m_stop;
    const size_t MAX_QUEUE_SIZE = 50;
};

static std::string get_target_inex_url()
{
    std::ifstream file(CONFIG_JSON_PATH);
    std::string ip = "127.0.0.1";
    std::string port = "8080";
    if (file.is_open())
    {
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        std::regex ip_regex("\"http_endpoint_ip\"\\s*:\\s*\"([^\"]+)\"");
        std::regex port_regex("\"http_endpoint_port\"\\s*:\\s*(\\d+)");
        std::smatch match;
        if (std::regex_search(content, match, ip_regex) && match.size() > 1)
            ip = match[1].str();
        if (std::regex_search(content, match, port_regex) && match.size() > 1)
            port = match[1].str();
    }
    return "http://" + ip + ":" + port + "/api/v1/lpr-events?cmd=uploadevent&api_version=1.7";
}

LprPipeline::LprPipeline(webserver::resources::ResourceRepository &resources, MediaLibrary &media_library,
                         RTPConverterStage &webrtc_stage, Architecture platform, pipeline_t pipeline_type)
    : BasePipeline(resources, media_library, webrtc_stage, platform, ProfileType::Daylight,
                   {ProfileType::Daylight, ProfileType::LowlightBayer}), m_pipeline_type(pipeline_type) {}

std::string LprPipeline::pipeline_name() const { return (m_pipeline_type == pipeline_t::LPR_USA) ? "LPR_USA" : "LPR"; }
std::string LprPipeline::get_profile_name_by_type(ProfileType type) const
{
    switch (type)
    {
    case ProfileType::Daylight:
        return "Daylight_FaceLandmarks";
    case ProfileType::LowlightBayer:
        return "Lowlight_Bayer_FaceLandmarks";
    case ProfileType::Lowlight:
        return "Lowlight_FaceLandmarks";
    case ProfileType::HighDynamicRange:
        return "High_Dynamic_Range_FaceLandmarks";
    default:
        throw std::runtime_error("Profile type not supported in LPR Pipeline");
    }
}

ProfileType LprPipeline::get_profile_type_by_name(const std::string &name) const
{
    if (name == "Daylight_FaceLandmarks")
        return ProfileType::Daylight;
    else if (name == "Lowlight_Bayer_FaceLandmarks")
        return ProfileType::LowlightBayer;
    else if (name == "Lowlight_FaceLandmarks")
        return ProfileType::Lowlight;
    else if (name == "High_Dynamic_Range_FaceLandmarks")
        return ProfileType::HighDynamicRange;
    else
        throw std::runtime_error("Profile name not supported in LPR Pipeline: " + name);
}

void LprPipeline::start()
{
    WEBSERVER_LOG_INFO("Starting LPR pipeline");
    build_pipeline();
    BasePipeline::start();
}

void LprPipeline::register_endpoints()
{
    BasePipeline::register_endpoints();

    // 1. Full History Endpoint (RAW TEXT STREAMING)
    m_resources.m_srv.GetHtml("/api/v1/saved-events", std::function<std::string()>([]()
                                                                                   {
        std::lock_guard<std::mutex> lock(g_db_mutex);
        std::ifstream infile(DB_FILE_PATH, std::ios::in | std::ios::binary);
        if (infile.is_open()) {
            std::string contents;
            infile.seekg(0, std::ios::end);
            contents.resize(infile.tellg());
            infile.seekg(0, std::ios::beg);
            infile.read(&contents[0], contents.size());
            infile.close();
            return contents;
        }
        return std::string("[]"); }));

    // 2. Clear Database Endpoint
    m_resources.m_srv.Delete("/api/v1/saved-events", std::function<nlohmann::json(const nlohmann::json &)>([](const nlohmann::json &j_body)
                                                                                                           {
        std::lock_guard<std::mutex> lock(g_db_mutex);
        std::ofstream outfile(DB_FILE_PATH);
        outfile << "[]";
        outfile.close();
        g_recent_events.clear(); // Empty RAM cache
        return nlohmann::json{{"status", "success"}}; }));

    // 3. Live Polling Endpoint (NO DISK READS)
    m_resources.m_srv.Get("/api/v1/lpr-events", std::function<nlohmann::json()>([]()
                                                                                {
        std::lock_guard<std::mutex> lock(g_db_mutex);
        nlohmann::json recent = nlohmann::json::array();
        for (const auto& ev : g_recent_events) {
            recent.push_back(ev);
        }
        return recent; }));

    // Event Configuration Endpoints
    m_resources.m_srv.Get("/api/v1/event-config", std::function<nlohmann::json()>([]()
                                                                                  { return nlohmann::json{{"max_events", g_max_saved_events}}; }));

    m_resources.m_srv.Post("/api/v1/event-config", std::function<nlohmann::json(const nlohmann::json &)>([](const nlohmann::json &j_body)
                                                                                                         {
        if (j_body.contains("max_events")) {
            std::lock_guard<std::mutex> lock(g_db_mutex);
            g_max_saved_events = j_body["max_events"].get<size_t>();
        }
        return nlohmann::json{{"status", "success"}}; }));
}

void LprPipeline::unregister_endpoints()
{
    m_resources.m_srv.Unregister("/api/v1/saved-events");
    m_resources.m_srv.Unregister("/api/v1/lpr-events");
    BasePipeline::unregister_endpoints();
}

// =========================================================================
// ASYNC ENCODING HELPERS AND STRUCTURES (Placed globally outside functions)
// =========================================================================

struct RawFrameData {
    std::vector<uint8_t> y_copy;
    std::vector<uint8_t> uv_copy;
    bool is_split_plane = false;
    int width = 0;
    int height = 0;
};

static std::string encode_raw_frame_to_base64(const RawFrameData &raw) {
    if (raw.y_copy.empty()) return "";

    try {
        cv::Mat bgr;
        if (raw.is_split_plane && !raw.uv_copy.empty()) {
            cv::Mat y(raw.height, raw.width, CV_8UC1, (void*)raw.y_copy.data());
            cv::Mat uv(raw.height / 2, raw.width / 2, CV_8UC2, (void*)raw.uv_copy.data());
            cv::cvtColorTwoPlane(y, uv, bgr, cv::COLOR_YUV2BGR_NV12);
        } else {
            cv::Mat yuv(raw.height * 3 / 2, raw.width, CV_8UC1, (void*)raw.y_copy.data());
            cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
        }
        
        cv::Mat resized_bgr;
        cv::resize(bgr, resized_bgr, cv::Size(1280, 720));
        
        std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 70};
        std::vector<uchar> buf_jpg;
        cv::imencode(".jpg", resized_bgr, buf_jpg, encode_params);
        gchar* b64_char = g_base64_encode(buf_jpg.data(), buf_jpg.size());
        std::string base64_image(b64_char);
        g_free(b64_char);
        return base64_image;
    } catch (const std::exception& e) {
        std::cerr << "[LPR_SINK] Async encoding error: " << e.what() << std::endl;
        return "";
    }
}

static std::unordered_map<int, RawFrameData> g_temp_candidate_raw_frames;

// =========================================================================

void LprPipeline::build_pipeline()
{
    WEBSERVER_LOG_INFO("Building LPR pipeline");

    m_app_resources->valve_stage = std::make_shared<ValveStage>("valve", 1,true);
    m_app_resources->freeze_stage = std::make_shared<FreezeStage>("freeze", 1,true);

    auto webrtc_sink_stage = AppSinkStageBuild::create()
                                 .set_stage_name("webrtc_sink")
                                 .set_queue_size_opt(1)
                                 .set_leaky_opt(true)
                                 .set_process_func([&](hailo_analytics::pipeline::BufferPtr buf)
                                                   { 
            if (!buf) return;
            m_webrtc_stage.process(buf); })
                                 .buildptr();

    auto tiling_pipeline = lpr_app::build_tiling_pipeline("tiling_pipeline", lpr_app::TrackingMode::BALANCED).value();
    auto veh_attrs_pipeline = lpr_app::build_vehicle_attributes_pipeline("vehicle_attributes_pipeline").value();
    auto cls_pipeline = lpr_app::build_classification_pipeline("classification_pipeline", m_pipeline_type == pipeline_t::LPR_USA).value();
    auto ocr_pipeline = (m_pipeline_type == pipeline_t::LPR_USA) ? lpr_app::build_ocr_usa_pipeline("ocr_pipeline").value() : lpr_app::build_ocr_pipeline("ocr_pipeline").value();

    auto event_engine_stage = PostprocessStageBuild::create()
                                  .set_stage_name("lpr_event_engine_post")
                                  .set_so_path("/usr/lib/hailo-post-processes/liblpr_event_analytics_post.so")
                                  .set_function_name_opt("filter")
                                  .set_queue_size_opt(5)
                                  .set_leaky_opt(false)
                                  .buildptr();

    int ai_width = 1920;
    int ai_height = 1080;
    auto output_streams = m_app_resources->media_library.m_frontend->get_outputs_streams();
    if (output_streams.has_value())
    {
        for (const auto &stream : output_streams.value())
        {
            if (stream.id == "sink2")
            {
                ai_width = stream.width;
                ai_height = stream.height;
                break;
            }
        }
    }

    auto lpr_event_sink = AppSinkStageBuild::create()
        .set_stage_name("lpr_event_sink")
        .set_queue_size_opt(5)
        .set_leaky_opt(true)
        .set_process_func([ai_width, ai_height](hailo_analytics::pipeline::BufferPtr buf) {
            if (!buf) return;

            HailoROIPtr roi = buf->get_roi();
            if (!roi) return;

            std::string event_json_str = "";
            std::string candidate_json_str = "";

            for (const auto& obj : roi->get_objects()) {
                if (obj->get_type() == HAILO_CLASSIFICATION) {
                    auto cls = std::dynamic_pointer_cast<HailoClassification>(obj);
                    if (cls) {
                        if (cls->get_classification_type() == "inex_event") {
                            event_json_str = cls->get_label();
                        } else if (cls->get_classification_type() == "inex_candidate") {
                            candidate_json_str = cls->get_label();
                        }
                    }
                }
            }

            auto copy_raw_frame = [&](hailo_analytics::pipeline::BufferPtr frame_buf) -> RawFrameData {
                RawFrameData raw;
                raw.width = ai_width;
                raw.height = ai_height;
                auto ml_buf = frame_buf->get_buffer();

                if (ml_buf) {
                    int fd_y = ml_buf->get_plane_fd(0);
                    int fd_uv = ml_buf->get_plane_fd(1);
                    if (fd_y >= 0) {
                        struct dma_buf_sync sync = { 0 };
                        sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
                        ioctl(fd_y, DMA_BUF_IOCTL_SYNC, &sync);
                        if (fd_uv >= 0) ioctl(fd_uv, DMA_BUF_IOCTL_SYNC, &sync);

                        off_t size_y = lseek(fd_y, 0, SEEK_END); lseek(fd_y, 0, SEEK_SET);
                        size_t map_size_y = (size_y > 0) ? size_y : (ai_width * ai_height);
                        void* data_y = mmap(NULL, map_size_y, PROT_READ, MAP_SHARED, fd_y, 0);

                        void* data_uv = MAP_FAILED;
                        size_t map_size_uv = 0;
                        if (fd_uv >= 0) {
                            off_t size_uv = lseek(fd_uv, 0, SEEK_END); lseek(fd_uv, 0, SEEK_SET);
                            map_size_uv = (size_uv > 0) ? size_uv : (ai_width * ai_height / 2);
                            data_uv = mmap(NULL, map_size_uv, PROT_READ, MAP_SHARED, fd_uv, 0);
                        }

                        if (data_y != MAP_FAILED) {
                            if (fd_uv >= 0 && data_uv != MAP_FAILED) {
                                raw.is_split_plane = true;
                                raw.y_copy.assign((uint8_t*)data_y, (uint8_t*)data_y + (ai_width * ai_height));
                                raw.uv_copy.assign((uint8_t*)data_uv, (uint8_t*)data_uv + (ai_width * ai_height / 2));
                            } else {
                                raw.y_copy.assign((uint8_t*)data_y, (uint8_t*)data_y + (ai_width * ai_height * 3 / 2));
                            }
                        }

                        if (data_y != MAP_FAILED) munmap(data_y, map_size_y);
                        if (data_uv != MAP_FAILED) munmap(data_uv, map_size_uv);

                        sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
                        ioctl(fd_y, DMA_BUF_IOCTL_SYNC, &sync);
                        if (fd_uv >= 0) ioctl(fd_uv, DMA_BUF_IOCTL_SYNC, &sync);
                    }
                }
                return raw;
            };

            // 1. Process candidate frame (fast raw memory copy)
            if (!candidate_json_str.empty()) {
                try {
                    auto cand_json = nlohmann::json::parse(candidate_json_str);
                    int track_id = cand_json.value("tracking_id", -1);
                    int logical_id = cand_json.value("logical_track_id", track_id);

                    RawFrameData raw = copy_raw_frame(buf);
                    if (!raw.y_copy.empty() && track_id != -1) {
                        std::lock_guard<std::mutex> lock(g_candidate_mutex);
                        g_temp_candidate_raw_frames[track_id] = raw;
                        g_temp_candidate_raw_frames[logical_id] = raw;

                        if (g_temp_candidate_raw_frames.size() > 50) {
                            g_temp_candidate_raw_frames.erase(g_temp_candidate_raw_frames.begin());
                        }
                    }
                } catch (...) {}
            }

            // 2. Process event dispatch (offload JPEG encoding & HTTP post to worker)
            if (!event_json_str.empty()) {
                nlohmann::json new_db_entry;
                try {
                    new_db_entry = nlohmann::json::parse(event_json_str);
                } catch (...) {
                    new_db_entry = nlohmann::json::object();
                }

                int track_id = new_db_entry.value("tracking_id", -1);
                RawFrameData raw_to_encode;

                {
                    std::lock_guard<std::mutex> lock(g_candidate_mutex);
                    if (track_id != -1 && g_temp_candidate_raw_frames.count(track_id)) {
                        raw_to_encode = g_temp_candidate_raw_frames[track_id];
                        g_temp_candidate_raw_frames.erase(track_id);
                    }
                }

                if (raw_to_encode.y_copy.empty()) {
                    raw_to_encode = copy_raw_frame(buf);
                }

                EventUploader::get_instance().enqueue_task([new_db_entry, raw_to_encode]() mutable {
                    std::string base64_image = encode_raw_frame_to_base64(raw_to_encode);

                    new_db_entry["images"] = nlohmann::json::array({{
                        {"image_data", base64_image},
                        {"image_encoding", "jpeg"}
                    }});

                    save_event_to_db_safely(new_db_entry);

                    std::string target_upload_url = get_target_inex_url();
                    try {
                        size_t host_start = target_upload_url.find("://");
                        if (host_start != std::string::npos) {
                            host_start += 3;
                            size_t path_start = target_upload_url.find('/', host_start);

                            std::string host = target_upload_url.substr(0, path_start);
                            std::string path = (path_start != std::string::npos) ? target_upload_url.substr(path_start) : "/";

                            httplib::Client cli(host.c_str());
                            cli.set_connection_timeout(2, 0);
                            cli.set_read_timeout(2, 0);

                            cli.Post(path.c_str(), new_db_entry.dump(), "application/json");
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[LPR_SINK] HTTP Client Exception: " << e.what() << std::endl;
                    }
                });
            }
        })
        .buildptr();

    m_app_resources->pipeline = hailo_analytics::pipeline::PipelineBuilder()
                                    .add_stage("frontend", configure_frontend(), hailo_analytics::pipeline::StageType::SOURCE)
                                    .add_stage("freeze", m_app_resources->freeze_stage)
                                    .add_stage("valve", m_app_resources->valve_stage)
                                    .add_stage("encoder", configure_encoder_and_osd(DEFAULT_STREAM_4K_NAME))
                                    .add_stage("vision_tee", std::make_shared<TeeStage>("vision_tee", 2, true, false))
                                    .add_stage("udp", configure_udp(DEFAULT_STREAM_4K_NAME), hailo_analytics::pipeline::StageType::SINK)
                                    .add_stage("webrtc_sink", webrtc_sink_stage, hailo_analytics::pipeline::StageType::SINK)
                                    .add_stage(tiling_pipeline)
                                    .add_stage(veh_attrs_pipeline)
                                    .add_stage(cls_pipeline)
                                    .add_stage(ocr_pipeline)
                                    .add_stage("lpr_event_engine_post", event_engine_stage)
                                    .add_stage("lpr_event_sink", lpr_event_sink, hailo_analytics::pipeline::StageType::SINK)
                                    // Vision path connections
                                    .connect_frontend("frontend", DEFAULT_STREAM_4K_NAME, "freeze")
                                    .connect("freeze", "valve")
                                    .connect("valve", "encoder")
                                    .connect("encoder", "vision_tee")
                                    .connect("vision_tee", "udp")
                                    .connect("vision_tee", "webrtc_sink")
                                    // AI Path connections
                                    .connect_frontend("frontend", "sink2", "tiling_pipeline")
                                    .connect("tiling_pipeline", "vehicle_attributes_pipeline")
                                    .connect("vehicle_attributes_pipeline", "classification_pipeline")
                                    //.connect("tiling_pipeline", "classification_pipeline")
                                    .connect("classification_pipeline", "ocr_pipeline")
                                    .connect("ocr_pipeline", "lpr_event_engine_post")
                                    .connect("lpr_event_engine_post", "lpr_event_sink")
                                    .build("LprPipeline");

    WEBSERVER_LOG_INFO("LPR pipeline built successfully");
}