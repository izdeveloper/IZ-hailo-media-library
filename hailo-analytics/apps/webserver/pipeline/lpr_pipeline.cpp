#include "lpr_pipeline.hpp"
#include "lpr_pipeline_builder.hpp"
#include "common/common.hpp"
#include "hailo_analytics/pipeline/sinks/app_sink_stage.hpp"
#include "hailo_analytics/pipeline/routing/tee_stage.hpp"
#include "hailo_analytics/pipeline/ai/postprocess_stage.hpp"

using namespace hailo_analytics::pipeline;
using namespace hailo_analytics::pipeline::sinks;
using namespace hailo_analytics::pipeline::routing;
using namespace hailo_analytics::pipeline::ai;
using namespace webserver::pipeline;
using namespace webserver::resources;

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

    // 2. LPR AI sub-pipelines from builder
    auto tiling_pipeline = lpr_app::build_tiling_pipeline("tiling_pipeline", lpr_app::TrackingMode::BALANCED).value();
    auto veh_attrs_pipeline = lpr_app::build_vehicle_attributes_pipeline("vehicle_attributes_pipeline").value();
    auto cls_pipeline = lpr_app::build_classification_pipeline("classification_pipeline").value();
    auto ocr_pipeline = lpr_app::build_ocr_pipeline("ocr_pipeline").value();
    
    // Event Engine postprocess stage
    auto event_engine_stage = PostprocessStageBuild::create()
        .set_stage_name("lpr_event_engine_post")
        .set_so_path("/usr/lib/hailo-post-processes/liblpr_event_analytics_post.so")
        .set_function_name_opt("filter")
        .set_queue_size_opt(5)
        .set_leaky_opt(false)
        .buildptr();

    // 3. Assemble full pipeline
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
        .add_stage(event_engine_stage, StageType::SINK)
        
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
        .build("LPRPipeline", true);
}

void LprPipeline::start() {
    build_pipeline();
    BasePipeline::start();
}