#include "lpr_pipeline_builder.hpp"
#include "hailo_analytics/pipeline/ai/lightweight_tracker_stage.hpp"
#include "hailo_analytics/pipeline/core/pipeline_builder.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace lpr_app
{

namespace tiling = hailo_analytics::analytics::tiling;
namespace lpr = hailo_analytics::analytics::license_plate_recognition;

using hailo_analytics::pipeline::PipelineBuilder;
using hailo_analytics::pipeline::StageType;
using hailo_analytics::pipeline::ai::LightweightTrackerStageBuild;

// FIXED: License plate is Class Index 2 in YAML + 1 offset = Class ID 3
static constexpr int LICENSE_PLATE_CLASS_ID = 3;

static constexpr std::string_view TRACKING_MODE_FAST = "fast";
static constexpr std::string_view TRACKING_MODE_SLOW = "slow";
static constexpr std::string_view TRACKING_MODE_BALANCED = "balanced";

struct TrackerConfig
{
    int grace_period;
    float smooth_alpha;
    float weighted_average_decay;
    float iou_threshold;
};

static constexpr TrackerConfig FAST_CONFIG{1, 0.7f, 0.3f, 0.3f};
static constexpr TrackerConfig SLOW_CONFIG{5, 0.3f, 0.6f, 0.8f};
static constexpr TrackerConfig BALANCED_CONFIG{2, 0.5f, 0.4f, 0.5f};

static constexpr const TrackerConfig &get_tracker_config(TrackingMode mode)
{
    switch (mode)
    {
    case TrackingMode::FAST:
        return FAST_CONFIG;
    case TrackingMode::SLOW:
        return SLOW_CONFIG;
    case TrackingMode::BALANCED:
        return BALANCED_CONFIG;
    }
    __builtin_unreachable();
}

TrackingMode tracking_mode_from_string(const std::string &mode_str)
{
    std::string lower = mode_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == TRACKING_MODE_FAST)
    {
        return TrackingMode::FAST;
    }
    if (lower == TRACKING_MODE_SLOW)
    {
        return TrackingMode::SLOW;
    }
    if (lower == TRACKING_MODE_BALANCED)
    {
        return TrackingMode::BALANCED;
    }
    throw std::invalid_argument("Invalid tracking mode: '" + mode_str +
                                "'. Expected: " + std::string(TRACKING_MODE_SLOW) + ", " +
                                std::string(TRACKING_MODE_FAST) + ", or " + std::string(TRACKING_MODE_BALANCED));
}

static void apply_tracking_mode(LightweightTrackerStageBuild::Builder &builder, TrackingMode mode)
{
    const auto &config = get_tracker_config(mode);
    builder.set_grace_period(config.grace_period)
        .set_smooth_alpha(config.smooth_alpha)
        .set_weighted_average_decay(config.weighted_average_decay)
        .set_iou_threshold(config.iou_threshold);
}

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_tiling_pipeline(
    const std::string &name, TrackingMode mode)
{
    // Build the tiling detection sub-pipeline
    auto cfg = tiling::base_config();
    cfg.tiling_config.queue_size = 1;
    cfg.tiling_config.leaky = true;
    cfg.aggregator_config.main_queue_size = 3;

    cfg.tiling_config.output_width = 640;
    cfg.tiling_config.output_height = 640;

    // --- OVERRIDE DETECTION MODEL AND POST-PROCESS CONFIG HERE ---
    cfg.detection_config.ai_config.hef_path = 
        "/home/root/apps/license_plate_recognition/resources/yolov8_detection.hef";
    cfg.detection_config.post_config.so_path = 
        "/usr/lib/hailo-post-processes/libyolo_hailortpp_post.so";
    cfg.detection_config.post_config.function_name = "filter";
    cfg.detection_config.post_config.config_path = 
        "/home/root/apps/license_plate_recognition/resources/yolov8_lpr_detection.json";

    // Set hardware NMS threshold to 0.20 to match model score threshold
    cfg.detection_config.ai_config.nms_score_threshold = 0.35f;

    // Keep all 3 classes: four_wheels, two_wheels, license_plate
    cfg.detection_config.ai_config.nms_classes_filter_mask = {true, true, true};

    std::string tiling_pipeline_name = name + "_tiling";
    auto tiling_pipeline_status = tiling::generate_tiling_detection_pipeline(tiling_pipeline_name, cfg);
    if (!tiling_pipeline_status.has_value())
    {
        return tl::make_unexpected(tiling_pipeline_status.error());
    }
    auto tiling_pipeline = tiling_pipeline_status.value();

    // Build the lightweight tracker stage for license plate detections (Class ID 3)
    std::string tracker_stage_name = "lpr_tracker";
    auto tracker_builder = LightweightTrackerStageBuild::create()
                               .set_stage_name(tracker_stage_name)
                               .set_queue_size_opt(1)
                               .set_leaky_opt(false)
                               .set_classification_ids({LICENSE_PLATE_CLASS_ID})
                               .set_add_tracking_id(true, LICENSE_PLATE_CLASS_ID)
                               .set_trace_opt(true);
    apply_tracking_mode(tracker_builder, mode);
    auto tracker_stage = tracker_builder.buildptr();

    // Wrap tiling pipeline and tracker into a single pipeline
    PipelineBuilder pip_builder;
    pip_builder.add_stage(tiling_pipeline, StageType::SOURCE)
        .add_stage(tracker_stage, StageType::SINK)
        .connect(tiling_pipeline_name, tracker_stage_name);
    auto pipeline = pip_builder.build(name, true);

    // Set input/output stages so the wrapper pipeline can be connected in the outer pipeline
    pipeline->set_in_stage(tiling_pipeline);
    pipeline->set_out_stage(tracker_stage);

    return pipeline;
}

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_vehicle_attributes_pipeline(
    const std::string &name)
{
    using namespace hailo_analytics::pipeline;
    PipelineBuilder pip_builder;

    std::string filter_stage_name = name + "_guard"; // Unique name
    std::string crop_stage_name = name + "_crop";
    std::string ai_stage_name = name + "_ai";
    std::string post_stage_name = name + "_post";
    std::string agg_stage_name = name + "_agg";

    auto crop_stage = cropping::BBoxCropStageBuild::create()
        .set_stage_name(crop_stage_name)
        .set_input_width(1920) 
        .set_input_height(1080)
        .set_output_width(224)
        .set_output_height(224)
        .set_output_pool_size(30)
        .set_pool_mode_opt(StagePoolMode::BLOCKING)
        .set_labels({"vehicle", "two_wheels"})
        .set_main_sub_name(agg_stage_name)
        .set_sub_sub_name(ai_stage_name)   
        .buildptr();

    auto ai_stage = ai::HailortAsyncStageBuild::create()
        .set_stage_name(ai_stage_name)
        .set_hef_path("/home/root/apps/license_plate_recognition/resources/vehicle_attributes.hef")
        .set_batch_size(1)
        .set_queue_size(5)
        .set_output_pool_size(30)
        .set_pool_mode_opt(StagePoolMode::BLOCKING)
        .set_group_id("device0")
        .set_job_limit(20)
        .buildptr();

    auto post_stage = ai::PostprocessStageBuild::create()
        .set_stage_name(post_stage_name)
        .set_so_path("/usr/lib/hailo-post-processes/libvehicle_attributes_post.so")
        .set_function_name_opt("filter")
        .set_queue_size_opt(5)
        .set_leaky_opt(false)
        .buildptr();

    auto agg_stage = cropping::AggregatorStageBuild::create()
        .set_stage_name(agg_stage_name)
        .set_main_inlet_name(crop_stage_name)
        .set_sub_inlet_name(post_stage_name)
        .set_sub_queue_size(2)
        .set_sub_leaky(false)
        .buildptr();

    pip_builder.add_stage(crop_stage, StageType::SOURCE)
               .add_stage(ai_stage)
               .add_stage(post_stage)
               .add_stage(agg_stage, StageType::SINK);

    pip_builder.connect(crop_stage_name, agg_stage_name);
    pip_builder.connect(crop_stage_name, ai_stage_name);
    pip_builder.connect(ai_stage_name, post_stage_name);
    pip_builder.connect(post_stage_name, agg_stage_name);

    auto pipeline = pip_builder.build(name, true);

    pipeline->set_in_stage(crop_stage);
    pipeline->set_out_stage(agg_stage);

    return pipeline;
}

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_classification_pipeline(
    const std::string &name, bool is_usa = false)
{
    using namespace hailo_analytics::pipeline;
    PipelineBuilder pip_builder;

    std::string filter_stage_name = name + "_guard";
    std::string crop_stage_name = name + "_crop";
    std::string ai_stage_name = name + "_ai";
    std::string post_stage_name = name + "_post";
    std::string agg_stage_name = name + "_agg";

    // Set paths and dimensions based on USA pipeline flag
    std::string hef_path = is_usa ? 
        "/home/root/apps/license_plate_recognition/resources/plate_state_usa.hef" : 
        "/home/root/apps/license_plate_recognition/resources/plate_state.hef";

    std::string post_func = is_usa ? "filter_usa" : "filter";

    int crop_w = is_usa ? 192 : 320;
    int crop_h = is_usa ? 96 : 320;

    auto crop_stage = cropping::BBoxCropStageBuild::create()
        .set_stage_name(crop_stage_name)
        .set_input_width(1920) 
        .set_input_height(1080)
        .set_output_width(crop_w)
        .set_output_height(crop_h)
        .set_output_pool_size(15)
        .set_pool_mode_opt(StagePoolMode::BLOCKING)
        .set_labels({"license_plate"})
        .set_letterbox_opt()
        .set_main_sub_name(agg_stage_name)
        .set_sub_sub_name(ai_stage_name)   
        .buildptr();

    auto ai_stage = ai::HailortAsyncStageBuild::create()
        .set_stage_name(ai_stage_name)
        .set_hef_path(hef_path)
        .set_batch_size(1)
        .set_queue_size(5)
        .set_output_pool_size(15)
        .set_pool_mode_opt(StagePoolMode::BLOCKING)
        .set_group_id("device0")
        .set_job_limit(10)
        .buildptr();

    auto post_stage = ai::PostprocessStageBuild::create()
        .set_stage_name(post_stage_name)
        .set_so_path("/usr/lib/hailo-post-processes/libstate_cls_post.so")
        .set_function_name_opt(post_func)
        .set_queue_size_opt(5)
        .set_leaky_opt(false)
        .buildptr();

    auto agg_stage = cropping::AggregatorStageBuild::create()
        .set_stage_name(agg_stage_name)
        .set_main_inlet_name(crop_stage_name)
        .set_sub_inlet_name(post_stage_name)
        .set_sub_queue_size(5)
        .set_sub_leaky(false)
        .buildptr();

    pip_builder.add_stage(crop_stage, StageType::SOURCE)
               .add_stage(ai_stage)
               .add_stage(post_stage)
               .add_stage(agg_stage, StageType::SINK);

    pip_builder.connect(crop_stage_name, agg_stage_name);
    pip_builder.connect(crop_stage_name, ai_stage_name);
    pip_builder.connect(ai_stage_name, post_stage_name);
    pip_builder.connect(post_stage_name, agg_stage_name);

    auto pipeline = pip_builder.build(name, true);

    pipeline->set_in_stage(crop_stage);
    pipeline->set_out_stage(agg_stage);

    return pipeline;
}

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_ocr_pipeline(
    const std::string &name)
{
    auto cfg = lpr::base_config();

    cfg.bbox_crop_config.input_width = 1920;
    cfg.bbox_crop_config.input_height = 1080;
    cfg.bbox_crop_config.queue_size = 2;
    cfg.bbox_crop_config.output_pool_size = 20;
    cfg.bbox_crop_config.pool_mode = hailo_analytics::pipeline::StagePoolMode::BLOCKING;

    cfg.aggregator_config.main_queue_size = 3;
    cfg.aggregator_config.sub_queue_size = 5;

    cfg.ocr_config.ai_config.queue_size = 5;
    cfg.ocr_config.post_config.queue_size = 5;
    cfg.ocr_config.ai_config.output_pool_size = 20;
    cfg.ocr_config.ai_config.pool_mode = hailo_analytics::pipeline::StagePoolMode::BLOCKING;

    cfg.ocr_config.ai_config.hef_path = "/home/root/apps/license_plate_recognition/resources/lprnet_nv12.hef";
    cfg.bbox_crop_config.output_width = 300;
    cfg.bbox_crop_config.output_height = 74;

    return lpr::generate_bbox_crop_ocr_pipeline(name, cfg);
}

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_ocr_usa_pipeline(
    const std::string &name)
{
    auto cfg = lpr::base_config();
    cfg.bbox_crop_config.input_width = 1920;
    cfg.bbox_crop_config.input_height = 1080;
    cfg.bbox_crop_config.queue_size = 2;
    cfg.aggregator_config.main_queue_size = 3;
    cfg.aggregator_config.sub_queue_size = 2;
    cfg.ocr_config.ai_config.queue_size = 5;
    cfg.ocr_config.post_config.queue_size = 5;
    cfg.ocr_config.ai_config.output_pool_size = 20;

    cfg.bbox_crop_config.output_pool_size = 20;
    cfg.bbox_crop_config.pool_mode = hailo_analytics::pipeline::StagePoolMode::LEAKY;
    cfg.ocr_config.ai_config.pool_mode = hailo_analytics::pipeline::StagePoolMode::LEAKY;

    // --- USA Model Specific Overrides ---
    // 1. Path to your newly compiled HEF
    cfg.ocr_config.ai_config.hef_path = "/home/root/apps/license_plate_recognition/resources/usa_ocr.hef";
    
    // 2. Post-processing function name in libocr_post.so
    cfg.ocr_config.post_config.function_name = "filter_usa"; 
    
    // 3. USA OCR character map JSON
    cfg.ocr_config.post_config.config_path = "/home/root/apps/license_plate_recognition/resources/usa_ocr_chars.json";

    // 4. Crop dimensions matching your USA HEF input: NHWC(96x192x1)
    cfg.bbox_crop_config.output_width = 192;
    cfg.bbox_crop_config.output_height = 96;

    return lpr::generate_bbox_crop_ocr_pipeline(name, cfg);
}

} // namespace lpr_app