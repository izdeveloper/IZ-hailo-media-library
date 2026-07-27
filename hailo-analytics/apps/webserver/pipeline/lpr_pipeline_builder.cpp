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

static constexpr int LICENSE_PLATE_CLASS_ID = 4;
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
    cfg.aggregator_config.main_queue_size = 3;

    // NMS class indices (1-based): 1=person, 2=vehicle, 3=face, 4=license_plate
    // Skip person and face — keep only vehicle and license_plate
    cfg.detection_config.ai_config.nms_classes_filter_mask = {false, true, false, true};

    cfg.detection_config.ai_config.nms_score_threshold = 0.5f;

    std::string tiling_pipeline_name = name + "_tiling";
    auto tiling_pipeline_status = tiling::generate_tiling_detection_pipeline(tiling_pipeline_name, cfg);
    if (!tiling_pipeline_status.has_value())
    {
        return tl::make_unexpected(tiling_pipeline_status.error());
    }
    auto tiling_pipeline = tiling_pipeline_status.value();

    // Build the lightweight tracker stage for license plate detections
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
        .set_output_pool_size(25)
        .set_labels({"vehicle"})
        .set_main_sub_name(agg_stage_name)
        .set_sub_sub_name(ai_stage_name)   
        .buildptr();

    auto ai_stage = ai::HailortAsyncStageBuild::create()
        .set_stage_name(ai_stage_name)
        .set_hef_path("/home/root/apps/license_plate_recognition/resources/vehicle_attributes.hef")
        .set_batch_size(1)
        .set_queue_size(20)
        .set_output_pool_size(30)
        .set_group_id("device0")
        .set_job_limit(20)
        .buildptr();

    auto post_stage = ai::PostprocessStageBuild::create()
        .set_stage_name(post_stage_name)
        .set_so_path("/usr/lib/hailo-post-processes/libvehicle_attributes_post.so")
        .set_function_name_opt("filter") 
        .buildptr();

    auto agg_stage = cropping::AggregatorStageBuild::create()
        .set_stage_name(agg_stage_name)
        .set_main_inlet_name(crop_stage_name)
        .set_sub_inlet_name(post_stage_name)
        .buildptr();

    // WIRE IT TOGETHER (Added filter_stage here)
    pip_builder.add_stage(crop_stage, StageType::SOURCE)
               .add_stage(ai_stage)
               .add_stage(post_stage)
               .add_stage(agg_stage, StageType::SINK);

    // Connect filter output to crop input
    pip_builder.connect(crop_stage_name, agg_stage_name);
    pip_builder.connect(crop_stage_name, ai_stage_name);
    pip_builder.connect(ai_stage_name, post_stage_name);
    pip_builder.connect(post_stage_name, agg_stage_name);

    auto pipeline = pip_builder.build(name, true);

    // Filter stage is now the actual entrance of this sub-pipeline
    pipeline->set_in_stage(crop_stage);
    pipeline->set_out_stage(agg_stage);

    return pipeline;
}

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_classification_pipeline(
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
        .set_output_width(320)
        .set_output_height(320)
        .set_output_pool_size(20)
        .set_labels({"license_plate"})
        .set_main_sub_name(agg_stage_name)
        .set_sub_sub_name(ai_stage_name)   
        .buildptr();

    auto ai_stage = ai::HailortAsyncStageBuild::create()
        .set_stage_name(ai_stage_name)
        .set_hef_path("/home/root/apps/license_plate_recognition/resources/plate_state.hef")
        .set_batch_size(1)
        .set_queue_size(20)
        .set_output_pool_size(30)
        .set_group_id("device0")
        .set_job_limit(20)
        .buildptr();

    auto post_stage = ai::PostprocessStageBuild::create()
        .set_stage_name(post_stage_name)
        .set_so_path("/usr/lib/hailo-post-processes/libstate_cls_post.so") // /usr/lib/liboem_process.so.1
        .set_function_name_opt("filter") // state_filter
        .buildptr();

    auto agg_stage = cropping::AggregatorStageBuild::create()
        .set_stage_name(agg_stage_name)
        .set_main_inlet_name(crop_stage_name)
        .set_sub_inlet_name(post_stage_name)
        .buildptr();

    // WIRE IT TOGETHER (Added filter_stage here)
    pip_builder.add_stage(crop_stage, StageType::SOURCE)
               .add_stage(ai_stage)
               .add_stage(post_stage)
               .add_stage(agg_stage, StageType::SINK);

    // Connect filter output to crop input
    pip_builder.connect(crop_stage_name, agg_stage_name);
    pip_builder.connect(crop_stage_name, ai_stage_name);
    pip_builder.connect(ai_stage_name, post_stage_name);
    pip_builder.connect(post_stage_name, agg_stage_name);

    auto pipeline = pip_builder.build(name, true);

    // Filter stage is now the true entrance
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
    cfg.bbox_crop_config.queue_size = 1;
    cfg.aggregator_config.main_queue_size = 3;
    cfg.aggregator_config.sub_queue_size = 20;
    cfg.ocr_config.ai_config.queue_size = 20;
    cfg.ocr_config.post_config.queue_size = 20;

    cfg.ocr_config.ai_config.hef_path = "/home/root/apps/license_plate_recognition/resources/lprnet_nv12.hef";
    //cfg.ocr_config.post_config.so_path = "/usr/lib/liboem_process.so.1";
    //cfg.ocr_config.post_config.function_name = "ocr_filter";
    cfg.bbox_crop_config.output_width = 300;
    cfg.bbox_crop_config.output_height = 74;

    return lpr::generate_bbox_crop_ocr_pipeline(name, cfg);
}

} // namespace lpr_app
