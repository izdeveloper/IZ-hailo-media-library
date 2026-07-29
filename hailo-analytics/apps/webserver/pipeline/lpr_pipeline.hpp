#pragma once
#include "pipeline/pipeline.hpp"
#include "lpr_pipeline_builder.hpp"
#include <mutex>
#include <vector>
#include <nlohmann/json.hpp>

namespace webserver {
namespace pipeline {

class LprPipeline : public BasePipeline {
public:
    LprPipeline(webserver::resources::ResourceRepository &resources, MediaLibrary &media_library,
                RTPConverterStage &webrtc_stage, Architecture platform = Architecture::Hailo15H);
    
    virtual std::string pipeline_name() const override;
    void start() override;
    std::string get_profile_name_by_type(ProfileType type) const override;
    ProfileType get_profile_type_by_name(const std::string &name) const override;

    // Add endpoint registration overrides
    void register_endpoints() override;
    void unregister_endpoints() override;

protected:
    void build_pipeline() override;

private:
    // Storage for the LPR events
    std::mutex m_events_mutex;
    std::vector<nlohmann::json> m_lpr_events;
};

} // namespace pipeline
} // namespace webserver