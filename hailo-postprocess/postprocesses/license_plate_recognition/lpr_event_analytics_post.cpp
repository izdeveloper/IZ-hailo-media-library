#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

#include <nlohmann/json.hpp>
#include "hailo_postprocess_tools/objects/hailo_common.hpp"
#include "hailo_postprocess_tools/objects/hailo_objects.hpp"

// Configuration Constants
static constexpr float CONFIG_MIN_WIDTH = 0.02f;
static constexpr float CONFIG_MAX_WIDTH = 0.90f;
static constexpr float CONFIG_MIN_CONF = 0.85f;
static constexpr uint32_t CONFIG_MIN_STABLE_READS = 3; // Minimum threshold for valid dispatch
static constexpr uint32_t CONFIG_MAX_FRAMES = 15;      // Max frames to collect for candidate voting
static constexpr int CONFIG_TRACK_TIMEOUT_MS = 10000;
static constexpr int CONFIG_STATE_TTL_SECONDS = 60;
static constexpr int CONFIG_DISPATCH_COOLDOWN_SECONDS = 120;
static constexpr int CONFIG_MAX_TYPO_DISTANCE = 3;

static uint64_t g_transaction_counter = 1;

struct TrackedPlateState
{
    uint32_t total_valid_frames = 0;
    bool event_dispatched = false;

    std::unordered_map<std::string, uint32_t> text_counts; // Occurrence counter
    std::unordered_map<std::string, float> text_votes;     // Accumulated confidence

    std::string best_plate_text = "";
    uint32_t best_plate_count = 0;
    float highest_confidence = 0.0f;

    int last_tracking_id = -1;
    HailoBBox last_bbox{0, 0, 0, 0};
    std::chrono::steady_clock::time_point last_seen_time;

    std::string car_color = "Unknown";
    std::string car_type = "Unknown";
    std::string plate_state = "Unknown";
};

struct DispatchedEvent
{
    std::chrono::steady_clock::time_point dispatch_time;
    std::string plate_text;
};

static std::unordered_map<int, TrackedPlateState> g_state_tracker_map;
static std::vector<DispatchedEvent> g_recent_dispatches;
static std::mutex g_tracker_mutex;

std::string get_utc_iso_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return ss.str();
}

float get_bbox_center_distance(const HailoBBox &b1, const HailoBBox &b2)
{
    float c1_x = b1.xmin() + (b1.width() / 2.0f);
    float c1_y = b1.ymin() + (b1.height() / 2.0f);
    float c2_x = b2.xmin() + (b2.width() / 2.0f);
    float c2_y = b2.ymin() + (b2.height() / 2.0f);
    return std::sqrt(std::pow(c1_x - c2_x, 2) + std::pow(c1_y - c2_y, 2));
}

int levenshtein_distance(const std::string &s1, const std::string &s2)
{
    int len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));
    for (int i = 0; i <= len1; ++i)
        d[i][0] = i;
    for (int j = 0; j <= len2; ++j)
        d[0][j] = j;
    for (int i = 1; i <= len1; ++i)
    {
        for (int j = 1; j <= len2; ++j)
        {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost});
        }
    }
    return d[len1][len2];
}

bool is_valid_plate_format(const std::string &text)
{
    if (text.empty() || text[0] == '0')
        return false;

    std::string clean_text = "";
    bool has_letter = false; // Track if we encounter any letters

    for (char c : text)
    {
        if (std::isalnum(static_cast<unsigned char>(c)))
        {
            clean_text += c;
            if (std::isalpha(static_cast<unsigned char>(c)))
            {
                has_letter = true;
            }
        }
    }

    size_t length = clean_text.length();
    if (length == 0)
        return false;

    if(has_letter)  
    {   
        bool seen_digit = false;
        bool seen_trailing_letter = false;
        for (char c : clean_text)
        {
            if (std::isdigit(static_cast<unsigned char>(c)))
            {
                if (seen_trailing_letter)
                    return false;
                seen_digit = true;
            }
            else if (seen_digit)
            {
                seen_trailing_letter = true;
            }
        }

        if (!seen_digit)
            return false;
    }

    // Condition: If length is less than 7 or greater than 8, it MUST contain a letter
    if (length < 7 || length > 8)
    {
        if (!has_letter)
        {
            return false;
        }
    }

    return true;
}

bool is_recent_duplicate(const std::string &candidate_text, const std::chrono::steady_clock::time_point &now)
{
    if (candidate_text.empty())
        return true;

    for (const auto &past_event : g_recent_dispatches)
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - past_event.dispatch_time).count();
        if (elapsed > CONFIG_DISPATCH_COOLDOWN_SECONDS)
            continue;

        if (levenshtein_distance(candidate_text, past_event.plate_text) <= CONFIG_MAX_TYPO_DISTANCE)
        {
            return true;
        }

        if (candidate_text.length() >= 5 && past_event.plate_text.length() >= 5)
        {
            std::string suf1 = candidate_text.substr(candidate_text.length() - 5);
            std::string suf2 = past_event.plate_text.substr(past_event.plate_text.length() - 5);
            if (suf1 == suf2)
                return true;
        }
    }
    return false;
}

// Dynamically updates the highest-voted string across all frames collected so far
void update_best_candidate(TrackedPlateState &state)
{
    std::string top_text = "";
    uint32_t max_count = 0;
    float max_score = 0.0f;

    for (const auto &[text, count] : state.text_counts)
    {
        float score = state.text_votes[text];
        if (count > max_count || (count == max_count && score > max_score))
        {
            max_count = count;
            max_score = score;
            top_text = text;
        }
    }
    state.best_plate_text = top_text;
    state.best_plate_count = max_count;
}

void tag_event_on_roi(HailoROIPtr roi, const TrackedPlateState &state)
{
    uint64_t tx_id = g_transaction_counter++;
    std::string guid = "hailo-edge-tx-" + std::to_string(tx_id) + "-" + std::to_string(state.last_tracking_id);
    int out_confidence = static_cast<int>(state.highest_confidence * 100.0f);

    nlohmann::json event_payload = {
        {"transaction_id", tx_id},
        {"transaction_guid", guid},
        {"transaction_timestamp", get_utc_iso_timestamp()},
        {"event_version", 1},
        {"transaction_index", tx_id},
        {"lane_id", 1},
        {"lane_name", "lane_1"},
        {"lpr_results", nlohmann::json::array({{{"lpr_result_id", 0},
                                                {"plate_read", state.best_plate_text},
                                                {"plate_read_confidence", out_confidence},
                                                {"plate_state", state.plate_state},
                                                {"plate_state_confidence", out_confidence}}})},
        {"vehicles", nlohmann::json::array({{{"vehicle_id", 0},
                                             {"color", state.car_color},
                                             {"color_confidence", 95},
                                             {"vehicle_class", state.car_type},
                                             {"vehicle_class_confidence", 90}}})}};

    std::cout << "Dispatching INEX Event: " << event_payload.dump() << std::endl;
    roi->add_object(std::make_shared<HailoClassification>("inex_event", event_payload.dump(), state.highest_confidence));
}

extern "C"
{

    void filter(HailoROIPtr roi)
    {
        if (!roi)
            return;

        auto detections = hailo_common::get_hailo_detections(roi);
        std::lock_guard<std::mutex> lock(g_tracker_mutex);
        auto now = std::chrono::steady_clock::now();

        // 1. Process Timeout Dispatches (For cars exiting before reaching CONFIG_MAX_FRAMES)
        for (auto &[track_id, state] : g_state_tracker_map)
        {
            if (state.event_dispatched)
                continue;

            auto time_since_last_seen = std::chrono::duration_cast<std::chrono::milliseconds>(now - state.last_seen_time).count();

            if (time_since_last_seen > CONFIG_TRACK_TIMEOUT_MS && 
                time_since_last_seen < CONFIG_STATE_TTL_SECONDS * 1000 && 
                state.total_valid_frames >= CONFIG_MIN_STABLE_READS && 
                state.best_plate_count >= CONFIG_MIN_STABLE_READS)
            {
                bool recently_sent = false;
                for (const auto &past_event : g_recent_dispatches)
                {
                    if (levenshtein_distance(state.best_plate_text, past_event.plate_text) <= CONFIG_MAX_TYPO_DISTANCE)
                    {
                        recently_sent = is_recent_duplicate(state.best_plate_text, now);
                        ;
                        break;
                    }
                }

                state.event_dispatched = true;

                if (!recently_sent && !state.best_plate_text.empty())
                {
                    g_recent_dispatches.push_back({now, state.best_plate_text});
                    tag_event_on_roi(roi, state);
                }
            }
        }

        // 2. Cleanup stale state history
        for (auto it = g_state_tracker_map.begin(); it != g_state_tracker_map.end();)
        {
            if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_seen_time).count() > CONFIG_STATE_TTL_SECONDS)
            {
                it = g_state_tracker_map.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // 3. Cleanup dispatch cooldown list
        g_recent_dispatches.erase(
            std::remove_if(g_recent_dispatches.begin(), g_recent_dispatches.end(),
                           [&now](const DispatchedEvent &e)
                           {
                               return std::chrono::duration_cast<std::chrono::seconds>(now - e.dispatch_time).count() > CONFIG_DISPATCH_COOLDOWN_SECONDS;
                           }),
            g_recent_dispatches.end());

        // 4. Process Current Frame Detections
        for (const auto &detection : detections)
        {
            if (detection->get_label() != "license_plate")
                continue;

            int raw_tracking_id = -1;
            for (const auto &sub_obj : detection->get_objects())
            {
                if (sub_obj->get_type() == HAILO_UNIQUE_ID)
                {
                    auto id_ptr = std::dynamic_pointer_cast<HailoUniqueID>(sub_obj);
                    if (id_ptr && id_ptr->get_mode() == TRACKING_ID)
                    {
                        raw_tracking_id = id_ptr->get_id();
                        break;
                    }
                }
            }
            if (raw_tracking_id == -1)
                continue;

            std::string plate_string = "";
            float ocr_confidence = 0.0f;
            for (const auto &sub_obj : detection->get_objects())
            {
                if (sub_obj->get_type() == HAILO_CLASSIFICATION)
                {
                    auto class_ptr = std::dynamic_pointer_cast<HailoClassification>(sub_obj);
                    if (class_ptr && class_ptr->get_classification_type() == "ocr")
                    {
                        plate_string = class_ptr->get_label();
                        ocr_confidence = class_ptr->get_confidence();
                        break;
                    }
                }
            }

            HailoBBox bbox = detection->get_bbox();
            if (bbox.width() < CONFIG_MIN_WIDTH || bbox.width() > CONFIG_MAX_WIDTH)
                continue;

            if (ocr_confidence >= CONFIG_MIN_CONF && is_valid_plate_format(plate_string))
            {
                int logical_track_id = raw_tracking_id;

                // Match to existing active track
                if (g_state_tracker_map.find(raw_tracking_id) == g_state_tracker_map.end())
                {
                    float min_distance = 0.25f; // Max normalized movement threshold between frames (~15% of frame)
                    int matched_id = -1;

                    for (auto &[existing_id, existing_state] : g_state_tracker_map)
                    {
                        // Calculate spatial distance from last seen position
                        float dist = get_bbox_center_distance(bbox, existing_state.last_bbox);
                        auto ms_since_seen = std::chrono::duration_cast<std::chrono::milliseconds>(now - existing_state.last_seen_time).count();

                        if (dist < min_distance && ms_since_seen < 1000)
                        {
                            min_distance = dist;
                            matched_id = existing_id;
                        }
                    }

                    if (matched_id != -1)
                    {
                        logical_track_id = matched_id; // Re-associate fragmented tracker ID
                    }
                }

                if (g_state_tracker_map.find(logical_track_id) == g_state_tracker_map.end())
                {
                    TrackedPlateState new_state;
                    new_state.last_seen_time = now;
                    g_state_tracker_map[logical_track_id] = new_state;
                }
                TrackedPlateState &state = g_state_tracker_map[logical_track_id];

                state.last_seen_time = now;
                state.last_tracking_id = raw_tracking_id;
                state.last_bbox = bbox;

                if (state.event_dispatched)
                    continue;

                // Accumulate reads across all frames
                state.total_valid_frames++;
                state.text_counts[plate_string]++;
                state.text_votes[plate_string] += ocr_confidence;

                // Recalculate top candidate dynamically
                update_best_candidate(state);

                if (ocr_confidence > state.highest_confidence)
                {
                    state.highest_confidence = ocr_confidence;
                }

                for (const auto &sub_obj : detection->get_objects())
                {
                    if (sub_obj->get_type() == HAILO_CLASSIFICATION)
                    {
                        auto cls = std::dynamic_pointer_cast<HailoClassification>(sub_obj);
                        if (cls && cls->get_classification_type() == "plate_state")
                            state.plate_state = cls->get_label();
                    }
                }

                // Map vehicle attributes via spatial inclusion
                float center_x = bbox.xmin() + (bbox.width() / 2.0f);
                float center_y = bbox.ymin() + (bbox.height() / 2.0f);

                for (const auto &top_obj : roi->get_objects())
                {
                    if (top_obj->get_type() == HAILO_DETECTION)
                    {
                        auto vehicle = std::dynamic_pointer_cast<HailoDetection>(top_obj);
                        if (vehicle && vehicle->get_label() == "vehicle")
                        {
                            HailoBBox v_box = vehicle->get_bbox();
                            if (center_x >= v_box.xmin() && center_x <= v_box.xmax() &&
                                center_y >= v_box.ymin() && center_y <= v_box.ymax())
                            {
                                for (const auto &v_sub : vehicle->get_objects())
                                {
                                    if (v_sub->get_type() == HAILO_CLASSIFICATION)
                                    {
                                        auto v_cls = std::dynamic_pointer_cast<HailoClassification>(v_sub);
                                        if (v_cls)
                                        {
                                            if (v_cls->get_classification_type() == "car_color")
                                                state.car_color = v_cls->get_label();
                                            if (v_cls->get_classification_type() == "car_type")
                                                state.car_type = v_cls->get_label();
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }

                // Dispatch ONLY when max frames limit (15) is reached AND candidate has >= 3 reads
                if (state.best_plate_count >= CONFIG_MIN_STABLE_READS && state.total_valid_frames >= CONFIG_MAX_FRAMES && !state.event_dispatched)
                {
                    state.event_dispatched = true;

                    if (!state.best_plate_text.empty())
                    {
                        if (!is_recent_duplicate(state.best_plate_text, now))
                        {
                            g_recent_dispatches.push_back({now, state.best_plate_text});
                            tag_event_on_roi(roi, state);
                        }
                    }
                }
            }
        }
    }

} // extern "C"