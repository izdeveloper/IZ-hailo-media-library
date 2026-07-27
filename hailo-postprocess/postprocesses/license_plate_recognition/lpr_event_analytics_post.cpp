#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <regex>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <fstream> // Added for file stream parsing

// Hailo Analytics & Post-Process Framework Includes
#include "hailo_postprocess_tools/objects/hailo_common.hpp"
#include "hailo_postprocess_tools/objects/hailo_objects.hpp"

// Unique configuration for your application deployment
static constexpr float CONFIG_MIN_WIDTH = 0.01f;
static constexpr float CONFIG_MAX_WIDTH = 0.90f;
static constexpr float CONFIG_MIN_CONF = 0.85f;        
static constexpr uint32_t CONFIG_MIN_FRAMES = 5;      
static constexpr int CONFIG_STATE_TTL_SECONDS = 30;    
static constexpr int CONFIG_DISPATCH_COOLDOWN_SECONDS = 30; 
static constexpr int CONFIG_MAX_TYPO_DISTANCE = 2;          

// These will be loaded dynamically from the JSON file at runtime
static std::string g_http_endpoint_url = "http://127.0.0.1:8080/api/v1/lpr-events"; // Default fallback
static const std::string INEX_QUERY_PARAMS = "?cmd=uploadevent&api_version=1.7";
static const std::string CONFIG_JSON_PATH = "/home/root/apps/license_plate_recognition/resources/event_config_ip.json";

static uint64_t g_transaction_counter = 1;

struct TrackedPlateState {
    uint32_t total_valid_frames = 0;
    bool event_dispatched = false;
    std::unordered_map<std::string, int> text_votes;
    std::string best_plate_text = "";
    float highest_confidence = 0.0f;
    int last_tracking_id = -1;
    std::chrono::steady_clock::time_point last_seen_time;
    
    std::string car_color = "Unknown";
    std::string car_type = "Unknown";
    std::string plate_state = "Unknown";
};

struct DispatchedEvent {
    std::chrono::steady_clock::time_point dispatch_time;
    std::string plate_text;
};

static std::unordered_map<int, TrackedPlateState> g_state_tracker_map;
static std::vector<DispatchedEvent> g_recent_dispatches;
static std::mutex g_tracker_mutex;
static std::once_flag g_config_init_flag;

// Reusable manual string/JSON value extractor to keep the compilation straightforward
void load_config_from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[LPR_EVENT_ENGINE] ERROR: Could not open config JSON: " << path << ". Using default fallback URL." << std::endl;
        return;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Simple robust regex parsing to avoid bringing heavy library dependencies
    std::regex ip_regex("\"http_endpoint_ip\"\\s*:\\s*\"([^\"]+)\"");
    std::regex port_regex("\"http_endpoint_port\"\\s*:\\s*(\\d+)");
    std::smatch match;

    std::string ip = "127.0.0.1";
    std::string port = "8080";

    if (std::regex_search(content, match, ip_regex) && match.size() > 1) {
        ip = match[1].str();
    }
    if (std::regex_search(content, match, port_regex) && match.size() > 1) {
        port = match[1].str();
    }

    g_http_endpoint_url = "http://" + ip + ":" + port + "/api/v1/lpr-events";
    std::cout << "[LPR_EVENT_ENGINE] Successfully initialized network URL from JSON config: " << g_http_endpoint_url << std::endl;
}

std::string get_utc_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return ss.str();
}

int levenshtein_distance(const std::string& s1, const std::string& s2) {
    int len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));
    for (int i = 0; i <= len1; ++i) d[i][0] = i;
    for (int j = 0; j <= len2; ++j) d[0][j] = j;
    for (int i = 1; i <= len1; ++i) {
        for (int j = 1; j <= len2; ++j) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost });
        }
    }
    return d[len1][len2];
}

bool is_valid_plate_format(const std::string& text) {
    if (text.empty() || text[0] == '0') return false; 
    std::string clean_text = "";
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) clean_text += c;
    }
    size_t length = clean_text.length();
    if (length == 0) return false;
    if (length < 7) return std::isalpha(static_cast<unsigned char>(clean_text[0]));
    return true;
}

void dispatch_inex_event_async(uint64_t transaction_id, const TrackedPlateState& state) {
    std::thread([transaction_id, state]() {
        std::string timestamp = get_utc_iso_timestamp();
        std::string guid = "hailo-edge-tx-" + std::to_string(transaction_id) + "-" + std::to_string(state.last_tracking_id);
        int out_confidence = static_cast<int>(state.highest_confidence * 100.0f);

        std::string json_payload = "{\n"
            "  \"transaction_id\": " + std::to_string(transaction_id) + ",\n"
            "  \"transaction_guid\": \"" + guid + "\",\n"
            "  \"transaction_timestamp\": \"" + timestamp + "\",\n"
            "  \"event_version\": 1,\n"
            "  \"transaction_index\": " + std::to_string(transaction_id) + ",\n"
            "  \"lane_id\": 1,\n"
            "  \"lane_name\": \"lane_1\",\n"
            "  \"lpr_results\": [{\n"
            "    \"lpr_result_id\": 0,\n"
            "    \"plate_read\": \"" + state.best_plate_text + "\",\n"
            "    \"plate_read_confidence\": " + std::to_string(out_confidence) + ",\n"
            "    \"plate_state\": \"" + state.plate_state + "\",\n"
            "    \"plate_state_confidence\": " + std::to_string(out_confidence) + "\n"
            "  }],\n"
            "  \"vehicles\": [{\n"
            "    \"vehicle_id\": 0,\n"
            "    \"color\": \"" + state.car_color + "\",\n"
            "    \"color_confidence\": 95,\n"
            "    \"vehicle_class\": \"" + state.car_type + "\",\n"
            "    \"vehicle_class_confidence\": 90\n"
            "  }]\n"
            "}";

        // Read dynamically generated URL
        std::string full_url = g_http_endpoint_url + INEX_QUERY_PARAMS;
        std::cout << "\n[LPR_EVENT_ENGINE] Sending compliance package to destination: " << full_url << "\n" << std::endl;

        std::string command = "curl --max-time 2 -X POST -H \"Content-Type: application/json\" -d '" + json_payload + "' \"" + full_url + "\" > /dev/null 2>&1";
        
        int network_status = std::system(command.c_str());
        if (network_status != 0) {
            std::cerr << "[LPR_EVENT_ENGINE] Warning: Connection timeout/error code: " << network_status << std::endl;
        }
    }).detach();
}

extern "C" {

void filter(HailoROIPtr roi) {
    if (!roi) return;

    // Trigger one-time thread-safe config load from local disk storage[cite: 3]
    std::call_once(g_config_init_flag, []() {
        load_config_from_json(CONFIG_JSON_PATH);
    });

    auto detections = hailo_common::get_hailo_detections(roi);
    std::lock_guard<std::mutex> lock(g_tracker_mutex);
    auto now = std::chrono::steady_clock::now();

    // Clean up stale state tracker
    for (auto it = g_state_tracker_map.begin(); it != g_state_tracker_map.end(); ) {
        if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_seen_time).count() > CONFIG_STATE_TTL_SECONDS) {
            it = g_state_tracker_map.erase(it);
        } else {
            ++it;
        }
    }

    // Clean up dispatch cooldown cache
    g_recent_dispatches.erase(
        std::remove_if(g_recent_dispatches.begin(), g_recent_dispatches.end(),
            [&now](const DispatchedEvent& e) {
                return std::chrono::duration_cast<std::chrono::seconds>(now - e.dispatch_time).count() > CONFIG_DISPATCH_COOLDOWN_SECONDS;
            }),
        g_recent_dispatches.end()
    );

    for (const auto& detection : detections) {
        if (detection->get_label() != "license_plate") continue;

        int raw_tracking_id = -1;
        for (const auto& sub_obj : detection->get_objects()) {
            if (sub_obj->get_type() == HAILO_UNIQUE_ID) {
                auto id_ptr = std::dynamic_pointer_cast<HailoUniqueID>(sub_obj);
                if (id_ptr && id_ptr->get_mode() == TRACKING_ID) {
                    raw_tracking_id = id_ptr->get_id();
                    break;
                }
            }
        }
        if (raw_tracking_id == -1) continue;

        std::string plate_string = "";
        float ocr_confidence = 0.0f;
        for (const auto& sub_obj : detection->get_objects()) {
            if (sub_obj->get_type() == HAILO_CLASSIFICATION) {
                auto class_ptr = std::dynamic_pointer_cast<HailoClassification>(sub_obj);
                if (class_ptr && class_ptr->get_classification_type() == "ocr") {
                    plate_string = class_ptr->get_label();
                    ocr_confidence = class_ptr->get_confidence();
                    break;
                }
            }
        }

        HailoBBox bbox = detection->get_bbox();
        if (bbox.width() < CONFIG_MIN_WIDTH || bbox.width() > CONFIG_MAX_WIDTH) continue;

        if (ocr_confidence >= CONFIG_MIN_CONF && is_valid_plate_format(plate_string)) {
            
            int logical_track_id = raw_tracking_id;
            
            if (g_state_tracker_map.find(raw_tracking_id) == g_state_tracker_map.end()) {
                for (auto& [existing_id, existing_state] : g_state_tracker_map) {
                    if (levenshtein_distance(plate_string, existing_state.best_plate_text) <= CONFIG_MAX_TYPO_DISTANCE) {
                        logical_track_id = existing_id;
                        break;
                    }
                }
            }

            if (g_state_tracker_map.find(logical_track_id) == g_state_tracker_map.end()) {
                g_state_tracker_map[logical_track_id] = TrackedPlateState();
            }
            TrackedPlateState& state = g_state_tracker_map[logical_track_id];

            state.last_seen_time = now;
            state.last_tracking_id = raw_tracking_id;
            state.total_valid_frames++;
            state.text_votes[plate_string]++;

            int max_votes = 0;
            for (const auto& [text, votes] : state.text_votes) {
                if (votes > max_votes) {
                    max_votes = votes;
                    state.best_plate_text = text;
                }
            }

            if (ocr_confidence > state.highest_confidence) {
                state.highest_confidence = ocr_confidence;
            }

            for (const auto& sub_obj : detection->get_objects()) {
                if (sub_obj->get_type() == HAILO_CLASSIFICATION) {
                    auto cls = std::dynamic_pointer_cast<HailoClassification>(sub_obj);
                    if (cls && cls->get_classification_type() == "plate_state") state.plate_state = cls->get_label();
                }
            }

            float center_x = bbox.xmin() + (bbox.width() / 2.0f);
            float center_y = bbox.ymin() + (bbox.height() / 2.0f);

            for (const auto& top_obj : roi->get_objects()) {
                if (top_obj->get_type() == HAILO_DETECTION) {
                    auto vehicle = std::dynamic_pointer_cast<HailoDetection>(top_obj);
                    if (vehicle && vehicle->get_label() == "vehicle") {
                        HailoBBox v_box = vehicle->get_bbox();
                        if (center_x >= v_box.xmin() && center_x <= v_box.xmax() &&
                            center_y >= v_box.ymin() && center_y <= v_box.ymax()) {
                            for (const auto& v_sub : vehicle->get_objects()) {
                                if (v_sub->get_type() == HAILO_CLASSIFICATION) {
                                    auto v_cls = std::dynamic_pointer_cast<HailoClassification>(v_sub);
                                    if (v_cls) {
                                        if (v_cls->get_classification_type() == "car_color") state.car_color = v_cls->get_label();
                                        if (v_cls->get_classification_type() == "car_type") state.car_type = v_cls->get_label();
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }

            std::cout << "[LPR_ENGINE] Trace: " << logical_track_id 
                      << " | Read: " << plate_string 
                      << " | Consensus: " << state.best_plate_text
                      << " | Window: " << state.total_valid_frames << "/" << CONFIG_MIN_FRAMES << std::endl;

            if (state.total_valid_frames >= CONFIG_MIN_FRAMES && !state.event_dispatched) {
                
                bool recently_sent = false;
                for (const auto& past_event : g_recent_dispatches) {
                    if (levenshtein_distance(state.best_plate_text, past_event.plate_text) <= CONFIG_MAX_TYPO_DISTANCE) {
                        recently_sent = true;
                        break;
                    }
                }

                state.event_dispatched = true; 

                if (!recently_sent) {
                    g_recent_dispatches.push_back({now, state.best_plate_text});
                    dispatch_inex_event_async(g_transaction_counter++, state);
                } else {
                    std::cout << "[LPR_EVENT_ENGINE] Silently Suppressed Duplicate / Typo Plate: " << state.best_plate_text << std::endl;
                }
            }
        }
    }
}

} // extern "C"