#include "hailo_postprocess_tools/objects/hailo_common.hpp"
#include "hailo_postprocess_tools/objects/hailo_objects.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <mutex>
#include <algorithm>

// Reusable JSON parser without globals
void load_labels_from_json(const std::string& config_path, std::vector<std::string>& labels_vec) {
    std::ifstream file(config_path);
    
    if (file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        size_t start = content.find("\"labels\"");
        if (start != std::string::npos) {
            size_t array_start = content.find('[', start);
            size_t array_end = content.find(']', array_start);
            if (array_start != std::string::npos && array_end != std::string::npos) {
                std::string array_content = content.substr(array_start, array_end - array_start);
                size_t pos = 0;
                while ((pos = array_content.find('"', pos)) != std::string::npos) {
                    size_t end_pos = array_content.find('"', pos + 1);
                    if (end_pos != std::string::npos) {
                        labels_vec.push_back(array_content.substr(pos + 1, end_pos - pos - 1));
                        pos = end_pos + 1;
                    } else {
                        break;
                    }
                }
            }
        }
        file.close();
        std::cout << "[STATE_CLASSIFIER] Successfully loaded " << labels_vec.size() << " labels." << std::endl;
    } else {
        std::cerr << "[STATE_CLASSIFIER] ERROR: Could not open JSON " << config_path << std::endl;
    }
}

extern "C" {

// Standard signature (One argument only!)
void filter(HailoROIPtr roi)
{
    // ==========================================
    // SAFE STATIC INITIALIZATION:
    // Moved inside the function to prevent dlopen() heap corruption
    // ==========================================
    static std::vector<std::string> plate_state_labels;
    static std::once_flag init_flag;

    // Ensure labels are loaded on the first frame, thread-safe
    std::call_once(init_flag, []() {
        load_labels_from_json("/home/root/apps/license_plate_recognition/resources/state_labels.json", plate_state_labels);
    });

    if (!roi->has_tensors()) return;

    auto tensors = roi->get_tensors();
    if (tensors.empty()) return;
    auto scores = tensors[0];

    const uint8_t *data = scores->data();
    if (!data) return;

    float qp_scale = scores->qp_scale();
    float qp_zp = scores->qp_zp();

    int max_raw = -1;
    int max_idx = 0;
    
    // SAFE BOUNDS CHECKING: 
    // scores->size() returns exactly how many bytes are actually allocated in the tensor.
    uint32_t tensor_size_bytes = scores->size();
    
    // We strictly bound the loop so it NEVER reads out of bounds, even if the model compiles strangely
    uint32_t elements_to_check = plate_state_labels.size();
    if (tensor_size_bytes < elements_to_check) {
        elements_to_check = tensor_size_bytes; 
    }

    if (elements_to_check == 0) return;

    for (uint32_t i = 0; i < elements_to_check; i++) {
        if ((int)data[i] > max_raw) {
            max_raw = (int)data[i];
            max_idx = i;
        }
    }

    float confidence = (static_cast<float>(max_raw) - qp_zp) * qp_scale;

    // Safety Clamps
    if (confidence > 1.0f) confidence = 1.0f;
    if (confidence < 0.0f) confidence = 0.0f;

    std::string label = "Unknown";
    if (max_idx < (int)plate_state_labels.size()) {
        label = plate_state_labels[max_idx];
    }

    //std::cout << "[STATE_CLASSIFIER] Inferred state: " << label << " with confidence: " << confidence << std::endl;

    roi->add_object(std::make_shared<HailoClassification>("plate_state", label, confidence));
}

} // END EXTERN "C"