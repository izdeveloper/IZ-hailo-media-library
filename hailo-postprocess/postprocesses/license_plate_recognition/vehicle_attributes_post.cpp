#include "hailo_postprocess_tools/objects/hailo_common.hpp"
#include "hailo_postprocess_tools/objects/hailo_objects.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <cmath> // Required for std::exp

// Reusable JSON parser
void load_labels_from_json(const std::string& config_path, std::vector<std::string>& labels_vec, const std::string& log_name) {
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
        std::cout << "[" << log_name << "] Successfully loaded " << labels_vec.size() << " labels from JSON." << std::endl;
    } else {
        std::cerr << "[" << log_name << "] ERROR: Could not open JSON " << config_path << std::endl;
    }
}

extern "C" {

void filter(HailoROIPtr roi)
{
    // ==========================================
    // SAFE STATIC INITIALIZATION:
    // Moving these inside the function prevents dlopen() heap corruption!
    // ==========================================
    static std::vector<std::string> color_labels;
    static std::vector<std::string> type_labels;
    static std::once_flag init_flag;

    std::call_once(init_flag, []() {
        load_labels_from_json("/home/root/apps/license_plate_recognition/resources/color_labels.json", color_labels, "VEHICLE_COLOR");
        load_labels_from_json("/home/root/apps/license_plate_recognition/resources/type_labels.json", type_labels, "VEHICLE_TYPE");
    });

    if (!roi->has_tensors()) return;

    auto tensors = roi->get_tensors();
    if (tensors.size() < 2) return; 

    HailoTensorPtr color_tensor = nullptr;
    HailoTensorPtr type_tensor = nullptr;

    for (const auto& tensor : tensors) {
        if (tensor->name().find("dense_conv21") != std::string::npos) {
            color_tensor = tensor; 
        } else if (tensor->name().find("dense_conv22") != std::string::npos) {
            type_tensor = tensor;  
        }
    }

    if (!color_tensor || !type_tensor) return;

    // 1. Process Car Color 
    {
        const uint8_t *data = color_tensor->data();
        float qp_scale = color_tensor->qp_scale();
        float qp_zp = color_tensor->qp_zp();
        uint32_t num_classes = color_labels.size();
        if (color_tensor->size() < num_classes) num_classes = color_tensor->size();

        std::vector<float> logits(num_classes);
        float max_logit = -999999.0f;
        int max_idx = 0;

        for (uint32_t i = 0; i < num_classes; i++) {
            logits[i] = (static_cast<float>(data[i]) - qp_zp) * qp_scale;
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_idx = i;
            }
        }

        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < num_classes; i++) {
            sum_exp += std::exp(logits[i] - max_logit); 
        }
        float confidence = 1.0f / sum_exp; 

        std::string label = (max_idx < (int)color_labels.size()) ? color_labels[max_idx] : "Unknown";
        //std::cout << "[VEHICLE_COLOR] Detected Vehicle Color: " << label << " with confidence: " << confidence << std::endl;
        roi->add_object(std::make_shared<HailoClassification>("car_color", label, confidence));
    }

    // 2. Process Vehicle Type
    {
        const uint8_t *data = type_tensor->data();
        float qp_scale = type_tensor->qp_scale();
        float qp_zp = type_tensor->qp_zp();
        uint32_t num_classes = type_labels.size();
        if (type_tensor->size() < num_classes) num_classes = type_tensor->size();

        std::vector<float> logits(num_classes);
        float max_logit = -999999.0f;
        int max_idx = 0;

        for (uint32_t i = 0; i < num_classes; i++) {
            logits[i] = (static_cast<float>(data[i]) - qp_zp) * qp_scale;
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_idx = i;
            }
        }

        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < num_classes; i++) {
            sum_exp += std::exp(logits[i] - max_logit);
        }
        float confidence = 1.0f / sum_exp;

        std::string label = (max_idx < (int)type_labels.size()) ? type_labels[max_idx] : "Unknown";

        //std::cout << "[VEHICLE_TYPE] Detected Vehicle Type: " << label << " with confidence: " << confidence << std::endl;
        roi->add_object(std::make_shared<HailoClassification>("car_type", label, confidence));
    }
}

} // END EXTERN "C"