/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include "ocr_post.hpp"
#include "hailo_postprocess_tools/objects/hailo_common.hpp"
#include "hailo_postprocess_tools/objects/hailo_objects.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>
#include <nlohmann/json.hpp> // Enterprise-grade JSON parser

using json = nlohmann::json;

// --- LPRNET CONFIGURATION ---
static const char *OUTPUT_TENSOR_NAME = "lprnet_304x75/conv31"; 

// Robust JSON loader
void load_labels_from_json(const std::string& config_path, std::vector<std::string>& labels_vec) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "[OCR_POST] ERROR: Could not open JSON " << config_path << std::endl;
        return;
    }
    
    try {
        json j;
        file >> j; // Automatically handles whitespace, \r, and \n
        
        if (j.contains("labels") && j["labels"].is_array()) {
            labels_vec.clear();
            for (const auto& label : j["labels"]) {
                labels_vec.push_back(label.get<std::string>());
            }
            std::cout << "[OCR_POST] Successfully loaded " << labels_vec.size() << " characters from JSON." << std::endl;
        } else {
            std::cerr << "[OCR_POST] ERROR: JSON does not contain 'labels' array." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[OCR_POST] ERROR parsing JSON: " << e.what() << std::endl;
    }
}

void ocr_postprocess(HailoROIPtr roi)
{
    // SAFE STATIC INITIALIZATION
    static std::vector<std::string> ocr_characters;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        load_labels_from_json("/home/root/apps/license_plate_recognition/resources/ocr_chars.json", ocr_characters);
    });

    if (!roi->has_tensors()) return;

    auto tensor = roi->get_tensor(OUTPUT_TENSOR_NAME);
    if (!tensor) return; 

    const uint32_t H = tensor->height();   // Should be 5
    const uint32_t W = tensor->width();    // Should be 19
    const uint32_t C = tensor->features(); // Should be 15
    
    const float qp_scale = tensor->qp_scale();
    const float qp_zp = tensor->qp_zp();
    const uint8_t *data = tensor->data();

    // The blank token is the last class feature
    const uint32_t BLANK_INDEX = C - 1; 

    std::string decoded_text;
    float confidence_sum = 0.0f;
    uint32_t num_chars = 0;
    uint32_t prev_idx = UINT32_MAX;

    // Loop through the horizontal timesteps
    for (uint32_t w = 0; w < W; ++w)
    {
        uint32_t best_c = 0;
        float best_val = -1e9f; 

        for (uint32_t c = 0; c < C; ++c)
        {
            float sum_h = 0.0f;
            for (uint32_t h = 0; h < H; ++h)
            {
                uint32_t offset = (h * W * C) + (w * C) + c;
                sum_h += (static_cast<float>(data[offset]) - qp_zp) * qp_scale;
            }
            
            float mean_val = sum_h / static_cast<float>(H);
            
            if (mean_val > best_val)
            {
                best_val = mean_val;
                best_c = c;
            }
        }

        // 1. Skip the CTC Blank token
        if (best_c == BLANK_INDEX)
        {
            prev_idx = best_c;
            continue;
        }

        // 2. CTC duplicate removal
        if (best_c == prev_idx)
            continue;

        prev_idx = best_c;

        // 3. Map to Dictionary loaded from JSON
        if (best_c < ocr_characters.size()) 
        {
            decoded_text += ocr_characters[best_c];
            confidence_sum += best_val;
            ++num_chars;
        }
    }

    if (decoded_text.empty())
        return;

    // Convert raw logit to 0.0-1.0 probability
    float mean_confidence = confidence_sum / static_cast<float>(num_chars);
    mean_confidence = 1.0f / (1.0f + std::exp(-mean_confidence));
    
    if (mean_confidence > 1.0f) mean_confidence = 1.0f;
    if (mean_confidence < 0.0f) mean_confidence = 0.0f;

    std::cout << "[OCR_POST] Decoded License Plate: " << decoded_text << " with confidence: " << mean_confidence << std::endl;

    roi->add_object(std::make_shared<HailoClassification>("ocr", decoded_text, mean_confidence));
}

extern "C" {
    void filter(HailoROIPtr roi)
    {
        ocr_postprocess(roi);
    }
}