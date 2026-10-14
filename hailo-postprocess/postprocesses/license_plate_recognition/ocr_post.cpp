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
#include <algorithm>   // added: std::max_element (USA head)
#include <cstdio>      // added: fopen/fread (USA head weights)
#include <nlohmann/json.hpp> // Enterprise-grade JSON parser

using json = nlohmann::json;

// --- LPRNET CONFIGURATION ---
static const char *OUTPUT_TENSOR_NAME = "lprnet_304x75/conv31";
static const char *USA_OUTPUT_TENSOR_NAME = "backbone/conv15";   // re-check with `hailortcli parse-hef` after every recompile

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

// ---------------------------------------------------------------------------
// USA OCR
// The HEF outputs the MobileNet feature map NHWC (3 x 6 x 1024, UINT8).
// The Keras head is executed here on the CPU:
//   dequantize -> Permute (swap H/W) -> Reshape (6,3,1024)->(48,384)
//   -> Dense(384->37) -> softmax -> CTC greedy decode
// Weights come from usa_ocr_head.bin, written by head_tools.py:
//   float32 little-endian: W[384][37] (row-major) followed by b[37].
// ---------------------------------------------------------------------------
static const char *USA_HEAD_WEIGHTS_PATH =
    "/home/root/apps/license_plate_recognition/resources/usa_ocr_head.bin";

static bool load_usa_head_weights(const std::string &path, std::vector<float> &w,
                                  std::vector<float> &b, size_t feat, size_t classes)
{
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "[OCR_POST_USA] ERROR: cannot open " << path << std::endl;
        return false;
    }
    w.resize(feat * classes);
    b.resize(classes);
    size_t rw = std::fread(w.data(), sizeof(float), w.size(), f);
    size_t rb = std::fread(b.data(), sizeof(float), b.size(), f);
    std::fclose(f);
    if (rw != w.size() || rb != b.size()) {
        std::cerr << "[OCR_POST_USA] ERROR: head weights file has wrong size: " << path << std::endl;
        w.clear();
        b.clear();
        return false;
    }
    std::cout << "[OCR_POST_USA] Loaded head weights from " << path << std::endl;
    return true;
}

void ocr_postprocess_usa(HailoROIPtr roi) {
    static std::vector<std::string> ocr_characters_usa;
    static std::once_flag init_flag_usa;
    static std::vector<float> head_w, head_b;   // W[384][37], b[37]
    static std::once_flag head_flag;

    const uint32_t SEQ_LEN = 48;                // 48 timesteps
    const uint32_t FEAT = 384;                  // features per timestep after reshape
    const uint32_t CLASSES = 37;                // 37 classes
    const uint32_t BLANK_INDEX = CLASSES - 1;   // CTC blank = index 36

    std::call_once(init_flag_usa, []() {
        load_labels_from_json("/home/root/apps/license_plate_recognition/resources/usa_ocr_chars.json", ocr_characters_usa);
    });
    std::call_once(head_flag, [&]() {
        load_usa_head_weights(USA_HEAD_WEIGHTS_PATH, head_w, head_b, FEAT, CLASSES);
    });

    if (head_w.empty()) return;
    if (!roi->has_tensors()) return;
    auto tensor = roi->get_tensor(USA_OUTPUT_TENSOR_NAME);
    if (!tensor) return;

    const uint32_t FH = tensor->height();       // 3
    const uint32_t FW = tensor->width();        // 6
    const uint32_t FC = tensor->features();     // 1024
    if (FH != 3 || FW != 6 || FC != 1024) {
        std::cerr << "[OCR_POST_USA] unexpected output shape " << FH << "x" << FW << "x" << FC
                  << " (expected 3x6x1024)" << std::endl;
        return;
    }

    const float qp_scale = tensor->qp_scale();
    const float qp_zp = tensor->qp_zp();
    const uint8_t *data = tensor->data();

    // 1. Dequantize + Permute(2,1,3): NHWC (h,w,c) -> (w,h,c), stored flat as [(w*FH + h)*FC + c].
    //    The Keras Reshape to (48,384) is then a plain read of this flat buffer in rows of 384.
    std::vector<float> flat(FH * FW * FC);
    for (uint32_t h = 0; h < FH; ++h) {
        for (uint32_t w = 0; w < FW; ++w) {
            for (uint32_t c = 0; c < FC; ++c) {
                flat[(w * FH + h) * FC + c] =
                    (static_cast<float>(data[(h * FW + w) * FC + c]) - qp_zp) * qp_scale;
            }
        }
    }

    std::string decoded_text;
    float prob_sum = 0.0f;
    uint32_t num_chars = 0;
    uint32_t prev_idx = UINT32_MAX;

    std::vector<float> logits(CLASSES);

    for (uint32_t r = 0; r < SEQ_LEN; ++r) {
        const float *x = &flat[r * FEAT];

        // 2. Dense 384 -> 37
        for (uint32_t k = 0; k < CLASSES; ++k) logits[k] = head_b[k];
        for (uint32_t j = 0; j < FEAT; ++j) {
            const float xv = x[j];
            const float *wr = &head_w[j * CLASSES];
            for (uint32_t k = 0; k < CLASSES; ++k) logits[k] += xv * wr[k];
        }

        // 3. Softmax normalization across 37 classes
        const float max_logit = *std::max_element(logits.begin(), logits.end());
        float exp_sum = 0.0f;
        for (uint32_t k = 0; k < CLASSES; ++k) {
            logits[k] = std::exp(logits[k] - max_logit);
            exp_sum += logits[k];
        }

        uint32_t best_c = 0;
        float best_prob = 0.0f;
        for (uint32_t k = 0; k < CLASSES; ++k) {
            const float p = logits[k] / exp_sum;
            if (p > best_prob) {
                best_prob = p;
                best_c = k;
            }
        }

        // 4. CTC Decoding
        if (best_c == BLANK_INDEX) {
            prev_idx = best_c;
            continue;
        }

        if (best_c == prev_idx) continue;
        prev_idx = best_c;

        if (best_c < ocr_characters_usa.size()) {
            decoded_text += ocr_characters_usa[best_c];
            prob_sum += best_prob;
            ++num_chars;
        }
    }

    if (decoded_text.empty() || num_chars == 0) return;

    // 5. Average probability confidence
    float mean_confidence = prob_sum / static_cast<float>(num_chars);

    std::cout << "[OCR_POST_USA] Decoded Plate: " << decoded_text << " conf: " << mean_confidence << std::endl;
    roi->add_object(std::make_shared<HailoClassification>("ocr", decoded_text, mean_confidence));
}

extern "C" {
    void filter(HailoROIPtr roi)
    {
        ocr_postprocess(roi);
    }

    void filter_usa(HailoROIPtr roi) {
        ocr_postprocess_usa(roi);
    }
}