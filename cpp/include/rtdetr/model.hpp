#pragma once

#include <string>
#include <vector>

namespace rtdetr {

enum class OutputKind { RawQueries, Decoded, PaddleRows };
enum class AuxiliaryValue { OriginalWH, OriginalHW, InputHW, ResizeScaleYX, UnitScale };
enum class ResizeMode { PillowBilinear, AreaDownLinearUp, Linear, Cubic };

struct AuxiliaryInput {
    std::string name;
    AuxiliaryValue value;
};

struct ModelSpec {
    std::string name;
    std::string version;
    std::string image_input;
    int width = 640;
    int height = 640;
    int classes = 80;
    int top_k = 300;
    ResizeMode resize_mode = ResizeMode::PillowBilinear;
    OutputKind output_kind = OutputKind::RawQueries;
    std::vector<AuxiliaryInput> auxiliary_inputs;
    std::string logits_output;
    std::string boxes_output;
    std::string labels_output;
    std::string scores_output;
    std::string count_output;
    bool boxes_normalized = false;
    bool paddle_boxes_input_pixels = false;
};

ModelSpec load_model(const std::string& json_path);
void validate_model(const ModelSpec& spec);

}  // namespace rtdetr
