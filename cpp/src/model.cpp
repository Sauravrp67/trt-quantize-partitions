#include "rtdetr/model.hpp"

#include <opencv2/core/persistence.hpp>
#include <stdexcept>
#include <unordered_set>

namespace rtdetr {
namespace {
std::string required_string(const cv::FileNode& node, const char* key) {
    const cv::FileNode value = node[key];
    if (value.empty() || !value.isString()) throw std::runtime_error(std::string("missing string: ") + key);
    return static_cast<std::string>(value);
}

std::string optional_string(const cv::FileNode& node, const char* key) {
    const cv::FileNode value = node[key];
    return value.empty() ? "" : static_cast<std::string>(value);
}

int optional_int(const cv::FileNode& node, const char* key, int fallback) {
    const cv::FileNode value = node[key];
    return value.empty() ? fallback : static_cast<int>(value);
}

bool optional_bool(const cv::FileNode& node, const char* key) {
    const cv::FileNode value = node[key];
    return !value.empty() && static_cast<int>(value) != 0;
}
}  // namespace

ModelSpec load_model(const std::string& json_path) {
    cv::FileStorage file(json_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!file.isOpened()) throw std::runtime_error("cannot open model manifest: " + json_path);
    const cv::FileNode root = file.root();
    ModelSpec s;
    s.name = required_string(root, "name");
    s.version = required_string(root, "version");
    s.image_input = required_string(root, "image_input");
    s.width = optional_int(root, "width", 640);
    s.height = optional_int(root, "height", 640);
    s.classes = optional_int(root, "classes", 80);
    s.top_k = optional_int(root, "top_k", 300);
    const std::string resize = optional_string(root, "resize_mode");
    if (resize.empty() || resize == "pillow_bilinear") s.resize_mode = ResizeMode::PillowBilinear;
    else if (resize == "area_down_linear_up") s.resize_mode = ResizeMode::AreaDownLinearUp;
    else if (resize == "linear") s.resize_mode = ResizeMode::Linear;
    else if (resize == "cubic") s.resize_mode = ResizeMode::Cubic;
    else throw std::runtime_error("unsupported resize_mode: " + resize);
    const std::string kind = required_string(root, "output_kind");
    if (kind == "raw_queries") s.output_kind = OutputKind::RawQueries;
    else if (kind == "decoded") s.output_kind = OutputKind::Decoded;
    else if (kind == "paddle_rows") s.output_kind = OutputKind::PaddleRows;
    else throw std::runtime_error("unsupported output_kind: " + kind);
    s.logits_output = optional_string(root, "logits_output");
    s.boxes_output = optional_string(root, "boxes_output");
    s.labels_output = optional_string(root, "labels_output");
    s.scores_output = optional_string(root, "scores_output");
    s.count_output = optional_string(root, "count_output");
    s.boxes_normalized = optional_bool(root, "boxes_normalized");
    s.paddle_boxes_input_pixels = optional_bool(root, "paddle_boxes_input_pixels");
    const cv::FileNode aux = root["auxiliary_inputs"];
    if (!aux.empty()) {
        if (!aux.isSeq()) throw std::runtime_error("auxiliary_inputs must be an array");
        for (const auto& item : aux) {
            const std::string value = required_string(item, "value");
            AuxiliaryValue parsed;
            if (value == "original_wh") parsed = AuxiliaryValue::OriginalWH;
            else if (value == "original_hw") parsed = AuxiliaryValue::OriginalHW;
            else if (value == "input_hw") parsed = AuxiliaryValue::InputHW;
            else if (value == "resize_scale_yx") parsed = AuxiliaryValue::ResizeScaleYX;
            else if (value == "unit_scale") parsed = AuxiliaryValue::UnitScale;
            else throw std::runtime_error("unsupported auxiliary value: " + value);
            s.auxiliary_inputs.push_back({required_string(item, "name"), parsed});
        }
    }
    validate_model(s);
    return s;
}

void validate_model(const ModelSpec& s) {
    if (s.name.empty() || s.version.empty() || s.image_input.empty() ||
        s.width <= 0 || s.height <= 0 || s.classes <= 0 || s.top_k <= 0)
        throw std::runtime_error("invalid model identity or dimensions");
    std::unordered_set<std::string> names{s.image_input};
    for (const auto& aux : s.auxiliary_inputs) {
        if (aux.name.empty() || !names.insert(aux.name).second)
            throw std::runtime_error("duplicate or empty input name");
    }
    if (s.boxes_output.empty()) throw std::runtime_error("boxes_output is required");
    if (s.output_kind == OutputKind::RawQueries && s.logits_output.empty())
        throw std::runtime_error("raw_queries requires logits_output");
    if (s.output_kind == OutputKind::Decoded && (s.labels_output.empty() || s.scores_output.empty()))
        throw std::runtime_error("decoded requires labels_output and scores_output");
    if (s.output_kind == OutputKind::PaddleRows && s.count_output.empty())
        throw std::runtime_error("paddle_rows requires count_output");
}
}  // namespace rtdetr
