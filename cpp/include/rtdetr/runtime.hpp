#pragma once

#include "rtdetr/model.hpp"

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace rtdetr {

struct Detection {
    int label;
    float score;
    cv::Rect2f box;
};

struct StageTimes {
    double preprocess_ms = 0;
    double engine_ms = 0;
    double postprocess_ms = 0;
};

class Detector {
public:
    Detector(ModelSpec spec, const std::string& engine_path);
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    std::vector<Detection> infer(const cv::Mat& bgr, float threshold = 0.5f,
                                 StageTimes* times = nullptr);
    void warmup(int count);
    const ModelSpec& spec() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtdetr
