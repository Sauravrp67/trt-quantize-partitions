#pragma once
#include "rtdetr/model.hpp"
#include <opencv2/core.hpp>

namespace rtdetr {
cv::Mat resize_bgr(const cv::Mat& input, int width, int height, ResizeMode mode);
}
