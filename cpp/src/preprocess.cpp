#include "rtdetr/preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rtdetr {
namespace {
constexpr int fractional_bits = 22;

struct TapSet {
    int first;
    std::vector<int32_t> weights;
};

std::vector<TapSet> make_taps(int source, int target) {
    const double scale = static_cast<double>(source) / target;
    const double filter_scale = std::max(scale, 1.0);
    std::vector<TapSet> taps;
    taps.reserve(target);
    for (int out = 0; out < target; ++out) {
        const double center = (out + 0.5) * scale;
        const int first = std::max(0, static_cast<int>(center - filter_scale + 0.5));
        const int last = std::min(source, static_cast<int>(center + filter_scale + 0.5));
        std::vector<double> floating;
        double total = 0;
        for (int src = first; src < last; ++src) {
            const double distance = std::abs((src - center + 0.5) / filter_scale);
            const double weight = std::max(0.0, 1.0 - distance);
            floating.push_back(weight);
            total += weight;
        }
        TapSet set{first, {}};
        for (double weight : floating)
            set.weights.push_back(static_cast<int32_t>(std::lround(weight / total * (1 << fractional_bits))));
        taps.push_back(std::move(set));
    }
    return taps;
}

uint8_t clamp_byte(int64_t fixed) {
    return static_cast<uint8_t>(std::clamp<int64_t>(fixed >> fractional_bits, 0, 255));
}

cv::Mat horizontal(const cv::Mat& src, int width, const std::vector<TapSet>& taps) {
    cv::Mat dst(src.rows, width, CV_8UC3);
    cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            const auto* input = src.ptr<cv::Vec3b>(y);
            auto* output = dst.ptr<cv::Vec3b>(y);
            for (int x = 0; x < width; ++x) {
                int64_t sum[3] = {1LL << (fractional_bits - 1), 1LL << (fractional_bits - 1),
                                  1LL << (fractional_bits - 1)};
                const auto& tap = taps[x];
                for (size_t k = 0; k < tap.weights.size(); ++k) {
                    const auto pixel = input[tap.first + k];
                    for (int c = 0; c < 3; ++c) sum[c] += pixel[c] * static_cast<int64_t>(tap.weights[k]);
                }
                for (int c = 0; c < 3; ++c) output[x][c] = clamp_byte(sum[c]);
            }
        }
    });
    return dst;
}

cv::Mat vertical(const cv::Mat& src, int height, const std::vector<TapSet>& taps) {
    cv::Mat dst(height, src.cols, CV_8UC3);
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            auto* output = dst.ptr<cv::Vec3b>(y);
            const auto& tap = taps[y];
            for (int x = 0; x < src.cols; ++x) {
                int64_t sum[3] = {1LL << (fractional_bits - 1), 1LL << (fractional_bits - 1),
                                  1LL << (fractional_bits - 1)};
                for (size_t k = 0; k < tap.weights.size(); ++k) {
                    const auto pixel = src.ptr<cv::Vec3b>(tap.first + k)[x];
                    for (int c = 0; c < 3; ++c) sum[c] += pixel[c] * static_cast<int64_t>(tap.weights[k]);
                }
                for (int c = 0; c < 3; ++c) output[x][c] = clamp_byte(sum[c]);
            }
        }
    });
    return dst;
}
}  // namespace

cv::Mat resize_bgr(const cv::Mat& input, int width, int height, ResizeMode mode) {
    if (input.empty() || input.type() != CV_8UC3 || width <= 0 || height <= 0)
        throw std::runtime_error("resize_bgr requires BGR uint8 input and positive size");
    if (mode != ResizeMode::PillowBilinear) {
        int interpolation = cv::INTER_LINEAR;
        if (mode == ResizeMode::Cubic) interpolation = cv::INTER_CUBIC;
        else if (mode == ResizeMode::AreaDownLinearUp &&
                 (input.cols > width || input.rows > height)) interpolation = cv::INTER_AREA;
        cv::Mat out;
        cv::resize(input, out, {width, height}, 0, 0, interpolation);
        return out;
    }
    if (input.cols == width && input.rows == height) return input;
    const bool horizontal_first = !((input.rows - height) > 0 &&
                                    (input.rows - height) > (input.cols - width) * 2);
    cv::Mat result = input;
    if (horizontal_first) {
        if (result.cols != width) result = horizontal(result, width, make_taps(input.cols, width));
        if (result.rows != height) result = vertical(result, height, make_taps(input.rows, height));
    } else {
        if (result.rows != height) result = vertical(result, height, make_taps(input.rows, height));
        if (result.cols != width) result = horizontal(result, width, make_taps(input.cols, width));
    }
    return result;
}
}  // namespace rtdetr
