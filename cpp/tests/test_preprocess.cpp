#include "rtdetr/preprocess.hpp"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    for (const auto& shape : {std::pair{23, 17}, std::pair{11, 23}, std::pair{9, 11}}) {
        const int width = shape.first, height = shape.second;
        const int out_width = width == 9 ? 23 : 9;
        const int out_height = height == 11 ? 17 : 11;
        cv::Mat source(height, width, CV_8UC3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                source.at<cv::Vec3b>(y, x) = {static_cast<unsigned char>((x * 17 + y * 13) % 256),
                                              static_cast<unsigned char>((x * 7 + y * 31) % 256),
                                              static_cast<unsigned char>((x * 43 + y * 3) % 256)};
        const auto actual = rtdetr::resize_bgr(source, out_width, out_height, rtdetr::ResizeMode::PillowBilinear);
        const std::string filename = std::string(argv[1]) + "/pillow_" + std::to_string(width) + "x" +
                                     std::to_string(height) + "_to_" + std::to_string(out_width) + "x" +
                                     std::to_string(out_height) + ".bin";
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("missing test reference: " + filename);
        std::vector<char> expected(actual.total() * actual.elemSize());
        file.read(expected.data(), expected.size());
        if (!file) throw std::runtime_error("cannot read expected bytes: " + filename);
        int largest_difference = 0;
        for (size_t i = 0; i < expected.size(); ++i)
            largest_difference = std::max(largest_difference,
                std::abs(static_cast<int>(actual.data[i]) - static_cast<int>(static_cast<unsigned char>(expected[i]))));
        if (largest_difference > 1)
            throw std::runtime_error("Pillow resize difference exceeds one byte: " + filename);
    }
    std::cout << "Pillow resize fixtures within one byte\n";
}
