#include "rtdetr/runtime.hpp"
#include "rtdetr/preprocess.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace rtdetr {
namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void cuda_check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) fprintf(stderr, "[TensorRT] %s\n", message);
    }
};

template <class T> struct TrtDelete {
    void operator()(T* value) const { delete value; }
};

size_t type_size(nvinfer1::DataType type) {
    using D = nvinfer1::DataType;
    switch (type) {
        case D::kFLOAT: return 4;
        case D::kHALF: return 2;
        case D::kINT8: return 1;
        case D::kINT32: return 4;
        case D::kINT64: return 8;
        case D::kBOOL: return 1;
        default: throw std::runtime_error("unsupported tensor data type");
    }
}

size_t volume(const nvinfer1::Dims& dims) {
    size_t value = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) throw std::runtime_error("unresolved or zero tensor shape");
        value *= static_cast<size_t>(dims.d[i]);
    }
    return value;
}

float scalar(const void* ptr, nvinfer1::DataType dtype, size_t i) {
    using D = nvinfer1::DataType;
    switch (dtype) {
        case D::kFLOAT: return static_cast<const float*>(ptr)[i];
        case D::kHALF: return __half2float(static_cast<const __half*>(ptr)[i]);
        case D::kINT32: return static_cast<float>(static_cast<const int32_t*>(ptr)[i]);
        case D::kINT64: return static_cast<float>(static_cast<const int64_t*>(ptr)[i]);
        case D::kINT8: return static_cast<float>(static_cast<const int8_t*>(ptr)[i]);
        case D::kBOOL: return static_cast<float>(static_cast<const bool*>(ptr)[i]);
        default: throw std::runtime_error("unsupported scalar data type");
    }
}

void put_scalar(void* ptr, nvinfer1::DataType dtype, size_t i, float value) {
    using D = nvinfer1::DataType;
    switch (dtype) {
        case D::kFLOAT: static_cast<float*>(ptr)[i] = value; return;
        case D::kHALF: static_cast<__half*>(ptr)[i] = __float2half(value); return;
        case D::kINT32: static_cast<int32_t*>(ptr)[i] = static_cast<int32_t>(value); return;
        case D::kINT64: static_cast<int64_t*>(ptr)[i] = static_cast<int64_t>(value); return;
        default: throw std::runtime_error("unsupported input tensor data type");
    }
}

float sigmoid(float x) {
    return x >= 0 ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
}

cv::Rect2f xyxy(float x0, float y0, float x1, float y1) {
    return {x0, y0, x1 - x0, y1 - y0};
}
}  // namespace

struct Detector::Impl {
    struct Buffer {
        std::string name;
        nvinfer1::DataType dtype;
        nvinfer1::Dims dims;
        bool input;
        size_t elements = 0;
        size_t bytes = 0;
        void* host = nullptr;
        void* device = nullptr;

        Buffer(std::string n, nvinfer1::DataType t, nvinfer1::Dims d, bool in)
            : name(std::move(n)), dtype(t), dims(d), input(in) {}
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept
            : name(std::move(other.name)), dtype(other.dtype), dims(other.dims), input(other.input),
              elements(other.elements), bytes(other.bytes), host(other.host), device(other.device) {
            other.host = nullptr;
            other.device = nullptr;
        }
        ~Buffer() {
            if (device) cudaFree(device);
            if (host) cudaFreeHost(host);
        }
    };

    ModelSpec spec;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDelete<nvinfer1::IRuntime>> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDelete<nvinfer1::ICudaEngine>> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDelete<nvinfer1::IExecutionContext>> context;
    cudaStream_t stream = nullptr;
    std::unordered_map<std::string, Buffer> buffers;

    Impl(ModelSpec model, const std::string& path) : spec(std::move(model)) {
        validate_model(spec);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("cannot open TensorRT engine: " + path);
        const auto length = file.tellg();
        if (length <= 0) throw std::runtime_error("empty TensorRT engine: " + path);
        std::vector<char> blob(static_cast<size_t>(length));
        file.seekg(0);
        file.read(blob.data(), length);
        if (!file) throw std::runtime_error("cannot read TensorRT engine: " + path);
        initLibNvInferPlugins(&logger, "");
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("TensorRT runtime initialization failed");
        engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine) throw std::runtime_error("engine deserialization failed; build it on this GPU/TensorRT version");
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("TensorRT execution context creation failed");
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const std::string name = engine->getIOTensorName(i);
            const bool input = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
            buffers.emplace(name, Buffer{name, engine->getTensorDataType(name.c_str()),
                                         engine->getTensorShape(name.c_str()), input});
        }
        const auto& image = require(spec.image_input, true);
        if (image.dtype != nvinfer1::DataType::kFLOAT && image.dtype != nvinfer1::DataType::kHALF)
            throw std::runtime_error("image I/O must be FP32 or FP16; quantized engine internals are supported through the plan");
        if (image.dims.nbDims != 4 || (image.dims.d[0] != 1 && image.dims.d[0] != -1) ||
            image.dims.d[1] != 3 || image.dims.d[2] != spec.height || image.dims.d[3] != spec.width)
            throw std::runtime_error("image input does not match manifest NCHW shape");
        if (image.dims.d[0] == -1) {
            auto shape = image.dims;
            shape.d[0] = 1;
            if (!context->setInputShape(spec.image_input.c_str(), shape))
                throw std::runtime_error("cannot set image batch shape");
        }
        for (const auto& aux : spec.auxiliary_inputs) {
            auto& b = require(aux.name, true);
            if (b.dims.nbDims != 2 || (b.dims.d[0] != 1 && b.dims.d[0] != -1) || b.dims.d[1] != 2)
                throw std::runtime_error("auxiliary input must have shape [1,2]: " + aux.name);
            if (b.dims.d[0] == -1) {
                auto shape = b.dims;
                shape.d[0] = 1;
                if (!context->setInputShape(aux.name.c_str(), shape))
                    throw std::runtime_error("cannot set auxiliary input shape: " + aux.name);
            }
        }
        std::unordered_set<std::string> expected_inputs{spec.image_input};
        for (const auto& aux : spec.auxiliary_inputs) expected_inputs.insert(aux.name);
        for (const auto& entry : buffers) {
            if (entry.second.input && !expected_inputs.count(entry.first))
                throw std::runtime_error("engine has input not covered by model contract: " + entry.first);
        }
        require(spec.boxes_output, false);
        if (spec.output_kind == OutputKind::RawQueries) require(spec.logits_output, false);
        if (spec.output_kind == OutputKind::Decoded) {
            require(spec.labels_output, false);
            require(spec.scores_output, false);
        }
        if (spec.output_kind == OutputKind::PaddleRows) require(spec.count_output, false);

        for (auto& entry : buffers) {
            auto& b = entry.second;
            b.dims = context->getTensorShape(b.name.c_str());
            b.elements = volume(b.dims);
            b.bytes = b.elements * type_size(b.dtype);
            cuda_check(cudaMallocHost(&b.host, b.bytes), "cudaMallocHost");
            cuda_check(cudaMalloc(&b.device, b.bytes), "cudaMalloc");
            if (!context->setTensorAddress(b.name.c_str(), b.device))
                throw std::runtime_error("cannot bind tensor: " + b.name);
        }
    }

    ~Impl() {
        if (stream) cudaStreamDestroy(stream);
    }

    Buffer& require(const std::string& name, bool input) {
        auto it = buffers.find(name);
        if (it == buffers.end() || it->second.input != input)
            throw std::runtime_error("engine missing " + std::string(input ? "input: " : "output: ") + name);
        return it->second;
    }

    void fill_image(const cv::Mat& bgr) {
        auto& tensor = require(spec.image_input, true);
        cv::Mat resized = resize_bgr(bgr, spec.width, spec.height, spec.resize_mode), rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        for (int y = 0; y < spec.height; ++y) {
            const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < spec.width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const size_t index = static_cast<size_t>(c) * spec.width * spec.height + y * spec.width + x;
                    put_scalar(tensor.host, tensor.dtype, index, row[x][c] / 255.0f);
                }
            }
        }
    }

    void fill_auxiliary(const cv::Mat& bgr) {
        for (const auto& aux : spec.auxiliary_inputs) {
            auto& tensor = require(aux.name, true);
            float a = 1, b = 1;
            switch (aux.value) {
                case AuxiliaryValue::OriginalWH: a = static_cast<float>(bgr.cols); b = static_cast<float>(bgr.rows); break;
                case AuxiliaryValue::OriginalHW: a = static_cast<float>(bgr.rows); b = static_cast<float>(bgr.cols); break;
                case AuxiliaryValue::InputHW: a = static_cast<float>(spec.height); b = static_cast<float>(spec.width); break;
                case AuxiliaryValue::ResizeScaleYX: a = static_cast<float>(spec.height) / bgr.rows;
                    b = static_cast<float>(spec.width) / bgr.cols; break;
                case AuxiliaryValue::UnitScale: break;
            }
            put_scalar(tensor.host, tensor.dtype, 0, a);
            put_scalar(tensor.host, tensor.dtype, 1, b);
        }
    }

    std::vector<Detection> decode(const cv::Mat& image, float threshold) {
        std::vector<Detection> result;
        auto& boxes = require(spec.boxes_output, false);
        if (spec.output_kind == OutputKind::RawQueries) {
            auto& logits = require(spec.logits_output, false);
            if (boxes.elements % 4 != 0 || logits.elements != boxes.elements / 4 * static_cast<size_t>(spec.classes))
                throw std::runtime_error("raw output shapes do not match query/class contract");
            const size_t queries = boxes.elements / 4;
            struct Candidate { float score; size_t query; int label; };
            std::vector<Candidate> candidates;
            candidates.reserve(queries * spec.classes);
            for (size_t q = 0; q < queries; ++q) {
                for (int c = 0; c < spec.classes; ++c) {
                    float score = sigmoid(scalar(logits.host, logits.dtype, q * spec.classes + c));
                    if (score >= threshold) candidates.push_back({score, q, c});
                }
            }
            const size_t keep = std::min(candidates.size(), static_cast<size_t>(spec.top_k));
            std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
            for (size_t i = 0; i < keep; ++i) {
                const auto& d = candidates[i];
                const size_t p = d.query * 4;
                const float cx = scalar(boxes.host, boxes.dtype, p) * image.cols;
                const float cy = scalar(boxes.host, boxes.dtype, p + 1) * image.rows;
                const float w = scalar(boxes.host, boxes.dtype, p + 2) * image.cols;
                const float h = scalar(boxes.host, boxes.dtype, p + 3) * image.rows;
                result.push_back({d.label, d.score, xyxy(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)});
            }
        } else if (spec.output_kind == OutputKind::Decoded) {
            auto& labels = require(spec.labels_output, false);
            auto& scores = require(spec.scores_output, false);
            if (boxes.elements % 4 || labels.elements != boxes.elements / 4 || scores.elements != labels.elements)
                throw std::runtime_error("decoded output shapes mismatch");
            for (size_t i = 0; i < labels.elements; ++i) {
                const float score = scalar(scores.host, scores.dtype, i);
                if (score < threshold) continue;
                const int label = static_cast<int>(scalar(labels.host, labels.dtype, i));
                const size_t p = i * 4;
                const float sx = spec.boxes_normalized ? image.cols : 1;
                const float sy = spec.boxes_normalized ? image.rows : 1;
                result.push_back({label, score, xyxy(scalar(boxes.host, boxes.dtype, p) * sx,
                    scalar(boxes.host, boxes.dtype, p + 1) * sy,
                    scalar(boxes.host, boxes.dtype, p + 2) * sx,
                    scalar(boxes.host, boxes.dtype, p + 3) * sy)});
            }
        } else {
            auto& count = require(spec.count_output, false);
            if (boxes.elements % 6 || count.elements < 1) throw std::runtime_error("Paddle output shapes mismatch");
            const size_t n = std::min(static_cast<size_t>(std::max(0.0f, scalar(count.host, count.dtype, 0))), boxes.elements / 6);
            const float sx = spec.paddle_boxes_input_pixels ? static_cast<float>(image.cols) / spec.width : 1.0f;
            const float sy = spec.paddle_boxes_input_pixels ? static_cast<float>(image.rows) / spec.height : 1.0f;
            for (size_t i = 0; i < n; ++i) {
                const size_t p = i * 6;
                const float score = scalar(boxes.host, boxes.dtype, p + 1);
                if (score < threshold) continue;
                const int label = static_cast<int>(scalar(boxes.host, boxes.dtype, p));
                result.push_back({label, score, xyxy(scalar(boxes.host, boxes.dtype, p + 2) * sx,
                    scalar(boxes.host, boxes.dtype, p + 3) * sy,
                    scalar(boxes.host, boxes.dtype, p + 4) * sx,
                    scalar(boxes.host, boxes.dtype, p + 5) * sy)});
            }
        }
        std::sort(result.begin(), result.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
        return result;
    }

    std::vector<Detection> infer(const cv::Mat& bgr, float threshold, StageTimes* times) {
        if (bgr.empty() || bgr.type() != CV_8UC3) throw std::runtime_error("infer expects a nonempty BGR uint8 image");
        if (threshold < 0 || threshold > 1) throw std::runtime_error("score threshold must be in [0,1]");
        const auto start = Clock::now();
        fill_image(bgr);
        fill_auxiliary(bgr);
        const auto ready = Clock::now();
        for (auto& entry : buffers) {
            auto& b = entry.second;
            if (b.input) cuda_check(cudaMemcpyAsync(b.device, b.host, b.bytes, cudaMemcpyHostToDevice, stream), "H2D copy");
        }
        if (!context->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueueV3 failed");
        for (auto& entry : buffers) {
            auto& b = entry.second;
            if (!b.input) cuda_check(cudaMemcpyAsync(b.host, b.device, b.bytes, cudaMemcpyDeviceToHost, stream), "D2H copy");
        }
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        const auto executed = Clock::now();
        auto result = decode(bgr, threshold);
        const auto done = Clock::now();
        if (times) {
            times->preprocess_ms = elapsed_ms(start, ready);
            times->engine_ms = elapsed_ms(ready, executed);
            times->postprocess_ms = elapsed_ms(executed, done);
        }
        return result;
    }
};

Detector::Detector(ModelSpec spec, const std::string& engine_path)
    : impl_(std::make_unique<Impl>(std::move(spec), engine_path)) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;
std::vector<Detection> Detector::infer(const cv::Mat& frame, float threshold, StageTimes* times) {
    return impl_->infer(frame, threshold, times);
}
void Detector::warmup(int count) {
    if (count < 0) throw std::runtime_error("warmup count must be nonnegative");
    cv::Mat black(impl_->spec.height, impl_->spec.width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < count; ++i) impl_->infer(black, 1.0f, nullptr);
}
const ModelSpec& Detector::spec() const { return impl_->spec; }

}  // namespace rtdetr
