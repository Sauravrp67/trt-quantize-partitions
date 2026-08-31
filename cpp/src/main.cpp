#include "rtdetr/runtime.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
using Samples = std::unordered_map<std::string, std::vector<double>>;
constexpr const char* stages[] = {"read", "preprocess", "engine", "postprocess", "boxes", "hud", "video_write", "display", "frame"};

double milliseconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Args {
    std::string model, engine, source, save, table, profile, classes, detections;
    float score = 0.6f;
    int max_frames = 0, warmup = 10;
    bool power = false, no_draw = false, no_hud = false, show = false;
};

void usage() {
    std::cout << "Usage: rtdetr-infer --model CONTRACT.json --engine MODEL.plan --source VIDEO|URL|CAMERA [options]\n"
              << "  --save FILE       Write annotated MP4\n"
              << "  --table FILE      Append Markdown benchmark row\n"
              << "  --profile FILE    Append per-stage timing table\n"
              << "  --detections FILE Write per-frame detections as JSONL\n"
              << "  --classes FILE    Override class names (defaults to COCO names beside the model)\n"
              << "  --score-thr X     Confidence threshold (default 0.6)\n"
              << "  --max-frames N    Limit decoded frames\n"
              << "  --warmup N        Warmup iterations (default 10)\n"
              << "  --power           Sample Jetson VDD_IN with tegrastats\n"
              << "  --no-draw         Skip boxes and HUD\n"
              << "  --no-hud          Skip HUD only\n"
              << "  --show            Display video; press q to stop\n";
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            return argv[i];
        };
        if (key == "--help" || key == "-h") { usage(); std::exit(0); }
        else if (key == "--model") a.model = value();
        else if (key == "--engine") a.engine = value();
        else if (key == "--source") a.source = value();
        else if (key == "--save") a.save = value();
        else if (key == "--table") a.table = value();
        else if (key == "--profile") a.profile = value();
        else if (key == "--detections") a.detections = value();
        else if (key == "--classes") a.classes = value();
        else if (key == "--score-thr") a.score = std::stof(value());
        else if (key == "--max-frames") a.max_frames = std::stoi(value());
        else if (key == "--warmup") a.warmup = std::stoi(value());
        else if (key == "--power") a.power = true;
        else if (key == "--no-draw") a.no_draw = true;
        else if (key == "--no-hud") a.no_hud = true;
        else if (key == "--show") a.show = true;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (a.model.empty() || a.engine.empty() || a.source.empty()) throw std::runtime_error("--model, --engine, and --source are required");
    if (a.score < 0 || a.score > 1 || a.max_frames < 0 || a.warmup < 0) throw std::runtime_error("invalid threshold/frame/warmup value");
    return a;
}

std::vector<std::string> read_classes(const std::string& path) {
    std::vector<std::string> names;
    if (path.empty()) return names;
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open classes file: " + path);
    std::string line;
    while (std::getline(file, line)) {
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#' || line.substr(first, 6) == "names:") continue;
        if (line[first] == '-') names.push_back(line.substr(line.find_first_not_of(" \t", first + 1)));
        else names.push_back(line.substr(first));
    }
    return names;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const double rank = p * (values.size() - 1);
    const size_t lo = static_cast<size_t>(rank);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (rank - lo);
}

double mean(const std::vector<double>& x) {
    return x.empty() ? 0 : std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}

void append_report(const std::string& path, const std::string& text) {
    if (path.empty()) return;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream file(path, std::ios::app);
    if (!file) throw std::runtime_error("cannot write report: " + path);
    file << text;
}

class TegraPower {
public:
    explicit TegraPower(bool enabled) {
        if (!enabled) return;
        int pipes[2];
        if (pipe(pipes) != 0) throw std::runtime_error("cannot create tegrastats pipe");
        pid_ = fork();
        if (pid_ < 0) throw std::runtime_error("cannot fork tegrastats");
        if (pid_ == 0) {
            dup2(pipes[1], STDOUT_FILENO);
            close(pipes[0]); close(pipes[1]);
            execlp("tegrastats", "tegrastats", "--interval", "100", nullptr);
            _exit(127);
        }
        close(pipes[1]);
        reader_ = std::thread([this, fd = pipes[0]] {
            FILE* stream = fdopen(fd, "r");
            if (!stream) { close(fd); return; }
            char* line = nullptr;
            size_t size = 0;
            const std::regex rail(R"(VDD_IN\s+(\d+)mW)");
            while (getline(&line, &size, stream) != -1) {
                std::cmatch match;
                if (std::regex_search(line, match, rail)) {
                    const double watts = std::stod(match[1].str()) / 1000.0;
                    latest_.store(watts);
                    std::lock_guard<std::mutex> lock(mutex_);
                    samples_.push_back(watts);
                }
            }
            free(line);
            fclose(stream);
        });
    }
    ~TegraPower() {
        if (pid_ > 0) { kill(pid_, SIGTERM); waitpid(pid_, nullptr, 0); }
        if (reader_.joinable()) reader_.join();
    }
    double latest() const { return latest_.load(); }
    double average() const { std::lock_guard<std::mutex> lock(mutex_); return mean(samples_); }
    bool measured() const { std::lock_guard<std::mutex> lock(mutex_); return !samples_.empty(); }
private:
    pid_t pid_ = -1;
    std::thread reader_;
    std::atomic<double> latest_{0};
    mutable std::mutex mutex_;
    std::vector<double> samples_;
};

void draw_boxes(cv::Mat& frame, const std::vector<rtdetr::Detection>& detections,
                const std::vector<std::string>& classes) {
    const double scale = std::clamp(std::min(frame.cols / 1280.0, frame.rows / 720.0), 1.0, 3.0);
    const int box_thickness = std::max(3, cvRound(3 * scale));
    const int text_thickness = std::max(2, cvRound(2 * scale));
    const int pad = std::max(4, cvRound(4 * scale));
    const double font_scale = 0.72 * scale;
    const cv::Scalar palette[] = {
        {0, 215, 255}, {80, 220, 80}, {255, 190, 40}, {235, 100, 255},
        {70, 170, 255}, {255, 120, 120}, {100, 230, 200}, {200, 180, 255}
    };
    struct VisibleBox {
        const rtdetr::Detection* detection;
        cv::Rect box;
        cv::Scalar color;
    };
    std::vector<VisibleBox> visible;
    visible.reserve(detections.size());
    for (const auto& d : detections) {
        const cv::Scalar color = palette[static_cast<size_t>(std::max(0, d.label)) % std::size(palette)];
        const cv::Rect bounds(0, 0, frame.cols, frame.rows);
        const cv::Rect box = cv::Rect(cv::Point(cvRound(d.box.x), cvRound(d.box.y)),
                                     cv::Point(cvRound(d.box.x + d.box.width), cvRound(d.box.y + d.box.height))) & bounds;
        if (box.empty()) continue;
        cv::rectangle(frame, box, cv::Scalar(15, 15, 15), box_thickness + 2, cv::LINE_8);
        cv::rectangle(frame, box, color, box_thickness, cv::LINE_8);
        visible.push_back({&d, box, color});
    }
    std::vector<cv::Rect> occupied;
    for (const auto& item : visible) {
        const auto& d = *item.detection;
        const auto& box = item.box;
        const std::string label = d.label >= 0 && static_cast<size_t>(d.label) < classes.size()
            ? classes[d.label] : std::to_string(d.label);
        char score[32];
        snprintf(score, sizeof(score), " %.2f", d.score);
        const std::string caption = label + score;
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(caption, cv::FONT_HERSHEY_SIMPLEX,
                                                   font_scale, text_thickness, &baseline);
        const int panel_w = std::min(frame.cols, text_size.width + 2 * pad);
        const int panel_h = std::min(frame.rows, text_size.height + baseline + 2 * pad);
        const cv::Point candidates[] = {
            {box.x, box.y - panel_h}, {box.x, box.y},
            {box.br().x - panel_w, box.y - panel_h}, {box.br().x - panel_w, box.y},
            {box.x, box.br().y}, {box.br().x - panel_w, box.br().y}
        };
        cv::Rect chosen;
        int best_overlap = std::numeric_limits<int>::max();
        for (const auto& candidate : candidates) {
            const cv::Rect panel(std::clamp(candidate.x, 0, frame.cols - panel_w),
                                 std::clamp(candidate.y, 0, frame.rows - panel_h), panel_w, panel_h);
            int overlap = 0;
            for (const auto& other : occupied) overlap += (panel & other).area();
            if (overlap < best_overlap) { chosen = panel; best_overlap = overlap; }
            if (overlap == 0) break;
        }
        occupied.push_back(chosen);
        cv::rectangle(frame, chosen, cv::Scalar(15, 15, 15), cv::FILLED);
        cv::rectangle(frame, chosen, item.color, std::max(2, cvRound(scale)), cv::LINE_8);
        cv::putText(frame, caption, {chosen.x + pad, chosen.y + pad + text_size.height},
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255),
                    text_thickness, cv::LINE_AA);
    }
}

void draw_hud(cv::Mat& frame, int frame_index, size_t detections, double engine_ms,
              double loop_ms, double watts, bool power_enabled) {
    const double scale = std::clamp(std::min(frame.cols / 1280.0, frame.rows / 720.0), 1.0, 3.0);
    const int pad = std::max(8, cvRound(12 * scale));
    const int gap = std::max(5, cvRound(8 * scale));
    const int margin = std::max(8, cvRound(10 * scale));
    const int thickness = std::max(2, cvRound(2 * scale));
    const double font_scale = 0.8 * scale;
    char line[180];
    snprintf(line, sizeof(line), "%.1f FPS  engine %.1f ms  loop %.1f ms", 1000 / std::max(loop_ms, 0.001), engine_ms, loop_ms);
    const std::string top = line;
    if (power_enabled) snprintf(line, sizeof(line), "frame %d  objects %zu  power %.1f W", frame_index, detections, watts);
    else snprintf(line, sizeof(line), "frame %d  objects %zu", frame_index, detections);
    const std::string bottom = line;
    int baseline = 0;
    const cv::Size top_size = cv::getTextSize(top, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    const cv::Size bottom_size = cv::getTextSize(bottom, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    const int panel_w = std::min(frame.cols - margin, std::max(top_size.width, bottom_size.width) + 2 * pad);
    const int panel_h = top_size.height + bottom_size.height + gap + 2 * pad + baseline;
    cv::rectangle(frame, cv::Rect(margin, margin, panel_w, panel_h), cv::Scalar(15, 15, 15), cv::FILLED);
    cv::putText(frame, top, {margin + pad, margin + pad + top_size.height},
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    cv::putText(frame, bottom, {margin + pad, margin + pad + top_size.height + gap + bottom_size.height},
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(80, 220, 255), thickness, cv::LINE_AA);
}

void run(const Args& args) {
    const auto spec = rtdetr::load_model(args.model);
    rtdetr::Detector detector(spec, args.engine);
    std::string classes_path = args.classes;
    if (classes_path.empty() && spec.classes == 80) {
        const auto bundled = std::filesystem::path(args.model).parent_path() / "coco80.yaml";
        if (std::filesystem::exists(bundled)) classes_path = bundled.string();
    }
    std::vector<std::string> classes = read_classes(classes_path);
    cv::VideoCapture cap;
    if (!args.source.empty() && std::all_of(args.source.begin(), args.source.end(),
                                            [](unsigned char ch) { return std::isdigit(ch); }))
        cap.open(std::stoi(args.source));
    else cap.open(args.source);
    if (!cap.isOpened()) throw std::runtime_error("cannot open source: " + args.source);
    cv::VideoWriter writer;
    std::ofstream detections_file;
    if (!args.detections.empty()) {
        const auto parent = std::filesystem::path(args.detections).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        detections_file.open(args.detections);
        if (!detections_file) throw std::runtime_error("cannot write detections: " + args.detections);
    }
    double source_fps = cap.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(source_fps) || source_fps <= 0) source_fps = 30;
    detector.warmup(args.warmup);
    TegraPower power(args.power);
    Samples samples;
    for (const char* stage : stages) samples[stage] = {};
    size_t detection_count = 0;
    const auto wall_start = Clock::now();
    int frames = 0;
    while (args.max_frames == 0 || frames < args.max_frames) {
        const auto t0 = Clock::now();
        cv::Mat frame;
        if (!cap.read(frame)) break;
        const auto t1 = Clock::now();
        rtdetr::StageTimes infer_times;
        const auto detections = detector.infer(frame, args.score, &infer_times);
        detection_count += detections.size();
        if (detections_file) {
            detections_file << "{\"frame\":" << frames << ",\"detections\":[";
            for (size_t j = 0; j < detections.size(); ++j) {
                const auto& d = detections[j];
                if (j) detections_file << ',';
                detections_file << std::setprecision(8) << "{\"label\":" << d.label
                    << ",\"score\":" << d.score << ",\"xyxy\":[" << d.box.x << ',' << d.box.y
                    << ',' << d.box.x + d.box.width << ',' << d.box.y + d.box.height << "]}";
            }
            detections_file << "]}\n";
        }
        const auto t4 = Clock::now();
        if (!args.no_draw) draw_boxes(frame, detections, classes);
        const auto t5 = Clock::now();
        if (!args.no_draw && !args.no_hud) {
            const double prev_loop = samples["frame"].empty() ? milliseconds(t0, t5) : samples["frame"].back();
            draw_hud(frame, frames + 1, detections.size(), infer_times.engine_ms, prev_loop,
                     power.latest(), power.measured());
        }
        const auto t6 = Clock::now();
        if (!args.save.empty()) {
            if (!writer.isOpened()) {
                const auto parent = std::filesystem::path(args.save).parent_path();
                if (!parent.empty()) std::filesystem::create_directories(parent);
                writer.open(args.save, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), source_fps, frame.size());
                if (!writer.isOpened()) throw std::runtime_error("cannot open video writer: " + args.save);
            }
            writer.write(frame);
        }
        const auto t7 = Clock::now();
        bool stop = false;
        if (args.show) {
            cv::imshow("RT-DETR C++", frame);
            stop = cv::waitKey(1) == 'q';
        }
        const auto t8 = Clock::now();
        samples["read"].push_back(milliseconds(t0, t1));
        samples["preprocess"].push_back(infer_times.preprocess_ms);
        samples["engine"].push_back(infer_times.engine_ms);
        samples["postprocess"].push_back(infer_times.postprocess_ms);
        samples["boxes"].push_back(milliseconds(t4, t5));
        samples["hud"].push_back(milliseconds(t5, t6));
        samples["video_write"].push_back(milliseconds(t6, t7));
        samples["display"].push_back(milliseconds(t7, t8));
        samples["frame"].push_back(milliseconds(t0, t8));
        ++frames;
        if (frames % 50 == 0) std::cout << frames << " frames, engine p50 " << percentile(samples["engine"], 0.5) << " ms\n";
        if (stop) break;
    }
    const double wall = milliseconds(wall_start, Clock::now()) / 1000.0;
    if (frames == 0) throw std::runtime_error("no frames decoded from source");
    const double engine_p50 = percentile(samples["engine"], 0.5);
    std::vector<double> e2e;
    for (int i = 0; i < frames; ++i) e2e.push_back(samples["frame"][i] - samples["read"][i]);
    const double e2e_p50 = percentile(e2e, 0.5);
    const double fps = frames / wall;
    std::cout << std::fixed << std::setprecision(2)
              << "model=" << spec.name << " version=" << spec.version << " frames=" << frames
              << " engine_p50_ms=" << engine_p50 << " engine_p90_ms=" << percentile(samples["engine"], 0.9)
              << " e2e_p50_ms=" << e2e_p50 << " pipeline_fps=" << fps
              << " detections_per_frame=" << static_cast<double>(detection_count) / frames
              << " mean_vdd_in_w=";
    if (power.measured()) std::cout << power.average();
    else std::cout << "n/a";
    std::cout << '\n';
    if (!args.table.empty()) {
        if (!std::filesystem::exists(args.table)) append_report(args.table,
            "| Model | Version | Engine | Frames | Engine p50 ms | Engine p90 ms | E2E p50 ms | Pipeline FPS | VDD_IN W |\n"
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        std::ostringstream row;
        row << std::fixed << std::setprecision(2) << "| " << spec.name << " | " << spec.version << " | "
            << std::filesystem::path(args.engine).filename().string() << " | " << frames << " | "
            << engine_p50 << " | " << percentile(samples["engine"], 0.9) << " | " << e2e_p50
            << " | " << fps << " | ";
        if (power.measured()) row << power.average();
        else row << "n/a";
        row << " |\n";
        append_report(args.table, row.str());
    }
    if (!args.profile.empty()) {
        std::ostringstream text;
        text << "\n## " << spec.name << " - " << args.source << "\n\n"
             << "| Stage | Mean ms | p50 ms | p90 ms |\n| --- | ---: | ---: | ---: |\n";
        for (const char* stage : stages)
            text << "| " << stage << " | " << std::fixed << std::setprecision(2) << mean(samples[stage])
                 << " | " << percentile(samples[stage], 0.5) << " | " << percentile(samples[stage], 0.9) << " |\n";
        append_report(args.profile, text.str());
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
