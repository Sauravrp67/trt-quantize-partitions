#include "rtdetr/model.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string base = argv[1];
    for (const std::string name : {"v1-r101", "v2-r50", "v3-r18", "v4-s"}) {
        auto spec = rtdetr::load_model(base + "/" + name + ".json");
        assert(spec.width == 640 && spec.height == 640 && spec.classes == 80);
        assert(!spec.image_input.empty() && !spec.boxes_output.empty());
    }
    bool rejected = false;
    try { rtdetr::load_model(base + "/nonexistent.json"); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
    std::cout << "model manifests parsed and validated\n";
}
