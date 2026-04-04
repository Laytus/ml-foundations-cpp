#pragma once

#include <vector>

namespace ml::common {

struct Dataset {
    std::vector<std::vector<double>> features;
    std::vector<double> targets;
};

} // namespace ml::common