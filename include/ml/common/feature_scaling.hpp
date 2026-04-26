#pragma once

#include <vector>

namespace ml::common {

std::vector<double> min_max_scale(const std::vector<double>& values);
std::vector<double> standardize(const std::vector<double>& values);

} // namespace ml::common