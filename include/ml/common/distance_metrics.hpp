#pragma once

#include <vector>

namespace ml::common {

double euclidean_distance(double a, double b);

double euclidean_distance(const std::vector<double>& x,
                            const std::vector<double>& y);

}  // namespace ml::common