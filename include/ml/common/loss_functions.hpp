#pragma once

#include <vector>

namespace ml::common {

double binary_cross_entropy(double y_true, double y_pred);
double binary_cross_entropy(const std::vector<double>& y_true,
                            const std::vector<double>& y_pred);

}  // namespace ml::common