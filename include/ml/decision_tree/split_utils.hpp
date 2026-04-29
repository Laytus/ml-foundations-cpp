#pragma once

#include <vector>

namespace ml {

struct SplitResult {
    std::vector<double> left_x;
    std::vector<int> left_y;
    std::vector<double> right_x;
    std::vector<int> right_y;
};

SplitResult split_dataset(const std::vector<double>& x_values,
                          const std::vector<int>& y_values,
                          double threshold);

}  // namespace ml