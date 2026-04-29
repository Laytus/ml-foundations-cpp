#include "ml/decision_tree/split_utils.hpp"

#include <stdexcept>
#include <vector>

namespace ml {

SplitResult split_dataset(const std::vector<double>& x_values, const std::vector<int>& y_values, double threshold) {
    if (x_values.size() != y_values.size()) {
        throw std::invalid_argument("split_dataset: vectors must have the same size");
    }

    if (x_values.empty()) {
        throw std::invalid_argument("split_dataset: vectors must not be empty");
    }

    SplitResult result;

    for (std::size_t i = 0; i < x_values.size(); ++i) {
        if (y_values[i] != 0 && y_values[i] != 1) {
            throw std::invalid_argument("split_dataset: y_values must be either 0 or 1");
        }

        if (x_values[i] < threshold) {
            result.left_x.push_back(x_values[i]);
            result.left_y.push_back(y_values[i]);
        } else {
            result.right_x.push_back(x_values[i]);
            result.right_y.push_back(y_values[i]);
        }
    }

    return result;
}

}  // namespace ml