#include "ml/common/loss_functions.hpp"

#include <stdexcept>
#include <cmath>

namespace ml::common {

double binary_cross_entropy(double y_true, double y_pred) {
    if (y_true != 0.0 && y_true != 1.0) {
        throw std::invalid_argument("binary_cross_entropy: y_true must be either 0.0 or 1.0");
    }

    if (y_pred <= 0.0 || y_pred >= 1.0) {
        throw std::invalid_argument("binary_cross_entropy: y_pred must be strictly between 0.0 and 1.0");
    }

    return -(y_true * std::log(y_pred) + (1 - y_true) * std::log(1 - y_pred));
}

double binary_cross_entropy(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("binary_cross_entropy: vectors must have the same size");
    }

    if (y_true.empty()) {
        throw std::invalid_argument("binary_cross_entropy: vectors must not be empty");
    }

    double sum = 0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        sum += binary_cross_entropy(y_true[i], y_pred[i]);
    }

    return sum / y_true.size();
}

}  // namespace ml::common