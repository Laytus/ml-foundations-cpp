#include "ml/common/distance_metrics.hpp"

#include <stdexcept>
#include <cmath>

namespace ml::common {

double euclidean_distance(double a, double b) {
    return std::abs(a - b);
}

double euclidean_distance(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("euclidean_distance: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("euclidean_distance: vectors must not be empty");
    }

    double sum = 0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }

    return std::sqrt(sum);
}

}  // namespace ml::common