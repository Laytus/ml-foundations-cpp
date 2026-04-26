#include "ml/common/feature_scaling.hpp"
#include "ml/common/vector_ops.hpp"

#include <stdexcept>
#include <algorithm>

namespace ml::common {

std::vector<double> min_max_scale(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("min_max_scale: values must not be empty");
    }

    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    double min_value = *min_it;
    double max_value = *max_it;

    double range = max_value - min_value;

    if (range == 0.0) {
        throw std::invalid_argument("min_max_scale: range cannot be zero");
    }

    std::vector<double> result;
    result.reserve(values.size());

    for (double value : values) {
        result.push_back((value - min_value) / range);
    }

    return result;
}

std::vector<double> standardize(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("standardize: values must not be empty");
    }

    double mean = ml::common::mean(values);
    double sd = ml::common::population_standard_deviation(values, mean);

    if (sd == 0.0) {
        throw std::invalid_argument("standardize: standard deviation cannot be zero");
    }

    std::vector<double> result;
    result.reserve(values.size());

    for (double value : values) {
        result.push_back((value - mean) / sd);
    }

    return result;
}

}  // namespace common