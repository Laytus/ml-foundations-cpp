#include "ml/common/vector_ops.hpp"

#include <stdexcept>

namespace ml::common {

std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("add: vectors must have the same size");
    }

    std::vector<double> result(a.size());
    
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("subtract: vectors must have the same size");
    }

    std::vector<double> result(a.size());
    
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}

std::vector<double> scalar_multiply(const std::vector<double>& v, double scalar) {
    std::vector<double> result(v.size());

    for (std::size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }

    return result;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot: vectors must have the same size");
    }

    double result = 0.0;
    
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("mean: vector must not be empty");
    }

    double sum = 0.0;

    for (double v : values) {
        sum += v;
    }

    return sum / values.size();
}

} // namespace ml::common