#include "ml/linear_regression/linear_regression.hpp"

namespace ml {

LinearRegression::LinearRegression()
    : weight_(0.0), bias_(0.0) {}

double LinearRegression::predict(double x) const {
    return (weight_ * x) + bias_; // y_hat = w*x + b
}

std::vector<double> LinearRegression::predict(const std::vector<double>& xs) const {
    std::vector<double> predictions;
    predictions.reserve(xs.size());

    for (double x : xs) {
        predictions.push_back(predict(x));
    }

    return predictions;
}

double LinearRegression::weight() const {
    return weight_;
}

double LinearRegression::bias() const {
    return bias_;
}

void LinearRegression::set_parameters(double weight, double bias) {
    weight_ = weight;
    bias_ = bias;
}

} // namespace ml