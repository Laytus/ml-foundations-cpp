#include "ml/logistic_regression/logistic_regression.hpp"
#include "ml/common/activation_functions.hpp"

namespace ml {
 
LogisticRegression::LogisticRegression()
    : weight_(0.0), bias_(0.0) {}

double LogisticRegression::predict_proba(double x) const {
    double z = (weight_ * x) + bias_;

    return ml::common::sigmoid(z);
}

std::vector<double> LogisticRegression::predict_proba(std::vector<double>& x_values) const {
    if (x_values.empty()) {
        throw std::invalid_argument("predict_proba: vector must not be empty");
    }

    std::vector<double> predictions;
    predictions.reserve(x_values.size());

    for (double x : x_values) {
        predictions.push_back(predict_proba(x));
    }

    return predictions;
}

int LogisticRegression::predict(double x) const {
    double threshold = 0.5;
    return ml::common::binary_predict(predict_proba(x), threshold);
}

std::vector<int> LogisticRegression::predict(std::vector<double>& x_values) const {
    if (x_values.empty()) {
        throw std::invalid_argument("predict: vector must not be empty");
    }

    std::vector<int> predictions;
    predictions.reserve(x_values.size());

    for (double x : x_values) {
        predictions.push_back(predict(x));
    }

    return predictions;
}

std::pair<double, double> LogisticRegression::gradient(const std::vector<double>& x_values, const std::vector<double>& y_values) const {
    if (x_values.size() != y_values.size()) {
        throw std::invalid_argument("gradient: vectors must have the same size");
    }

    if (x_values.empty()) {
        throw std::invalid_argument("gradient: vectors must not be empty");
    }

    double sum_w = 0.0;
    double sum_b = 0.0;
    
    for (std::size_t i = 0; i < x_values.size(); ++i) {
        double error = predict_proba(x_values[i]) - y_values[i];
        sum_w += (error) * x_values[i];
        sum_b += error;
    }

    double gradient_w = sum_w / x_values.size();
    double gradient_b = sum_b / x_values.size();

    return {gradient_w, gradient_b};
}

void LogisticRegression::gradient_descent_step(const std::vector<double>& x_values, const std::vector<double>& y_values, double learning_rate) {
    if (x_values.size() != y_values.size()) {
        throw std::invalid_argument("gradient: vectors must have the same size");
    }

    if (x_values.empty()) {
        throw std::invalid_argument("gradient: vectors must not be empty");
    }
    
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("gradient: learning_rate must be greater than zero");
    }

    auto [gradient_w, gradient_b] = gradient(x_values, y_values);
    weight_ -= learning_rate * gradient_w;
    bias_ -= learning_rate * gradient_b;
}

void LogisticRegression::fit_batch(const std::vector<double>& x_values, const std::vector<double>& y_values, double learning_rate, std::size_t epochs) {
    if (x_values.size() != y_values.size()) {
        throw std::invalid_argument("fit_batch: vectors must have the same size");
    }

    if (x_values.empty()) {
        throw std::invalid_argument("fit_batch: vectors must not be empty");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("fit_batch: learning_rate must be greater than zero");
    }

    if (epochs == 0) {
        throw std::invalid_argument("fit_batch: epochs must be greater than zero");
    }

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        gradient_descent_step(x_values, y_values, learning_rate);
    }
}

double LogisticRegression::weight() const {
    return weight_;
}

double LogisticRegression::bias() const {
    return bias_;
}

void LogisticRegression::set_parameters(double weight, double bias) {
    weight_ = weight;
    bias_ = bias;
}

}  // namespace ml