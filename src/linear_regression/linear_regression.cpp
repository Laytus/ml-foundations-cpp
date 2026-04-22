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

double LinearRegression::mse(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size()) {
        throw std::invalid_argument("mse: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("mse: vectors must not be empty");
    }

    double sum = 0.0;
    
    for (std::size_t i = 0; i < y.size(); ++i) {
        double error = predict(x[i]) - y[i]; 
        sum += error * error;
    }

    return sum / y.size();
}

std::pair<double, double> LinearRegression::gradient(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size()) {
        throw std::invalid_argument("gradient: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("gradient: vectors must not be empty");
    }

    double sum_w = 0.0;
    double sum_b = 0.0;
    
    for (std::size_t i = 0; i < y.size(); ++i) {
        double error = predict(x[i]) - y[i];
        sum_w += (error) * x[i];
        sum_b += error;
    }

    double gradient_w = 2 * sum_w / y.size();
    double gradient_b = 2 * sum_b / y.size();

    return {gradient_w, gradient_b};
}

void LinearRegression::gradient_descent_step(const std::vector<double>& x, const std::vector<double>& y, double learning_rate){
    if (x.size() != y.size()) {
        throw std::invalid_argument("gradient_descent_step: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("gradient_descent_step: vectors must not be empty");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("gradient_descent_step: learning_rate must be greater than zero");
    }

    auto [gradient_w, gradient_b] = gradient(x, y);
    weight_ -= learning_rate * gradient_w;
    bias_ -= learning_rate * gradient_b;
}

void LinearRegression::fit_batch(const std::vector<double>& x, const std::vector<double>& y, double learning_rate, std::size_t epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("fit_batch: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("fit_batch: vectors must not be empty");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("fit_batch: learning_rate must be greater than zero");
    }

    if (epochs == 0) {
        throw std::invalid_argument("fit_batch: epochs must be greater than zero");
    }

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        gradient_descent_step(x, y, learning_rate);
    }
}

std::vector<TrainingSnapshot> LinearRegression::fit_batch_with_history(const std::vector<double>& x, const std::vector<double>& y, double learning_rate, std::size_t epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("fit_batch_with_history: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("fit_batch_with_history: vectors must not be empty");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("fit_batch_with_history: learning_rate must be greater than zero");
    }

    if (epochs == 0) {
        throw std::invalid_argument("fit_batch_with_history: epochs must be greater than zero");
    }

    std::vector<TrainingSnapshot> result;
    result.reserve(epochs);

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        gradient_descent_step(x, y, learning_rate);

        TrainingSnapshot snapshot; 
        snapshot.epoch = epoch + 1;
        snapshot.mse = mse(x, y);
        snapshot.weight = weight();
        snapshot.bias = bias();

        result.push_back(snapshot);
    }

    return result;
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

}  // namespace ml