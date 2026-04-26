#include "ml/linear_regression/linear_regression.hpp"
#include <cmath>

namespace ml {

namespace {

constexpr double kDivergenceThreshold = 1e6;

StabilityAnalysisResult analyze_history(const std::vector<TrainingSnapshot>& history, const char* method_name) {
    if (history.empty()) {
        throw std::invalid_argument(std::string(method_name) + ": history must not be empty");
    }

    for (std::size_t i = 0; i < history.size(); ++i) {
        const TrainingSnapshot& snapshot = history[i];

        if (snapshot.epoch == 0) {
            throw std::invalid_argument(std::string(method_name) + ": epoch values must be greater than zero");
        }

        if (i > 0 && snapshot.epoch <= history[i - 1].epoch) {
            throw std::invalid_argument(std::string(method_name) + ": history epochs must be strictly increasing");
        }

        if (!std::isfinite(snapshot.mse) ||
            !std::isfinite(snapshot.weight) || 
            !std::isfinite(snapshot.bias)) {
            throw std::invalid_argument(std::string(method_name) + ": history contains non-finite values");
        }
    }

    StabilityAnalysisResult result;
    result.initial_mse = history.front().mse;
    result.final_mse = history.back().mse;
    result.max_abs_weight = 0.0;
    result.max_abs_bias = 0.0;

    bool divergent = false;

    for (const auto& snapshot : history) {
        const double abs_weight = std::abs(snapshot.weight);
        const double abs_bias = std::abs(snapshot.bias);

        if (abs_weight > result.max_abs_weight) {
            result.max_abs_weight = abs_weight;
        }

        if (abs_bias > result.max_abs_bias) {
            result.max_abs_bias = abs_bias;
        }

        if (snapshot.mse > kDivergenceThreshold ||
            abs_weight > kDivergenceThreshold ||
            abs_bias > kDivergenceThreshold) {
            divergent = true;
        }
    }

    if (divergent) {
        result.status = StabilityStatus::Divergent;
    } else if (result.final_mse > result.initial_mse) {
        result.status = StabilityStatus::Unstable;
    } else {
        result.status = StabilityStatus::Stable;
    }

    return result;
}

ConvergenceAnalysisResult analyze_convergence_history(
    const std::vector<TrainingSnapshot>& history,
    const char* method_name) {
    if (history.empty()) {
        throw std::invalid_argument(std::string(method_name) + ": history must not be empty");
    }

    for (std::size_t i = 0; i < history.size(); ++i) {
        const TrainingSnapshot& snapshot = history[i];

        if (snapshot.epoch == 0) {
            throw std::invalid_argument(std::string(method_name) + ": epoch values must be greater than zero");
        }

        if (i > 0 && snapshot.epoch <= history[i - 1].epoch) {
            throw std::invalid_argument(std::string(method_name) + ": history epochs must be strictly increasing");
        }

        if (!std::isfinite(snapshot.mse) ||
            !std::isfinite(snapshot.weight) || 
            !std::isfinite(snapshot.bias)) {
            throw std::invalid_argument(std::string(method_name) + ": history contains non-finite values");
        }
    }

    ConvergenceAnalysisResult result;
    result.initial_mse = history.front().mse;
    result.final_mse = history.back().mse;
    result.final_weight = history.back().weight;
    result.final_bias = history.back().bias;

    if (result.initial_mse <= 0.0) {
        throw std::invalid_argument(std::string(method_name) + ": initial_mse must be greater than zero");
    }

    result.relative_improvement = (result.initial_mse - result.final_mse) / result.initial_mse;

    if (result.final_mse >= result.initial_mse) {
        result.status = ConvergenceStatus::Failed;
    } else if (result.relative_improvement < 0.1) {
        result.status = ConvergenceStatus::Stalled;
    } else if (result.relative_improvement < 0.8) {
        result.status = ConvergenceStatus::Slow;
    } else {
        result.status = ConvergenceStatus::Strong;
    }

    return result;
}
}  // namespace

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

    double gradient_w = 2.0 * sum_w / y.size();
    double gradient_b = 2.0 * sum_b / y.size();

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

BatchGDExperimentResult LinearRegression::run_batch_gd_experiment(const std::vector<double>& x, const std::vector<double>& y, double learning_rate, std::size_t epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("run_batch_gd_experiment: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("run_batch_gd_experiment: vectors must not be empty");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("run_batch_gd_experiment: learning_rate must be greater than zero");
    }

    if (epochs == 0) {
        throw std::invalid_argument("run_batch_gd_experiment: epochs must be greater than zero");
    }

    BatchGDExperimentResult result;
    result.learning_rate = learning_rate;
    result.epochs = epochs;
    result.history = fit_batch_with_history(x, y, learning_rate, epochs);

    return result;
}

std::pair<double, double> LinearRegression::stochastic_gradient(double x, double y) const {
    double error = predict(x) - y;

    double gradient_w = 2.0 * error * x;
    double gradient_b = 2.0 * error;

    return {gradient_w, gradient_b};
}

void LinearRegression::stochastic_gradient_descent_step(double x, double y, double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("stochastic_gradient_descent_step: learning_rate must be greater than zero");
    }

    auto [gradient_w, gradient_b] = stochastic_gradient(x, y);
    weight_ -= learning_rate * gradient_w;
    bias_ -= learning_rate * gradient_b;
}

void LinearRegression::fit_stochastic(const std::vector<double>& x, const std::vector<double>& y, double learning_rate, std::size_t epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("fit_stochastic: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("fit_stochastic: vectors must not be empty");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("fit_stochastic: learning_rate must be greater than zero");
    }

    if (epochs == 0) {
        throw std::invalid_argument("fit_stochastic: epochs must be greater than zero");
    }

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t sample_index = 0; sample_index < x.size(); ++sample_index) {
            stochastic_gradient_descent_step(x[sample_index], y[sample_index], learning_rate);
        }
    }
}

std::vector<TrainingSnapshot> LinearRegression::fit_stochastic_with_history(const std::vector<double>& x, const std::vector<double>& y, double learning_rate, std::size_t epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("fit_stochastic_with_history: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("fit_stochastic_with_history: vectors must not be empty");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("fit_stochastic_with_history: learning_rate must be greater than zero");
    }

    if (epochs == 0) {
        throw std::invalid_argument("fit_stochastic_with_history: epochs must be greater than zero");
    }

    std::vector<TrainingSnapshot> result;
    result.reserve(epochs);

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t sample = 0; sample < x.size(); ++sample) {
            stochastic_gradient_descent_step(x[sample], y[sample], learning_rate);
        }

        TrainingSnapshot snapshot; 
        snapshot.epoch = epoch + 1;
        snapshot.mse = mse(x, y);
        snapshot.weight = weight();
        snapshot.bias = bias();

        result.push_back(snapshot);
    }

    return result;
}

SGDExperimentResult LinearRegression::run_sgd_experiment(const std::vector<double>& x, const std::vector<double>& y, double learning_rate, std::size_t epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("run_sgd_experiment: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("run_sgd_experiment: vectors must not be empty");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("run_sgd_experiment: learning_rate must be greater than zero");
    }

    if (epochs == 0) {
        throw std::invalid_argument("run_sgd_experiment: epochs must be greater than zero");
    }

    SGDExperimentResult result;
    result.learning_rate = learning_rate;
    result.epochs = epochs;
    result.history = fit_stochastic_with_history(x, y, learning_rate, epochs);

    return result;
}

BatchGDLearningRateStudyResult LinearRegression::run_batch_gd_learning_rate_study(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& learning_rates, std::size_t epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("run_batch_gd_learning_rate_study: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("run_batch_gd_learning_rate_study: vectors must not be empty");
    }
    
    if (epochs == 0) {
        throw std::invalid_argument("run_batch_gd_learning_rate_study: epochs must be greater than zero");
    }
    
    if (learning_rates.empty()) {
        throw std::invalid_argument("run_batch_gd_learning_rate_study: learning_rates must not be empty");
    }

    BatchGDLearningRateStudyResult result;
    result.learning_rates = learning_rates;
    result.epochs = epochs;
    result.experiments.reserve(learning_rates.size());

    for (double learning_rate : learning_rates) {
        if (learning_rate <= 0.0) {
            throw std::invalid_argument("run_batch_gd_learning_rate_study: learning_rate must be greater than zero");
        }

        result.experiments.push_back(
            run_batch_gd_experiment(x, y, learning_rate, epochs)
        );
    }

    return result;
}

SGDLearningRateStudyResult LinearRegression::run_sgd_learning_rate_study(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& learning_rates, std::size_t epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("run_sgd_learning_rate_study: vectors must have the same size");
    }

    if (x.empty()) {
        throw std::invalid_argument("run_sgd_learning_rate_study: vectors must not be empty");
    }
    
    if (epochs == 0) {
        throw std::invalid_argument("run_sgd_learning_rate_study: epochs must be greater than zero");
    }
    
    if (learning_rates.empty()) {
        throw std::invalid_argument("run_sgd_learning_rate_study: learning_rates must not be empty");
    }

    SGDLearningRateStudyResult result;
    result.learning_rates = learning_rates;
    result.epochs = epochs;
    result.experiments.reserve(learning_rates.size());

    for (double learning_rate : learning_rates) {
        if (learning_rate <= 0.0) {
            throw std::invalid_argument("run_sgd_learning_rate_study: learning_rate must be greater than zero");
        }

        result.experiments.push_back(
            run_sgd_experiment(x, y, learning_rate, epochs)
        );
    }

    return result;
}

StabilityAnalysisResult LinearRegression::analyze_batch_gd_stability(const BatchGDExperimentResult& experiment) const {
    return analyze_history(experiment.history, "analyze_batch_gd_stability");
}

StabilityAnalysisResult LinearRegression::analyze_sgd_stability(const SGDExperimentResult& experiment) const {
    return analyze_history(experiment.history, "analyze_sgd_stability");
}

ConvergenceAnalysisResult LinearRegression::analyze_batch_gd_convergence(const BatchGDExperimentResult& experiment) const {
    return analyze_convergence_history(experiment.history, "analyze_batch_gd_convergence");
}

ConvergenceAnalysisResult LinearRegression::analyze_sgd_convergence(const SGDExperimentResult& experiment) const {
    return analyze_convergence_history(experiment.history, "analyze_sgd_convergence");
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