#pragma once

#include <vector>

namespace ml {

struct TrainingSnapshot {
    std::size_t epoch;
    double mse;
    double weight;
    double bias;
};

struct BatchGDExperimentResult {
    double learning_rate;
    std::size_t epochs;
    std::vector<TrainingSnapshot> history;
};

struct SGDExperimentResult {
    double learning_rate;
    std::size_t epochs;
    std::vector<TrainingSnapshot> history;
};

// enum class OptimizationMethod {
//     BatchGD,
//     SGD
// };

struct BatchGDLearningRateStudyResult {
    std::vector<double> learning_rates;
    std::size_t epochs;
    std::vector<BatchGDExperimentResult> experiments;
};

struct SGDLearningRateStudyResult {
    std::vector<double> learning_rates;
    std::size_t epochs;
    std::vector<SGDExperimentResult> experiments;
};

enum class StabilityStatus {
    Stable,
    Unstable,
    Divergent
};

struct StabilityAnalysisResult {
    StabilityStatus status;
    double initial_mse;
    double final_mse;
    double max_abs_weight;
    double max_abs_bias;
};

enum class ConvergenceStatus {
    Strong,
    Slow,
    Stalled,
    Failed
};

struct ConvergenceAnalysisResult {
    ConvergenceStatus status;
    double initial_mse;
    double final_mse;
    double relative_improvement;
    double final_weight;
    double final_bias;
};

class LinearRegression {
public:
    LinearRegression();

    double predict (double x) const;
    std::vector<double> predict(const std::vector<double>& xs) const;

    double mse(const std::vector<double>& x, 
               const std::vector<double>& y) const;

    std::pair<double, double> gradient(const std::vector<double>& x, 
                                       const std::vector<double>& y) const;

    void gradient_descent_step(const std::vector<double>& x,
                               const std::vector<double>& y,
                               double learning_rate);

    void fit_batch(const std::vector<double>& x,
                   const std::vector<double>& y,
                   double learning_rate,
                   std::size_t epochs);
    
    std::vector<TrainingSnapshot> fit_batch_with_history(
        const std::vector<double>& x,
        const std::vector<double>& y,
        double learning_rate,
        std::size_t epochs);

    BatchGDExperimentResult run_batch_gd_experiment(
        const std::vector<double>& x,
        const std::vector<double>& y,
        double learning_rate,
        std::size_t epochs);

    std::pair<double, double> stochastic_gradient(double x, 
                                                  double y) const;
    
    void stochastic_gradient_descent_step(double x,
                                          double y,
                                          double learning_rate);

    void fit_stochastic(const std::vector<double>& x,
                        const std::vector<double>& y,
                        double learning_rate,
                        std::size_t epochs);
    
    std::vector<TrainingSnapshot> fit_stochastic_with_history(
        const std::vector<double>& x,
        const std::vector<double>& y,
        double learning_rate,
        std::size_t epochs);

    SGDExperimentResult run_sgd_experiment(
        const std::vector<double>& x,
        const std::vector<double>& y,
        double learning_rate,
        std::size_t epochs);

    BatchGDLearningRateStudyResult run_batch_gd_learning_rate_study(
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& learning_rates,
        std::size_t epochs);

    SGDLearningRateStudyResult run_sgd_learning_rate_study(
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& learning_rates,
        std::size_t epochs);

    StabilityAnalysisResult analyze_batch_gd_stability(
        const BatchGDExperimentResult& experiment) const;

    StabilityAnalysisResult analyze_sgd_stability(
        const SGDExperimentResult& experiment) const;

    ConvergenceAnalysisResult analyze_batch_gd_convergence(
        const BatchGDExperimentResult& experiment) const;
    
    ConvergenceAnalysisResult analyze_sgd_convergence(
        const SGDExperimentResult& experiment) const;

    double weight() const;
    double bias() const;

    void set_parameters(double weight, double bias);

private:
    double weight_;
    double bias_;
};

}  // namespace ml