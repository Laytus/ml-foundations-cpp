#pragma once

#include <vector>

namespace ml {

class LogisticRegression {
public:
    LogisticRegression();

    double predict_proba(double x) const;
    std::vector<double> predict_proba(std::vector<double>& x_values) const;
    
    int predict(double x) const;
    std::vector<int> predict(std::vector<double>& x_values) const;

    std::pair<double, double> gradient(
        const std::vector<double>& x_values,
        const std::vector<double>& y_values) const;

    void gradient_descent_step(
        const std::vector<double>& x_values,
        const std::vector<double>& y_values,
        double learning_rate);

    void fit_batch(
        const std::vector<double>& x_values,
        const std::vector<double>& y_values,
        double learning_rate,
        std::size_t epochs);

    double weight() const;
    double bias() const;

    void set_parameters(double weight, double bias);

private:
    double weight_;
    double bias_;
};

}  // namespace ml