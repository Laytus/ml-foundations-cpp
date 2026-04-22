#pragma once

#include <vector>

namespace ml {

struct TrainingSnapshot {
    std::size_t epoch;
    double mse;
    double weight;
    double bias;
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
    
    std::vector<TrainingSnapshot> fit_batch_with_history(const std::vector<double>& x,
                                                         const std::vector<double>& y,
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