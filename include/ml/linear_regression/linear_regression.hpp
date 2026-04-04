#pragma once

#include <vector>

namespace ml {

class LinearRegression {
public:
    LinearRegression();

    double predict (double x) const;
    std::vector<double> predict(const std::vector<double>& xs) const;

    double weight() const;
    double bias() const;

    void set_parameters(double weight, double bias);

private:
    double weight_;
    double bias_;
};

} // namespace ml