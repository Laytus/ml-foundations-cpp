#pragma once

#include <vector>

namespace ml::common {

std::vector<double> add(const std::vector<double>& a,
                        const std::vector<double>& b);

std::vector<double> subtract(const std::vector<double>& a,
                              const std::vector<double>& b);

std::vector<double> scalar_multiply(const std::vector<double>& v,
                                    double scalar);

double dot(const std::vector<double>& a,
           const std::vector<double>& b);

double mean(const std::vector<double>& values);

} // namespace ml::common