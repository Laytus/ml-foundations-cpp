#pragma once

#include <vector>
#include <cstddef>

namespace ml {

class KNNClassifier {
public:
    KNNClassifier(const std::vector<double>& x_train,
                  const std::vector<int>& y_train,
                  int k);

    int predict(double x) const;
    std::vector<int> predict(const std::vector<double>& x_values) const;

    int k() const;
    const std::vector<double>& x_train() const;
    const std::vector<int>& y_train() const;

private:
    std::vector<double> x_train_;
    std::vector<int> y_train_;
    int k_;
};

}  // namespace ml