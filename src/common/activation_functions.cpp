#include "ml/common/activation_functions.hpp"

#include <stdexcept>
#include <cmath>

namespace ml::common {

double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

int binary_predict(double probability, double threshold) {
    if (threshold <= 0.0 || threshold >= 1.0) {
        throw std::invalid_argument("binary_predict: threshold must be strictly between 0.0 and 1.0 ");
    }

    if (probability >= threshold) {
        return 1;
    } else {
        return 0;
    }
}
}  // namespace ml::common