#include "ml/decision_tree/impurity_utils.hpp"

#include <stdexcept>
#include <vector>

namespace ml {

double gini_impurity(const std::vector<int>& y_values) {
    if (y_values.empty()) {
        throw std::invalid_argument("gini_impurity: y_values must not be empty");
    }

    int count_0 = 0;
    int count_1 = 0;

    for (int y : y_values) {
        if (y != 0 && y != 1) {
            throw std::invalid_argument("gini_impurity: y_values must be either 0 or 1");
        }

        if (y == 0) {
            count_0 += 1;
        } else {
            count_1 += 1;
        }
    }
    
    double p0 = static_cast<double>(count_0) / y_values.size();
    double p1 = static_cast<double>(count_1) / y_values.size();

    return 1.0 - ((p0 * p0) + (p1 * p1));
}

}  // namespace ml