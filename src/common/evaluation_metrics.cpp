#include "ml/common/evaluation_metrics.hpp"

#include <stdexcept>
#include <vector>

namespace ml::common {

double accuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("accuracy: vectors must have the same size");
    }

    if (y_true.empty()) {
        throw std::invalid_argument("accuracy: vectors must not be empty");
    }

    int correct_count = 0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] != 0 && y_true[i] != 1) {
            throw std::invalid_argument("accuracy: y_true must be either 0 or 1");
        }
        
        if (y_pred[i] != 0 && y_pred[i] != 1) {
            throw std::invalid_argument("accuracy: y_pred must be either 0 or 1");
        }

        if (y_true[i] == y_pred[i]) {
            correct_count += 1;
        }
    }

    return static_cast<double>(correct_count) / y_true.size();
}

double precision(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("precision: vectors must have the same size");
    }

    if (y_true.empty()) {
        throw std::invalid_argument("precision: vectors must not be empty");
    }

    double TP = 0.0;
    double FP = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] != 0 && y_true[i] != 1) {
            throw std::invalid_argument("precision: y_true must be either 0 or 1");
        }
        
        if (y_pred[i] != 0 && y_pred[i] != 1) {
            throw std::invalid_argument("precision: y_pred must be either 0 or 1");
        }

        if (y_true[i] == 1 && y_pred[i] == 1) {
            TP += 1.0;
        } else if (y_true[i] == 0 && y_pred[i] == 1) {
            FP += 1.0;
        }
    }

    if ((TP + FP) == 0.0) {
        return 0.0;
    }

    return TP / (TP + FP);
}

double recall(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("recall: vectors must have the same size");
    }

    if (y_true.empty()) {
        throw std::invalid_argument("recall: vectors must not be empty");
    }

    double TP = 0.0;
    double FN = 0.0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] != 0 && y_true[i] != 1) {
            throw std::invalid_argument("recall: y_true must be either 0 or 1");
        }
        
        if (y_pred[i] != 0 && y_pred[i] != 1) {
            throw std::invalid_argument("recall: y_pred must be either 0 or 1");
        }

        if (y_true[i] == 1 && y_pred[i] == 1) {
            TP += 1.0;
        } else if (y_true[i] == 1 && y_pred[i] == 0) {
            FN += 1.0;
        }
    }

    if ((TP + FN) == 0.0) {
        return 0.0;
    }

    return TP / (TP + FN);
}

ConfusionMatrix confusion_matrix(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("confusion_matrix: vectors must have the same size");
    }

    if (y_true.empty()) {
        throw std::invalid_argument("confusion_matrix: vectors must not be empty");
    }

    ConfusionMatrix matrix;
    matrix.true_negatives = 0;
    matrix.false_positives = 0;
    matrix.false_negatives = 0;
    matrix.true_positives = 0;

    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] != 0 && y_true[i] != 1) {
            throw std::invalid_argument("confusion_matrix: y_true must be either 0 or 1");
        }
        
        if (y_pred[i] != 0 && y_pred[i] != 1) {
            throw std::invalid_argument("confusion_matrix: y_pred must be either 0 or 1");
        }

        if (y_true[i] == 0 && y_pred[i] == 0) {
            matrix.true_negatives += 1;
        } else if (y_true[i] == 0 && y_pred[i] == 1) {
            matrix.false_positives += 1;
        } else if (y_true[i] == 1 && y_pred[i] == 0) {
            matrix.false_negatives += 1;
        } else {
            matrix.true_positives+= 1;
        }
    }

    return matrix;
}

}  // namespace ml::common