#pragma once

#include <vector>

namespace ml::common {

struct ConfusionMatrix {
    int true_negatives;
    int false_positives;
    int false_negatives;
    int true_positives;
};

double accuracy(const std::vector<int>& y_true,
                const std::vector<int>& y_pred);

double precision(const std::vector<int>& y_true,
                 const std::vector<int>& y_pred);

double recall(const std::vector<int>& y_true,
              const std::vector<int>& y_pred);

ConfusionMatrix confusion_matrix(const std::vector<int>& y_true,
                                 const std::vector<int>& y_pred);

}  // namespace ml::common