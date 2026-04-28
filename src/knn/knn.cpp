#include "ml/knn/knn.hpp"
#include "ml/knn/knn_utils.hpp"
#include "ml/common/distance_metrics.hpp"

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <utility>

namespace ml {

KNNClassifier::KNNClassifier(const std::vector<double>& x_train, const std::vector<int>& y_train, int k)
    : x_train_(x_train), y_train_(y_train), k_(k) {
    if (x_train_.empty()) {
        throw std::invalid_argument("KNNClassifier: x_train must not be empty");
    }
    
    if (y_train_.empty()) {
        throw std::invalid_argument("KNNClassifier: y_train must not be empty");
    }
    
    if (x_train_.size() != y_train_.size()) {
        throw std::invalid_argument("KNNClassifier: x_train and y_train must have the same size");
    }
    
    for (int label : y_train_) {
        if (label != 0 && label != 1) {
            throw std::invalid_argument("KNNClassifier: y_train must be either 0 or 1");
        }
    }

    if (!ml::is_valid_k(k_, x_train_.size())) {
        throw std::invalid_argument("KNNClassifier: k must satisfy 1 <= k <= number of training samples");
    }
}

int KNNClassifier::predict(double x) const {
    std::vector<std::pair<double, int>> neighbors;
    neighbors.reserve(x_train_.size());

    for (std::size_t i = 0; i < x_train_.size(); ++i) {
        double distance = ml::common::euclidean_distance(x, x_train_[i]);
        neighbors.push_back({distance, y_train_[i]});
    }

    std::sort(neighbors.begin(), neighbors.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
              });

    int count_0 = 0;
    int count_1 = 0;

    for (int i = 0; i < k_; ++i) {
        if (neighbors[i].second == 1) {
            count_1 += 1;
        } else {
            count_0 += 1;
        }
    }

    if (count_1 > count_0) {
        return 1;
    }

    return 0;
}

std::vector<int> KNNClassifier::predict(const std::vector<double>& x_values) const {
    if (x_values.empty()) {
        throw std::invalid_argument("predict: vector must not be empty");
    }

    std::vector<int> result;
    result.reserve(x_values.size());

    for (double x : x_values) {
        result.push_back(predict(x));
    }

    return result;
}

int KNNClassifier::k() const {
    return k_;
}

const std::vector<double>& KNNClassifier::x_train() const  {
    return x_train_;
}

const std::vector<int>& KNNClassifier::y_train() const {
    return y_train_;
}

}