#include "ml/common/data_split.hpp"

#include <stdexcept>

namespace ml::common {

std::pair<Dataset, Dataset> train_test_split(const Dataset& dataset, double train_ratio) {
    const std::size_t n = dataset.features.size();

    if (n != dataset.targets.size()) {
        throw std::invalid_argument("train_test_split: features and targets must have the same number of samples");
    }

    if (n == 0) {
        throw std::invalid_argument("train_test_split: dataset must not be empty");
    }

    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw std::invalid_argument("train_test_split: train_ratio must be between 0 and 1");
    }

    const std::size_t split_index = static_cast<std::size_t>(train_ratio * static_cast<double>(n));

    if (split_index == 0 || split_index >= n) {
        throw std::invalid_argument("train_test_split: split results in empty train or test set");
    }

    Dataset train_dataset;
    Dataset test_dataset;

    train_dataset.features.assign(dataset.features.begin(), dataset.features.begin() + split_index);
    train_dataset.targets.assign(dataset.targets.begin(), dataset.targets.begin() + split_index);
    
    test_dataset.features.assign(dataset.features.begin() + split_index, dataset.features.end());
    test_dataset.targets.assign(dataset.targets.begin() + split_index, dataset.targets.end());

    return {train_dataset, test_dataset};
}

} // namespace ml::common