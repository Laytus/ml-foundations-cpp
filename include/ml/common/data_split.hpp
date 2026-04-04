#pragma once

#include <utility>

#include "ml/common/dataset.hpp"

namespace ml::common {

std::pair<Dataset, Dataset> train_test_split(const Dataset& dataset,
                                             double train_ratio);

} // namespace ml::common