#pragma once

#include <string>

#include "ml/common/dataset.hpp"

namespace ml::common {

Dataset load_csv(const std::string& filepath, bool has_header = true);

} // namespace ml::common