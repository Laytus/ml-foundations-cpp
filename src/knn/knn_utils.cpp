#include <cstddef>

namespace ml {

bool is_valid_k(int k, std::size_t n_samples) {
    if (n_samples == 0) {
        return false;
    }

    if (k < 1) {
        return false;
    }

    if (k > static_cast<int>(n_samples)) {
        return false;
    }

    return true;
}

}  // namespace ml