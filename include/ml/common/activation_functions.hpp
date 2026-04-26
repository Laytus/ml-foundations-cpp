#pragma once

namespace ml::common {

double sigmoid(double z);
int binary_predict(double probability, double threshold);

} // namespace ml::common