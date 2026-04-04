#include "ml/common/csv_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml::common {

namespace {

std::vector<double> parse_csv_row(const std::string& line) {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            throw std::runtime_error("load_csv: empty value found in row");
        }

        std::size_t pos = 0;
        double value = 0.0;

        try {
            value = std::stod(token, &pos);
        } catch (const std::exception&) {
            throw std::runtime_error("load_csv: failed to parse numeric value");
        }

        if (pos != token.size()) {
            throw std::runtime_error("load_csv: non-numeric value found in row");
        }

        row.push_back(value);
    }

    if (row.empty()) {
        throw std::runtime_error("load_csv: empty row found");
    }

    return row;
}

} // namespace

Dataset load_csv(const std::string& filepath, bool has_header) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("load_csv: could not open file: " + filepath);
    }

    Dataset dataset;
    std::string line;
    bool first_line = true;
    std::size_t expected_columns = 0;
    bool found_data_row = false;

    while (std::getline(file, line)) {
        if (first_line && has_header) {
            first_line = false;
            continue;
        }

        first_line = false;

        if (line.empty()) {
            throw std::runtime_error("load_csv: empty line found");
        }

        std::vector<double> row = parse_csv_row(line);

        if (row.size() < 2) {
            throw std::runtime_error("load_csv: each row must contain at least one feature and one target");
        }

        if (expected_columns == 0) {
            expected_columns = row.size();
        } else if (row.size() != expected_columns) {
            throw std::runtime_error("load_csv: inconsistent number of columns across rows");
        }

        std::vector<double> feature_row(row.begin(), row.end() - 1);
        double target = row.back();

        dataset.features.push_back(feature_row);
        dataset.targets.push_back(target);
        found_data_row = true;
    }

    if (!found_data_row) {
        throw std::runtime_error("load_csv: no data rows found");
    }

    return dataset;
}

} // namespace ml::common