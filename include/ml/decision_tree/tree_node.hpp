#pragma once

namespace ml {

struct DecisionTreeNode {
    bool is_leaf;
    double threshold;
    int prediction;

    DecisionTreeNode* left;
    DecisionTreeNode* right;
};

}  // namespace ml