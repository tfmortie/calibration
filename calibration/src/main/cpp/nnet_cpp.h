/** 
* Header NNet inference C++ core
* 
* Author: Thomas Mortier
* Date: March 2022
*
*/
#ifndef NNET_H
#define NNET_H

#include <torch/torch.h>

/* structure which represents the basic component for NNet  */
struct HNode : torch::nn::Module {
    // attributes
    torch::nn::Sequential estimator {nullptr};
    std::vector<int64_t> y;
    std::vector<HNode*> chn;
    torch::nn::Module *par;
    // functions
    void addch(int64_t in_features, double dp, std::vector<int64_t> y, int64_t id); 
    torch::Tensor forward(torch::Tensor input, torch::nn::CrossEntropyLoss criterion, int64_t y_ind={});
};

/* PQ struct used for inference */
struct QNode
{
    HNode* node;
    double prob;
    /* comparator */
    bool operator<(const QNode& n) const { return prob < n.prob;}
};

/* class which represents an NNet object */
struct NNet : torch::nn::Module {
    // attributes
    int64_t num_classes;
    double dp;
    HNode* root;
    // forward-pass functions
    NNet(int64_t in_features, int64_t num_classes, double dp, std::vector<std::vector<int64_t>> hstruct={});
    torch::Tensor forward(torch::Tensor input, std::vector<std::vector<int64_t>> target={}); /* forward pass for hierarchical model */
    torch::Tensor forward(torch::Tensor input, torch::Tensor target={}); /* forward pass for flat model */
    std::vector<int64_t> predict(torch::Tensor input); /* top-1 prediction */
    torch::Tensor predict_proba(torch::Tensor input); /* top-1 prediction */
};

#endif
