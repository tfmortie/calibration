/** 
* NNet inference C++ core
* 
* Author: Thomas Mortier
* Date: March 2022
*
*/
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <queue>
#include <math.h>
#include <tuple>
#include <sstream>
#include <string>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>
#include "nnet_cpp.h"

void HNode::addch(int64_t in_features, double dp, std::vector<int64_t> y, int64_t id) {
    // check if leaf or internal node 
    if (this->chn.size() > 0)
    {
        // check if y is a subset of one of the children
        int64_t ind = -1;
        for (int64_t i=0; i<static_cast<int64_t>(this->chn.size()); ++i)
        {
            if (std::includes(this->chn[i]->y.begin(), this->chn[i]->y.end(), y.begin(), y.end()) == 1)
            {
                ind = i;
                break;
            }
        }
        if (ind != -1)
            // subset found, hence, recursively pass to child
            this->chn[ind]->addch(in_features, dp, y, id);
        else
        {
            // no children for which y is a subset, hence, put in children list
            HNode* new_node = new HNode();
            new_node->y = y;
            new_node->chn = {};
            new_node->par = this->par;
            this->chn.push_back(new_node);
            unsigned long tot_len_y_chn {0};
            for (auto c : this->chn)
                tot_len_y_chn += c->y.size();
            // check if the current node has all its children
            if (tot_len_y_chn == this->y.size())
            {
                // get string representation of y
                std::stringstream ystr;
                std::copy(y.begin(), y.end(), std::ostream_iterator<int>(ystr, " "));
                std::string lbl {ystr.str()+std::to_string(id)};
                torch::nn::Sequential clf(
                    torch::nn::Dropout(dp),
                    torch::nn::Linear(in_features, this->chn.size())
                );
                this->estimator = this->par->register_module(lbl, clf);
            }
        }
    }
    else
    { 
        // no children yet, hence, put in children list
        HNode* new_node = new HNode();
        new_node->y = y;
        new_node->chn = {};
        new_node->par = this->par;
        this->chn.push_back(new_node);
        // check if we have a single path
        if (new_node->y.size() == this->y.size())
        {
            std::stringstream ystr;
            std::copy(y.begin(), y.end(), std::ostream_iterator<int>(ystr, " "));
            std::string lbl {ystr.str()+std::to_string(id)};
            // create estimator
            torch::nn::Sequential clf(
                torch::nn::Dropout(dp),
                torch::nn::Linear(in_features, 1)
            );
            this->estimator = this->par->register_module(lbl, clf);
        }
    }
}

torch::Tensor HNode::forward(torch::Tensor input, torch::nn::CrossEntropyLoss criterion, int64_t y_ind) {
    torch::Tensor loss = torch::tensor({0}).to(input.device());
    torch::Tensor y = torch::tensor({y_ind}).to(input.device());
    if (this->chn.size() > 1)
    {
        auto o = this->estimator->forward(input);
        loss = loss + criterion(o, y);
    }

    return loss;
}

NNet::NNet(int64_t in_features, int64_t num_classes, double dp, std::vector<std::vector<int64_t>> hstruct) {
    this->num_classes = num_classes;
    // create root node 
    this->root = new HNode();
    if (hstruct.size() == 0) {
        torch::nn::Sequential clf(
            torch::nn::Dropout(dp),
            torch::nn::Linear(in_features, this->num_classes)
        );
        this->root->estimator = this->register_module("root", clf);
        this->root->y = {};
        this->root->chn = {};
        this->root->par = this;
    } else {
        // construct tree for h-softmax
        this->root->y = hstruct[0];
        this->root->chn = {};
        this->root->par = this;
        for (int64_t i=1; i<static_cast<int64_t>(hstruct.size()); ++i)
            this->root->addch(in_features, dp, hstruct[i], i);   
    }
}

torch::Tensor NNet::forward(torch::Tensor input, std::vector<std::vector<int64_t>> target) {
    torch::Tensor loss = torch::tensor({0}).to(input.device());
    torch::nn::CrossEntropyLoss criterion;
    // run over each sample in batch
    for (int64_t bi=0;bi<input.size(0);++bi)
    {
        // begin at root
        HNode* visit_node = this->root;
        for (int64_t yi=0;yi<static_cast<unsigned int>(target[bi].size());++yi)
        {
            auto o = visit_node->forward(input[bi].view({1,-1}), criterion, target[bi][yi]);
            loss = loss + o;
            visit_node = visit_node->chn[target[bi][yi]];
        }
    }
    loss = loss/input.size(0);
        
    return loss;
}

torch::Tensor NNet::forward(torch::Tensor input, torch::Tensor target) {
    torch::Tensor loss = torch::tensor({0});
    torch::nn::CrossEntropyLoss criterion;
    auto o = this->root->estimator->forward(input);
    loss = criterion(o, target);

    return loss;
}
    
std::vector<int64_t> NNet::predict(torch::Tensor input) {
    if (this->root->y.size() == 0)
    {
        auto o = this->root->estimator->forward(input);
        o = o.argmax(1).to(torch::kInt64);
        o = o.to(torch::kCPU);
        std::vector<int64_t> prediction(o.data_ptr<int64_t>(), o.data_ptr<int64_t>() + o.numel());

        return prediction;
    }
    else
    {
        std::vector<int64_t> prediction;
        // run over each sample in batch
        for (int64_t bi=0;bi<input.size(0);++bi)
        {
            // begin at root
            HNode* visit_node = this->root;
            while (visit_node->y.size() > 1)
            {
                auto o = visit_node->estimator->forward(input[bi].view({1,-1}));
                int64_t max_ch_ind = o.argmax(1).item<int64_t>();
                visit_node = visit_node->chn[max_ch_ind];
            }
            prediction.push_back(visit_node->y[0]);
        }
        return prediction;
    }
}

torch::Tensor NNet::predict_proba(torch::Tensor input) {
    if (this->root->y.size() == 0)
    {
        auto o = this->root->estimator->forward(input);
        o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1));
    
        return o;
    }
    else
    {
        std::vector<double> probs;
        // run over each sample in batch
        for (int64_t bi=0;bi<input.size(0);++bi)
        {
            // run over all classes
            for (int64_t yi=0;yi<this->num_classes;++yi)
            {
                // begin at root
                HNode* visit_node = this->root;
                double prob {1.0};
                while (visit_node->y.size() > 1)
                {
                    int64_t ind = -1;
                    for (int64_t i=0; i<static_cast<int64_t>(visit_node->chn.size()); ++i)
                    {
                        if (std::find(visit_node->chn[i]->y.begin(), visit_node->chn[i]->y.end(), yi) != visit_node->chn[i]->y.end()) 
                        {
                            ind = i;
                            break;
                        }
                    }
                    auto o = visit_node->estimator->forward(input[bi].view({1,-1}));
                    o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1));
                    prob = prob*o[0][ind].item<double>();
                    visit_node = visit_node->chn[ind];
                }
                probs.push_back(prob);
            }
        }

        return torch::from_blob(probs.data(),{input.size(0)*this->num_classes},torch::TensorOptions(torch::kFloat64)).view({input.size(0),this->num_classes});
    }
}
 
/* cpp->py bindings */ 
PYBIND11_MODULE(nnet_cpp, m) {
    using namespace pybind11::literals;
    torch::python::bind_module<NNet>(m, "NNet")
        .def(py::init<int64_t, int64_t, double, std::vector<std::vector<int64_t>>>(), "in_features"_a, "num_classes"_a, "dp"_a, "hstruct"_a=py::list())
        .def("forward", py::overload_cast<torch::Tensor, torch::Tensor>(&NNet::forward))
        .def("forward", py::overload_cast<torch::Tensor, std::vector<std::vector<int64_t>>>(&NNet::forward))
        .def("predict", &NNet::predict, "input"_a)
        .def("predict_proba", &NNet::predict_proba, "input"_a);
}
