#ifndef TEXTNET_LAYER_PAIT_WEIGHTED_HINGE_LOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_PAIT_WEIGHTED_HINGE_LOSS_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class PairWeightedHingeLossLayer : public Layer<xpu>{
 public:
  PairWeightedHingeLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~PairWeightedHingeLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 3; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["delta"] = SettingV(1.0f);
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(),
                  "PairWeightedHingeLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PairWeightedHingeLossLayer:top size problem.");
    delta = setting["delta"].fVal();
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "PairWeightedHingeLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PairWeightedHingeLossLayer:top size problem.");
    nbatch = bottom[0]->data.size(0);    
    utils::Check(nbatch % 2 == 0, "PairWeightedHingeLossLayer:nBatch must be even. Batch size: %d.", nbatch);              
    top[0]->Resize(1, 1, 1, 1, true);
    if (show_info) {
        top[0]->PrintShape("top0");
    }
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    nbatch = bottom[0]->data.size(0);
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> bottom2_data = bottom[2]->data_d1();
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();
    
    for (int i = 0; i < nbatch; i += 2) {
      utils::Check(bottom1_data[i] > bottom1_data[i+1], 
                    "Instances come like x y ... x > y");
      top_data[0] += std::max(0.0f, delta + bottom0_data[i+1] - bottom0_data[i]) * bottom2_data[i];
    }
    
    top_data[0] /= (nbatch/2);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom0_diff = bottom[0]->diff_d1();
    mshadow::Tensor<xpu, 1> bottom2_data = bottom[2]->data_d1();
    
    if (this->prop_error[0]) {
      for (int i = 0; i < nbatch; i+=2) {
        float gate = (delta + bottom0_data[i+1] - bottom0_data[i]) > 0 ? 1 : 0;
        bottom0_diff[i] += -gate / nbatch * bottom2_data[i];
        bottom0_diff[i+1] += gate / nbatch * bottom2_data[i+1];
      }
    }

  }
  
 protected:
  int nbatch;
  float delta;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_PAIT_WEIGHTED_HINGE_LOSS_LAYER_INL_HPP_

