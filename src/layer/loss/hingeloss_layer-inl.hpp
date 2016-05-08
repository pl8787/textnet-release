#ifndef TEXTNET_LAYER_HINGELOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_HINGELOSS_LAYER_INL_HPP_

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
class HingeLossLayer : public Layer<xpu>{
 public:
  HingeLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~HingeLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want

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
                  "HingeLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "HingeLossLayer:top size problem.");
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "HingeLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "HingeLossLayer:top size problem.");
    nbatch = bottom[0]->data.size(0);                 
    top[0]->Resize(nbatch, 1, 1, 1, true);
    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        bottom[1]->PrintShape("bottom1");
        top[0]->PrintShape("top0");
    }
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();
    
    bottom1_data = 2 * (bottom1_data - 0.5);
    top_data = F<op::relu>(1.0f - bottom0_data * bottom1_data);
    
    for (int i = 1; i < nbatch; ++i) {
      top_data[0] += top_data[i];
    }

    top_data[0] /= nbatch;
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> bottom0_diff = bottom[0]->diff_d1();
    
    if (this->prop_error[0]) {
      bottom0_diff += -1.0f/nbatch * F<op::relu_grad>(1.0f - bottom0_data * bottom1_data) * bottom1_data;
    }
  }
  
 protected:
  int nbatch;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_HINGELOSS_LAYER_INL_HPP_

