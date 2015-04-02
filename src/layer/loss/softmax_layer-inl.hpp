#ifndef TEXTNET_LAYER_SOFTMAX_LAYER_INL_HPP_
#define TEXTNET_LAYER_SOFTMAX_LAYER_INL_HPP_

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
class SoftmaxLayer : public Layer<xpu>{
 public:
  SoftmaxLayer(LayerType type) { this->layer_type = type; }
  virtual ~SoftmaxLayer(void) {}
  
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
                  "SoftmaxLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "SoftmaxLayer:top size problem.");
      
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "SoftmaxLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "SoftmaxLayer:top size problem.");
    nbatch = bottom[0]->data.size(0);  
    top[0]->Resize(1, 1, 1, 1, true);
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();
    
    mshadow::Softmax(bottom0_data, bottom0_data);
	
	top_data[0] = 0.0f;
    
    for (int i = 0; i < nbatch; ++i) {
      int k = static_cast<int>(bottom1_data[i]);
      if (bottom0_data[i][k] == 0.) {
        top_data[0] += 88; // by min float number
      } else { 
        top_data[0] += -log(bottom0_data[i][k]);
      }
    }

    top_data[0] /= nbatch;
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 2> bottom0_diff = bottom[0]->diff_d2();
    
    bottom0_diff = F<op::identity>(bottom0_data);
    
    if (this->prop_error[0]) {
      for (int i = 0; i < nbatch; ++i) {
        int k = static_cast<int>(bottom1_data[i]);
        bottom0_diff[i][k] -= 1.0f; 
      }
    }
  }
  
 protected:
  int nbatch;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_SOFTMAX_LAYER_INL_HPP_


