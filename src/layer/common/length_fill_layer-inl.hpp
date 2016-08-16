#ifndef TEXTNET_LAYER_LENGTH_FILL_LAYER_INL_HPP_
#define TEXTNET_LAYER_LENGTH_FILL_LAYER_INL_HPP_

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
class LengthFillLayer : public Layer<xpu>{
 public:
  LengthFillLayer(LayerType type) { this->layer_type = type; }
  virtual ~LengthFillLayer(void) {}
  
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
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "LengthFillLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LengthFillLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    
    top[0]->Resize(nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), bottom[0]->data.size(3), 
                   bottom[1]->data.size(0), bottom[1]->data.size(1), true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        bottom[1]->PrintShape("bottom1");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (nbatch != bottom[0]->data.size(0)) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom0_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;

    // Directly copy data
    top_data = F<op::identity>(bottom0_data);
    top_len = F<op::identity>(bottom1_data);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;

    // Directly copy data
    bottom_diff += F<op::identity>(top_diff);

  }
  
 protected:
  int nbatch;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LENGTH_FILL_LAYER_INL_HPP_

