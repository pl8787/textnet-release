#ifndef TEXTNET_LAYER_DROPOUT_LAYER_INL_HPP_
#define TEXTNET_LAYER_DROPOUT_LAYER_INL_HPP_

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
class DropoutLayer : public Layer<xpu>{
 public:
  DropoutLayer(LayerType type) { this->layer_type = type; }
  virtual ~DropoutLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["rate"] = SettingV(0.5f);
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
                  "DropoutLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "DropoutLayer:top size problem.");
                  
    rate = setting["rate"].fVal(); 
    utils::Check(rate >= 0.0 && rate <= 1.0, 
                  "Dropout rate must between 0.0 and 1.0.");    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "DropoutLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "DropoutLayer:top size problem.");
                  
    top[0]->Resize(bottom[0]->data.shape_, true);
    mask.Resize(bottom[0]->data.shape_, true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (! (bottom[0]->data.shape_ == top[0]->data.shape_)) {
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
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    top[0]->length = F<op::identity>(bottom[0]->length);

    const float pkeep = 1.0f - rate;
    if (this->phrase_type == kTrain) {
      mask = F<op::threshold>(this->prnd_->uniform(mask.shape_), pkeep); 
      top_data = bottom_data * mask;
    } else {
      top_data = bottom_data * pkeep;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    if (this->prop_error[0]) {
      bottom_diff += top_diff * mask;
    }
  }
  
 protected:
  float rate;
  mshadow::TensorContainer<xpu, 4> mask;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_DROPOUT_LAYER_INL_HPP_

