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
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "DropoutLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "DropoutLayer:top size problem.");
                  
    rate = setting["rate"].f_val; 
    utils::Check(rate >= 0.0 && rate <= 1.0, 
                  "Dropout rate must between 0.0 and 1.0.");    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "DropoutLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "DropoutLayer:top size problem.");
                  
    top[0]->Resize(bottom[0]->data.shape_, true);
	mask.Resize(bottom[0]->data.shape_);

	bottom[0]->PrintShape("bottom0");
	top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    const float pkeep = 1.0f - rate;
    if (this->phrase_type == kTrain) {
      mask = F<op::threshold>(this->prnd_->uniform(mask.shape_), pkeep)  
                * (1.0f/pkeep);
      top_data = bottom_data * mask;
    }
    
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    if (this->prop_error[0]) {
      bottom_diff = top_diff * mask;
    }
  }
  
 protected:
  float rate;
  mshadow::TensorContainer<xpu, 4> mask;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_DROPOUT_LAYER_INL_HPP_

