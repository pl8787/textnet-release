#ifndef TEXTNET_LAYER_ACTIVATION_LAYER_INL_HPP_
#define TEXTNET_LAYER_ACTIVATION_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu,typename ForwardOp, typename BackOp>
class ActivationLayer : public Layer<xpu>{
 public:
  ActivationLayer(LayerType type) { this->layer_type = type; }
  virtual ~ActivationLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
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
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ActivationLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ActivationLayer:top size problem.");
    top[0]->Resize(bottom[0]->data.shape_);

	  bottom[0]->PrintShape("bottom0");
	  top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    top[0]->length = F<op::identity>(bottom[0]->length);
    top[0]->data = F<ForwardOp>(bottom[0]->data);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    if (this->prop_error[0]) {
      bottom[0]->diff += F<BackOp>(top[0]->data) * top[0]->diff;
    }
  }
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_ACTIVATION_LAYER_INL_HPP_

