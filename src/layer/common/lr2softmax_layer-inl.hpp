#ifndef TEXTNET_LAYER_LR2SOFTMAX_LAYER_INL_HPP_
#define TEXTNET_LAYER_LR2SOFTMAX_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

// sum aross one axis
template<typename xpu>
class Lr2softmaxLayer : public Layer<xpu> {
 public:
  Lr2softmaxLayer(LayerType type) { this->layer_type = type; }
  virtual ~Lr2softmaxLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["score_class"] = SettingV(); // the other class will be pad as zero
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "Lr2softmaxLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "Lr2softmaxLayer:top size problem.");
    score_class = setting["score_class"].iVal();
    utils::Check(0 == score_class || 1 == score_class, "Lr2softmaxLayer: score class setting error.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "Lr2softmaxLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "Lr2softmaxLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    utils::Check(1 == shape_in[1] && 1 == shape_in[2] && 1 == shape_in[3], "Lr2softmaxLayer: input size error.");
    mshadow::Shape<4> shape_out= shape_in;
    shape_out[1] = 2;

    top[0]->Resize(shape_out, true);

	bottom[0]->PrintShape("bottom0");
	top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    mshadow::Tensor<xpu, 1> bottom_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 2> top_data    = top[0]->data_d2();
    top_data = 0.f;
    for (int i = 0; i < bottom_data.size(0); ++i) {
      top_data[i][score_class] = bottom_data[i]; 
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    mshadow::Tensor<xpu, 1> bottom_diff = bottom[0]->diff_d1();
    mshadow::Tensor<xpu, 2> top_diff    = top[0]->diff_d2();
    for (int i = 0; i < bottom_diff.size(0); ++i) {
      bottom_diff[i] += top_diff[i][score_class];
    }
  }
 protected:
  int score_class;
};
}  // namespace layer
}  // namespace textnet
#endif
