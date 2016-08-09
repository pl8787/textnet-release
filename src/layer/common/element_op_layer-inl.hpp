#ifndef TEXTNET_LAYER_ELEMENT_OP_LAYER_INL_HPP_
#define TEXTNET_LAYER_ELEMENT_OP_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

// this layer elem-wise products bottom representations on the last 2 dimensions
template<typename xpu>
class ElementOpLayer : public Layer<xpu> {
 public:
  ElementOpLayer(LayerType type) { this->layer_type = type; }
  virtual ~ElementOpLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["mu1"] = SettingV(0.0f);   // for activation mu1*bottom0_data
    this->defaults["mu2"] = SettingV(0.0f);   // for activation mu2*bootom1_data
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["op"] = SettingV();  
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);

	mu1 = setting["mu1"].fVal();
	mu2 = setting["mu2"].fVal();
    op = setting["op"].sVal();
    
    utils::Check(bottom.size() == BottomNodeNum(), "ElementOpLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ElementOpLayer:top size problem.");

    utils::Check(op == "product" || op == "sum", "ElementOpLayer: only support product and sum");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "ElementOpLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ElementOpLayer:top size problem.");
    
    mshadow::Shape<4> shape0 = bottom[0]->data.shape_;
    mshadow::Shape<4> shape1 = bottom[1]->data.shape_;
	mshadow::Shape<2> shape_len = bottom[0]->length.shape_;

    utils::Check(shape0[0] == shape1[0], "ElementOpLayer: bottom sizes does not match.");
    utils::Check(shape0[1] == shape1[1], "ElementOpLayer: bottom sizes does not match.");
    utils::Check(shape0[2] == shape1[2], "ElementOpLayer: bottom sizes does not match.");
    utils::Check(shape0[3] == shape1[3], "ElementOpLayer: bottom sizes does not match.");

    top[0]->Resize(shape0[0], shape0[1], shape0[2], shape0[3], shape_len[0], shape_len[1], true);

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
    if (! (bottom[0]->data.size(0) == top[0]->data.size(0))) {
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
    top[0]->length = F<op::identity>(bottom[0]->length); // bottom nodes should have the same length

    mshadow::Tensor<xpu, 4> bottom0_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom1_data = bottom[1]->data;
    mshadow::Tensor<xpu, 4> top_data     = top[0]->data;

    if (op == "product") {
      top_data = bottom0_data * bottom1_data;
    } else if (op == "sum") {
      top_data = bottom0_data + bottom1_data;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff     = top[0]->diff;
    mshadow::Tensor<xpu, 4> bottom0_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom0_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> bottom1_data = bottom[1]->data;
    mshadow::Tensor<xpu, 4> bottom1_diff = bottom[1]->diff;

    if (op == "product") {
      bottom0_diff += top_diff * bottom1_data;
      bottom1_diff += top_diff * bottom0_data;
    } else if (op == "sum") {
      bottom0_diff += top_diff;
      bottom1_diff += top_diff;
    }
  }

 protected:
  float mu1;
  float mu2;
  string op;

};
}  // namespace layer
}  // namespace textnet
#endif
