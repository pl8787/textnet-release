#ifndef TEXTNET_LAYER_LENGTH_TRANS_LAYER_INL_HPP_
#define TEXTNET_LAYER_LENGTH_TRANS_LAYER_INL_HPP_

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
class LengthTransLayer : public Layer<xpu>{
 public:
  LengthTransLayer(LayerType type) { this->layer_type = type; }
  virtual ~LengthTransLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["source_axis"] = SettingV(0);
    this->defaults["target_axis"] = SettingV(0);
	
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["trans_type"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    trans_type = setting["trans_type"].sVal();
    target_axis = setting["target_axis"].iVal();
    source_axis = setting["source_axis"].iVal();

    utils::Check(trans_type == "2D->1D" || trans_type == "1D->2D",  "LengthTransLayer: trans_type, 2D->1D or 1D->2D.");
    utils::Check(target_axis == 0 || target_axis == 1, "LengthTransLayer: target_axis, 0 or 1.");
    utils::Check(source_axis == 0 || source_axis == 1, "LengthTransLayer: target_axis, 0 or 1.");

    utils::Check(bottom.size() == BottomNodeNum(),
                  "LengthTransLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LengthTransLayer:top size problem.");

  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "LengthTransLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LengthTransLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    
    if (trans_type == "2D->1D") { 
      utils::Check(bottom[0]->length.size(1) >= 2, "LengthTransLayer: need 2d length.");
      top[0]->Resize(nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), bottom[0]->data.size(3), nbatch, 1, true);
    } else if (trans_type == "1D->2D") {
      utils::Check(bottom[0]->length.size(1) == 1, "LengthTransLayer: need 1d length.");
      top[0]->Resize(nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), bottom[0]->data.size(3), nbatch, 2, true);
    }

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
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
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;

    // Directly copy data
    top_data = F<op::identity>(bottom_data);

    if (trans_type == "2D->1D") {
      for (int i = 0; i < nbatch; ++i) {
        top_len[i][0] = bottom_len[i][source_axis];
      }
    } else if (trans_type == "1D->2D") {
      for (int i = 0; i < nbatch; ++i) {
        top_len[i][target_axis] = bottom_len[i][0];
        top_len[i][1-target_axis] = bottom[0]->data.size(2+source_axis);
      }
    }
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
  int target_axis;
  int source_axis;
  string trans_type;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LENGTH_TRANS_LAYER_INL_HPP_

