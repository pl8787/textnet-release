#ifndef TEXTNET_LAYER_SWAP_AXIS_LAYER_INL_HPP_
#define TEXTNET_LAYER_SWAP_AXIS_LAYER_INL_HPP_

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
class SwapAxisLayer : public Layer<xpu>{
 public:
  SwapAxisLayer(LayerType type) { this->layer_type = type; }
  virtual ~SwapAxisLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["axis1"] = SettingV();
    this->defaults["axis2"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "SwapAxisLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "SwapAxisLayer:top size problem.");    
    
    axis1 = setting["axis1"].iVal();
    axis2 = setting["axis2"].iVal();
    if (axis1 > axis2) {
        int temp = axis1;
        axis1 = axis2;
        axis2 = temp;
    }
    utils::Check(axis1 != axis2, "SwapAxisLayer: axis1 == axis2.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "SwapAxisLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "SwapAxisLayer:top size problem.");
                  
    bottom_shape = bottom[0]->data.shape_;
    top_shape = bottom_shape;
    top_shape[axis1] = bottom_shape[axis2];
    top_shape[axis2] = bottom_shape[axis1];

    top[0]->Resize(top_shape, true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (! (bottom[0]->data.shape_ == bottom_shape)) {
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
    top_len = F<op::identity>(bottom_len);

    switch (axis1) {
        case 0:
            switch (axis2) {
                case 1:
                    top_data = swapaxis<1, 0>(bottom_data);
                    break;
                case 2:
                    top_data = swapaxis<2, 0>(bottom_data);
                    break;
                case 3:
                    top_data = swapaxis<3, 0>(bottom_data);
                    break;
            }
            break;
        case 1:
            switch (axis2) {
                case 2:
                    top_data = swapaxis<2, 1>(bottom_data);
                    break;
                case 3:
                    top_data = swapaxis<3, 1>(bottom_data);
                    break;
            }
            break;
        case 2:
            top_data = swapaxis<3, 2>(bottom_data);
            break;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    if (this->prop_error[0]) {
        switch (axis1) {
            case 0:
                switch (axis2) {
                    case 1:
                        bottom_diff = swapaxis<1, 0>(top_diff);
                        break;
                    case 2:
                        bottom_diff = swapaxis<2, 0>(top_diff);
                        break;
                    case 3:
                        bottom_diff = swapaxis<3, 0>(top_diff);
                        break;
                }
                break;
            case 1:
                switch (axis2) {
                    case 2:
                        bottom_diff = swapaxis<2, 1>(top_diff);
                        break;
                    case 3:
                        bottom_diff = swapaxis<3, 1>(top_diff);
                        break;
                }
                break;
            case 2:
                bottom_diff = swapaxis<3, 2>(top_diff);
                break;
        }
    }
  }
  
 protected:
  int axis1;
  int axis2;
  mshadow::Shape<4> top_shape;
  mshadow::Shape<4> bottom_shape;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_SWAP_AXIS_LAYER_INL_HPP_

