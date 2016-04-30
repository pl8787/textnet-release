#ifndef TEXTNET_LAYER_AXIS_SPLIT_LAYER_INL_HPP_
#define TEXTNET_LAYER_AXIS_SPLIT_LAYER_INL_HPP_

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
class AxisSplitLayer : public Layer<xpu>{
 public:
  AxisSplitLayer(LayerType type) { this->layer_type = type; }
  virtual ~AxisSplitLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["axis"] = SettingV();
    this->defaults["split_length"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    axis = setting["axis"].iVal();
    split_length = setting["split_length"].iVal();

    utils::Check(axis < 4, 
                  "AxisSplitLayer: axis greater than 4.");
    utils::Check(split_length < bottom[0]->data.size(axis), 
                  "AxisSplitLayer: split_length must less than data size.");

    utils::Check(bottom.size() == BottomNodeNum(),
                  "AxisSplitLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "AxisSplitLayer:top size problem.");

  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "AxisSplitLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "AxisSplitLayer:top size problem.");
                  
    shape_in = bottom[0]->data.shape_;
    shape_out0 = bottom[0]->data.shape_;
    shape_out1 = bottom[0]->data.shape_;
    nbatch = shape_in[0]; 
                  
    shape_out0[axis] = split_length;
    shape_out1[axis] = shape_in[axis] - split_length;

    top[0]->Resize(shape_out0, bottom[0]->length.shape_, true);
    top[1]->Resize(shape_out1, bottom[0]->length.shape_, true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
        top[1]->PrintShape("top1");
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
    mshadow::Tensor<xpu, 4> top0_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top0_len = top[0]->length;
    mshadow::Tensor<xpu, 4> top1_data = top[1]->data;
    mshadow::Tensor<xpu, 2> top1_len = top[1]->length;

    vector<int> p(4);
    vector<int> s(4);
    
    for (p[0] = 0; p[0] < shape_in[0]; ++p[0]) {
      top0_len[p[0]] = shape_out0[3];
      top1_len[p[1]] = shape_out1[3];
      for (p[1] = 0; p[1] < shape_in[0]; ++p[1]) {
        for (p[2] = 0; p[2] < shape_in[0]; ++p[2]) {
          for (p[3] = 0; p[3] < shape_in[0]; ++p[3]) {
            if (p[axis] < split_length) {
              top0_data[p[0]][p[1]][p[2]][p[3]] = bottom_data[p[0]][p[1]][p[2]][p[3]];
            } else {
              s[0] = p[0], s[1] = p[1], s[2] = p[2], s[3] = p[3];
              s[axis] -= split_length;
              top1_data[s[0]][s[1]][s[2]][s[3]] = bottom_data[p[0]][p[1]][p[2]][p[3]];
            }
          }
        }
      }
    }

  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top0_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> top1_diff = top[1]->diff;

  }
  
 protected:
  int nbatch;
  int axis;
  int split_length;
  mshadow::Shape<4> shape_in;
  mshadow::Shape<4> shape_out0;
  mshadow::Shape<4> shape_out1;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_AXIS_SPLIT_LAYER_INL_HPP_

