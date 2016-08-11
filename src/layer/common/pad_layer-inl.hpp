#ifndef TEXTNET_LAYER_PAD_LAYER_INL_HPP_
#define TEXTNET_LAYER_PAD_LAYER_INL_HPP_

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
class PadLayer : public Layer<xpu>{
 public:
  PadLayer(LayerType type) { this->layer_type = type; }
  virtual ~PadLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["pad_value"] = SettingV(0.0f);

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["dim"] = SettingV();
    this->defaults["pad_x1"] = SettingV();
    this->defaults["pad_x2"] = SettingV();
    this->defaults["pad_y1"] = SettingV(0);
    this->defaults["pad_y2"] = SettingV(0);
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "PadLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PadLayer:top size problem.");    
    
    dim = setting["dim"].iVal();
    pad_x1 = setting["pad_x1"].iVal();
    pad_x2 = setting["pad_x2"].iVal();
    pad_y1 = setting["pad_y1"].iVal();
    pad_y2 = setting["pad_y2"].iVal();
    pad_value = setting["pad_value"].fVal();

    utils::Check(pad_x1 >= 0 && pad_x2 >= 0 && pad_y1 >= 0 && pad_y2 >= 0, "PadLayer: positive needed.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "PadLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PadLayer:top size problem.");
  
    bottom_shape = bottom[0]->data.shape_;

    top_shape[0] = bottom_shape[0];
    top_shape[1] = bottom_shape[1];
    top_shape[2] = bottom_shape[2] + pad_y1 + pad_y2;
    top_shape[3] = bottom_shape[3] + pad_x1 + pad_x2;

    top[0]->Resize(top_shape, bottom[0]->length.shape_, true);

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
    
    for (int i = 0; i < top_data.size(0); ++i) {
      if(dim == 1){
        top_len[i][0] = bottom_len[i][0] + pad_x1 + pad_x2;
      }else if(dim == 2){
        top_len[i][0] = bottom_len[i][0] + pad_y1 + pad_y2;
        top_len[i][1] = bottom_len[i][1] + pad_x1 + pad_x2;
      }
      for (int c = 0; c < top_data.size(1); ++c) {
        if(dim == 1){
          for( int y = 0 ; y < top_data.size(2); ++ y){
            for( int x = 0 ; x < top_len[i][0]; ++ x){
              if((x >= pad_x1 && x < bottom_len[i][0] + pad_x1)){
                top_data[i][c][y][x] = bottom_data[i][c][y][x - pad_x1];
              }else{
                top_data[i][c][y][x] = pad_value;
              }
            }
          }
        }else if (dim == 2){
          for (int y = 0; y < top_len[i][0]; ++y) {
            for (int x = 0; x < top_len[i][1]; ++x) {
              if ((x >= pad_x1 && x < bottom_len[i][1] + pad_x1) && (y >= pad_y1 && y < bottom_len[i][0] + pad_y1)) {
                top_data[i][c][y][x] = bottom_data[i][c][y - pad_y1][x - pad_x1];
              } else {
                top_data[i][c][y][x] = pad_value;
              }
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
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;
    if (!this->prop_error[0]) {
        return;
    }
    for (int i = 0; i < top_diff.size(0); ++i) {
      for (int c = 0; c < top_diff.size(1); ++c) {
        if(dim == 1){
          for(int y = 0 ; y < top_diff.size(2); ++ y){
            for(int x = 0 ; x < top_len[i][0]; ++ x){
              if (x >= pad_x1 && x < bottom_len[i][1] + pad_x1) {
                bottom_diff[i][c][y][x - pad_x1] += top_diff[i][c][y][x];
              }
            }
          }
        }else if(dim == 2){
          for (int y = 0; y < top_len[i][0]; ++y) {
            for (int x = 0; x < top_len[i][1]; ++x) {
              if ((x >= pad_x1 && x < bottom_len[i][1] + pad_x1) && (y >= pad_y1 && y < bottom_len[i][0] + pad_y1)) {
                bottom_diff[i][c][y - pad_y1][x - pad_x1] += top_diff[i][c][y][x];
              }
            }
          }
        }
      }
    }
  }
  
 protected:
  int pad_x1;
  int pad_x2;
  int pad_y1;
  int pad_y2;
  int pad_value;
  int dim;
  mshadow::Shape<4> top_shape;
  mshadow::Shape<4> bottom_shape;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_PAD_LAYER_INL_HPP_

