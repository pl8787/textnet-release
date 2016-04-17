#ifndef TEXTNET_LAYER_FILL_CURVE_XY2D_LAYER_INL_HPP_
#define TEXTNET_LAYER_FILL_CURVE_XY2D_LAYER_INL_HPP_

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
class FillCurveXY2DLayer : public Layer<xpu>{
 public:
  FillCurveXY2DLayer(LayerType type) { this->layer_type = type; }
  virtual ~FillCurveXY2DLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["curve_type"] = SettingV("Hilbert");

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
                  "FillCurveXY2DLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "FillCurveXY2DLayer:top size problem.");    
    
    curve_type = setting["curve_type"].sVal();

    utils::Check(curve_type == "Hilbert" || curve_type == "Zigzag" || curve_type == "Peano", "FillCurveXY2DLayer: CurveXY2D Type Error.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "FillCurveXY2DLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "FillCurveXY2DLayer:top size problem.");
  
    bottom_shape = bottom[0]->data.shape_;
    bottom_len_shape = bottom[0]->length.shape_;

    int max_n = max(bottom_shape[2], bottom_shape[3]);
    top_shape[0] = bottom_shape[0];
    top_shape[1] = 1;
    top_shape[2] = max_n * max_n;
    top_shape[3] = bottom_shape[1];

    top_len_shape[0] = bottom_len_shape[0];
    top_len_shape[1] = 1;

    top[0]->Resize(top_shape, top_len_shape, true);

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
  
  //rotate/flip a quadrant appropriately
  void rot(int n, int *x, int *y, int rx, int ry) {
      if (ry == 0) {
          if (rx == 1) {
              *x = n-1 - *x;
              *y = n-1 - *y;
          }
  
          //Swap x and y
          int t  = *x;
          *x = *y;
          *y = t;
      }
  }
  
  //convert (x,y) to d
  int xy2d (int n, int x, int y) {
      int rx, ry, s, d=0;
      for (s=n/2; s>0; s/=2) {
          rx = (x & s) > 0;
          ry = (y & s) > 0;
          d += s * s * ((3 * rx) ^ ry);
          rot(s, &x, &y, rx, ry);
      }
      return d;
  }
  
  //convert d to (x,y)
  void d2xy(int n, int d, int *x, int *y) {
      int rx, ry, s, t=d;
      *x = *y = 0;
      for (s=1; s<n; s*=2) {
          rx = 1 & (t/2);
          ry = 1 & (t ^ rx);
          rot(s, x, y, rx, ry);
          *x += s * rx;
          *y += s * ry;
          t /= 4;
      }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;    

    for (int i = 0; i < bottom_shape[0]; ++i) {
      int n = max(bottom_len[i][0], bottom_len[i][1]);
      // top_len[i][0] = bottom_len[i][0] * bottom_len[i][1];
      top_len[i][0] = n * n;
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
          int d = xy2d(n, x, y);
          for (int c = 0; c < bottom_shape[1]; ++c) {
            top_data[i][0][d][c] = bottom_data[i][c][y][x];
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
    for (int i = 0; i < bottom_shape[0]; ++i) {
      int n = max(bottom_len[i][0], bottom_len[i][1]);
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
          int d = xy2d(n, x, y);
          for (int c = 0; c < bottom_shape[1]; ++c) {
            bottom_diff[i][c][y][x] += top_diff[i][0][d][c];
          } 
        }
      }
    }
  }
  
 protected:
  string curve_type;
  mshadow::Shape<4> top_shape;
  mshadow::Shape<4> bottom_shape;
  mshadow::Shape<2> top_len_shape;
  mshadow::Shape<2> bottom_len_shape;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_FILL_CURVE_XY2D_LAYER_INL_HPP_

