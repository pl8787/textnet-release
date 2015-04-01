#ifndef TEXTNET_LAYER_WHOLEPOOLING_LAYER_INL_HPP_
#define TEXTNET_LAYER_WHOLEPOOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename Reducer, typename xpu>
class WholePoolingLayer : public Layer<xpu>{
 public:
  WholePoolingLayer(LayerType type) { this->layer_type = type; }
  virtual ~WholePoolingLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "WholePoolingLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "WholePoolingLayer:top size problem.");
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], 1, shape_in[3]);
    top[0]->Resize(shape_out);
    pos.Resize(shape_out);
  }

  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  void wholeAvePooling(Tensor2D in, Tensor2D out) {
    utils::Check(in.size(1) == out.size(1) && out.size(0) == 1, "WholePoolingLayer:pooling io size error");
    out = 0.;
    for (index_t row = 0; row < in.size(0); ++row) {
      out += in.Slice(row, row+1);
    }
    out /= float(in.size(0));
  }
  void wholeUnAvePooling(Tensor2D in, Tensor2D out) {
    utils::Check(in.size(1) == out.size(1) && in.size(0) == 1, "WholePoolingLayer:pooling io size error");
    for (index_t row = 0; row < out.size(0); ++row) {
      out.Slice(row, row+1) = in / (float)(out.size(0));
    }
  }

  void wholeMaxPooling(Tensor2D in, Tensor2D pos, Tensor2D out) {
    utils::Check(in.size(1) == out.size(1) && out.size(0) == 1, "WholePoolingLayer:pooling io size error");
    utils::Check(in.size(1) == pos.size(1) && pos.size(0) == 1, "WholePoolingLayer:pooling io size error");
    out = -1000000;
    for (index_t col = 0; col < in.size(1); ++col) {
      for (index_t row = 0; row < in.size(0); ++row) {
        if (in[row][col] > out[0][col]) {
            out[0][col] = in[row][col];
            pos[0][col] = row;
        }
      }
    }
  }
  void wholeUnMaxPooling(Tensor2D in, Tensor2D pos, Tensor2D out) {
    utils::Check(in.size(1) == out.size(1) && in.size(0) == 1, "WholePoolingLayer:pooling io size error");
    utils::Check(in.size(1) == pos.size(1) && pos.size(0) == 1, "WholePoolingLayer:pooling io size error");
    out = 0.;
    for (index_t col = 0; col < in.size(1); ++col) {
      index_t row = pos[0][col];
      out[row][col] = in[0][col];
    }
  }

  void checkNan(float *p, int l) {
      for (int i = 0; i < l; ++i) {
          assert(!isnan(p[i]));
      }
  }
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Shape<2> pshape = top_data[0][0].shape_;

    for (index_t i = 0; i < bottom_data.size(0); ++i) {
        int begin, end; 
        LocateBeginEnd(bottom_data[i][0], begin, end);
        // wholePooling(bottom_data[i][0].Slice(begin,end), pos[i][0], top_data[i][0]);
        wholeAvePooling(bottom_data[i][0].Slice(begin,end), top_data[i][0]);
    }
    checkNan(top_data.dptr_, top_data.size(3) * top_data.size(0));
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;

    bottom_diff = NAN;
    for (index_t i = 0; i < bottom_data.size(0); ++i) {
      int begin, end; 
      LocateBeginEnd(bottom_data[i][0], begin, end);

      if (this->prop_error[0]) {
        wholeUnAvePooling(top_diff[i][0], bottom_diff[i][0].Slice(begin, end));
        // wholeUnPooling(top_diff[i][0], pos[i][0], bottom_diff[i][0].Slice(begin, end));
      }
    }
  }
  void LocateBeginEnd(mshadow::Tensor<xpu, 2> seq, 
                      int &begin, int &end) { // input a 2D tensor, out put a sub 2d tensor, with 0 padding
    for (int i = 0; i < seq.size(0); ++i) {
      if (!isnan(seq[i][0])) {
          begin = i;
          break;
      }
    }
    for (int i = seq.size(0)-1; i >= 0; --i) {
      if (!isnan(seq[i][0])) {
          end = i + 1;
          break;
      }
    }
    utils::Check(begin < end && begin >= 0, "WholePoolingLayer: input error."); 
  }
 protected:
  mshadow::TensorContainer<xpu, 4> pos;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_WHOLEPOOLING_LAYER_INL_HPP_

