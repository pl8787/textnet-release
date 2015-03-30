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

        top_data[i][0] = pool<Reducer>(bottom_data[i][0].Slice(begin,end), pshape, bottom_data.size(2), 1, 1);
    }
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
        bottom_diff = unpool<Reducer>(bottom_data[i][0].Slice(begin, end), top_data, top_diff, end-begin, 1, 1);
      }
    }
  }
  void LocateBeginEnd(mshadow::Tensor<xpu, 2> seq, 
                      index_t &begin, index_t &end) { // input a 2D tensor, out put a sub 2d tensor, with 0 padding
    for (index_t i = 0; i < seq.size(0); ++i) {
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
  }
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_WHOLEPOOLING_LAYER_INL_HPP_

