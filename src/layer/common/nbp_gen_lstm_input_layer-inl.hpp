#ifndef TEXTNET_LAYER_NBP_GEN_LSTM_INPUT_LAYER_INL_HPP_
#define TEXTNET_LAYER_NBP_GEN_LSTM_INPUT_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class NbpGenLstmInputLayer : public Layer<xpu>{
 public:
  NbpGenLstmInputLayer(LayerType type) { this->layer_type = type; }
  virtual ~NbpGenLstmInputLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; } // embedding, len
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
    utils::Check(bottom.size() == BottomNodeNum(), "NbpGenLstmInputLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "NbpGenLstmInputLayer:top size problem.");

    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], 1, shape_in[1], shape_in[3]);
    top[0]->Resize(shape_out, true);

    bottom[0]->PrintShape("bottom0");
    bottom[1]->PrintShape("bottom1");
    top[0]->PrintShape("top0");
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_len  = bottom[1]->data; // note: this layer is to merge the length to one node
    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len     = top[0]->length;

    top_data = 0;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      top_len[batch_idx][0] = bottom_len[batch_idx][0][0][0];
      utils::Check(top_len[batch_idx][0] <= bottom_data.shape_[1], "NbpGenLstmInputLayer: length error.");
      for (index_t i = 0; i < bottom_len[batch_idx][0][0][0]; ++i) {
        top_data[batch_idx][0][i] = F<op::identity>(bottom_data[batch_idx][i][0]);
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 2> top_len  = top[0]->length;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;

    for (index_t batch_idx = 0; batch_idx < top_diff.size(0); ++batch_idx) {
      for (index_t i = 0; i < top_len[batch_idx][0]; ++i) {
        bottom_diff[batch_idx][i][0] += top_diff[batch_idx][0][i];
      }
    }
  }
 protected:
};
}  // namespace layer
}  // namespace textnet
#endif

