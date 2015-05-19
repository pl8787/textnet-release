#ifndef TEXTNET_LAYER_DUPLICATE4LSTM_INL_HPP_
#define TEXTNET_LAYER_DUPLICATE4LSTM_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class Duplicate4lstmLayer : public Layer<xpu> {
 public:
  Duplicate4lstmLayer(LayerType type) { this->layer_type = type; }
  virtual ~Duplicate4lstmLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["num_duplicate"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "Duplicate4lstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "Duplicate4lstmLayer:top size problem.");
                            
    num_duplicate = setting["num_duplicate"].iVal();
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                 "Duplicate4lstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                 "Duplicate4lstmLayer:top size problem.");
    
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = shape_in;
    shape_out[2] *= num_duplicate;
    
    top[0]->Resize(shape_out, true);

	bottom[0]->PrintShape("bottom0");
	top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len     = top[0]->length;

    for (int batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      for (int seq_idx = 0; seq_idx < bottom_data.size(1); ++seq_idx) {
        int length = bottom_len[batch_idx][seq_idx];
        top_len[batch_idx][seq_idx] = length * num_duplicate;
        for (int m = 0; m < num_duplicate; ++m) {
          for (int n = 0; n < length; ++n) {
            top_data[batch_idx][seq_idx][m*length+n] = F<op::identity>(bottom_data[batch_idx][seq_idx][n]);
          }
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_diff    = top[0]->diff;

    for (int batch_idx = 0; batch_idx < bottom_diff.size(0); ++batch_idx) {
      for (int seq_idx = 0; seq_idx < bottom_diff.size(1); ++seq_idx) {
        int length = bottom_len[batch_idx][seq_idx];
        for (int m = 0; m < num_duplicate; ++m) {
          for (int n = 0; n < length; ++n) {
            bottom_diff[batch_idx][seq_idx][n] += top_diff[batch_idx][seq_idx][m*length+n];
          }
        }
      }
    }
  }

 protected:
  /*! \brief random number generator */
  int num_duplicate;
};
}  // namespace layer
}  // namespace textnet
#endif
