#ifndef TEXTNET_LAYER_CONV_LSTM_SPLIT_LAYER_INL_HPP_
#define TEXTNET_LAYER_CONV_LSTM_SPLIT_LAYER_INL_HPP_

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
class ConvLstmSplitLayer : public Layer<xpu>{
 public:
  ConvLstmSplitLayer(LayerType type) { this->layer_type = type; }
  virtual ~ConvLstmSplitLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 2; }
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "ConvLstmSplitLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ConvLstmSplitLayer:top size problem.");    
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "ConvLstmSplitLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ConvLstmSplitLayer:top size problem.");
                  
    batch_size= bottom[0]->data.size(0); 
    doc_count = bottom[0]->data.size(1);
    doc_len   = bottom[0]->data.size(2);  
    feat_size = bottom[0]->data.size(3);

    utils::Assert(doc_count == 2, "ConvLstmSplitLayer:bottom size problem");
                  
    top[0]->Resize(batch_size, feat_size, doc_len, 1, true);
    top[1]->Resize(batch_size, feat_size, doc_len, 1, true);

	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		top[0]->PrintShape("top0");
		top[1]->PrintShape("top1");
	}
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top0_data = top[0]->data;
    mshadow::Tensor<xpu, 4> top1_data = top[1]->data;
    
    for (index_t i = 0; i < batch_size; ++i) {
      // for (index_t j = 0; j < doc_count; ++j) {
        for (index_t m = 0; m < doc_len; ++m) {
          for (index_t n = 0; n < feat_size; ++n) {
            top0_data[i][n][m][0] = bottom_data[i][0][m][n];
            top1_data[i][n][m][0] = bottom_data[i][1][m][n];
          }
        }
      // }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top0_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> top1_diff = top[1]->diff;

    if (this->prop_error[0]) {
      for (index_t i = 0; i < batch_size; ++i) {
        // for (index_t j = 0; j < doc_count; ++j) {
          for (index_t m = 0; m < doc_len; ++m) {
            for (index_t n = 0; n < feat_size; ++n) {
              bottom_diff[i][0][m][n] += top0_diff[i][n][m][0];
              bottom_diff[i][1][m][n] += top1_diff[i][n][m][0];
            }
          }
        // }
      }
    }
  }
  
 protected:
  int batch_size;
  int doc_count;
  int doc_len;
  int feat_size;

};
}  // namespace layer
}  // namespace textnet
#endif 

