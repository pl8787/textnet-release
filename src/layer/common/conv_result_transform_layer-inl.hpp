#ifndef TEXTNET_LAYER_CONV_RESULT_TRANSFORM_LAYER_INL_HPP_
#define TEXTNET_LAYER_CONV_RESULT_TRANSFORM_LAYER_INL_HPP_

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
class ConvResultTransformLayer : public Layer<xpu>{
 public:
  ConvResultTransformLayer(LayerType type) { this->layer_type = type; }
  virtual ~ConvResultTransformLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "ConvResultTransformLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ConvResultTransformLayer:top size problem.");    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "ConvResultTransformLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ConvResultTransformLayer:top size problem.");
                  
    int batch_size = bottom[0]->data.size(0); 
    int channel_out = bottom[0]->data.size(1);
    int doc_len = bottom[0]->data.size(2);  
    utils::Check(bottom[0]->data.size(3) == 1, "ConvResultTransformLayer: bottom size problem.");
                  
    top[0]->Resize(batch_size, 1, doc_len, channel_out, true);

	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    mshadow::Tensor<xpu, 3> bottom_data = bottom[0]->data_d3();
	mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
	mshadow::Tensor<xpu, 2> top_len = top[0]->length;

    for (int i = 0; i < bottom_data.size(0); ++i) {
      top_data[i][0] = bottom_data[i].T();
      top_len[i][0]  = bottom_len[i][0];
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    mshadow::Tensor<xpu, 3> bottom_diff = bottom[0]->diff_d3();
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    if (this->prop_error[0]) {
      for (int i = 0; i < bottom_diff.size(0); ++i) {
        bottom_diff[i] += top_diff[i][0].T(); 
      }
    }
  }
};
}  // namespace layer
}  // namespace textnet
#endif

