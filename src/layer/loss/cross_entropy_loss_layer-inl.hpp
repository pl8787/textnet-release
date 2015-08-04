#ifndef TEXTNET_LAYER_CROSS_ENTROPY_LOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_CROSS_ENTROPY_LOSS_LAYER_INL_HPP_

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
class CrossEntropyLossLayer : public Layer<xpu>{
 public:
  CrossEntropyLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~CrossEntropyLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(), "CrossEntropyLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "CrossEntropyLossLayer:top size problem.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "CrossEntropyLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "CrossEntropyLossLayer:top size problem.");
                  
    top[0]->Resize(1, 1, 1, 1, true);
	if (show_info) {
		top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();
    
    top_data = 0.0f;
    int batch_size = bottom0_data.size(0);    
    for (int i = 0; i < batch_size; ++i) {
      int k = static_cast<int>(bottom1_data[i]);
      if (bottom0_data[i][k] == 0.) {
        top_data[0] += 88; // by min float number
      } else { 
        top_data[0] += -log(bottom0_data[i][k]);
      }
    }
    top_data[0] /= batch_size;
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 2> bottom0_diff = bottom[0]->diff_d2();
    
    int batch_size = bottom0_data.size(0);    
    for (int i = 0; i < batch_size; ++i) {
      int k = static_cast<int>(bottom1_data[i]);
      if (bottom0_data[i][k] <= 0.000001f) {
        utils::Check(false, "CrossEntropyLossLayer: prob zero error.");
      } else {
        bottom0_diff[i][k] -= (1.0f/bottom0_data[i][k]); 
      }
    }
  }
  
 // protected:

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CORSS_ENTROPY_LAYER_INL_HPP_

