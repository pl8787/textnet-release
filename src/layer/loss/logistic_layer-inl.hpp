#ifndef TEXTNET_LAYER_LOGISTIC_LAYER_INL_HPP_
#define TEXTNET_LAYER_LOGISTIC_LAYER_INL_HPP_

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
class LogisticLayer : public Layer<xpu>{
 public:
  LogisticLayer(LayerType type) { this->layer_type = type; }
  virtual ~LogisticLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(), "LogisticLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LogisticLayer:top size problem.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "LogisticLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LogisticLayer:top size problem.");
                  
    top[0]->Resize(1, 1, 1, 1, true);
	if (show_info) {
		top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();
    
    top_data = 0.0f;
    int batch_size = bottom0_data.size(0);    

	utils::Check(batch_size==bottom[0]->data.size(0),
			"LogisticLayer: Input need only one value.");

	// Apply logistic function
	bottom0_data = F<op::sigmoid>(bottom0_data);

    for (int i = 0; i < batch_size; ++i) {
      int k = static_cast<int>(bottom1_data[i]);
	  utils::Check(k==0 || k==1, "LogisticLayer: Only support binary class.");
      if (k == 0) {
        top_data[0] += -log(1.0 - bottom0_data[i]);
      } else if (k == 1) { 
        top_data[0] += -log(bottom0_data[i]);
      }
    }

    top_data[0] /= batch_size;
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> bottom0_diff = bottom[0]->diff_d1();
    
    int batch_size = bottom0_data.size(0);    
    for (int i = 0; i < batch_size; ++i) {
      int k = static_cast<int>(bottom1_data[i]);
	  utils::Check(k==0 || k==1, "LogisticLayer: Only support binary class.");
      if (k == 0) {
        bottom0_diff[i] += bottom0_data[i];
      } else if (k == 1) { 
        bottom0_diff[i] += bottom0_data[i] - 1.0;
      }
    }
  }
  
 // protected:

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CORSS_ENTROPY_LAYER_INL_HPP_

