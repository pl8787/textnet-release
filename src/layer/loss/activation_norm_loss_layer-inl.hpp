#ifndef TEXTNET_LAYER_ACTIVATION_NORM_LOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_ACTIVATION_NORM_LOSS_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

// this layer is originally designed for adding norm on activations
template<typename xpu>
class ActivationNormLossLayer : public Layer<xpu>{
 public:
  ActivationNormLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~ActivationNormLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }  
  virtual int TopNodeNum() { return 1; }    
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["norm"] = SettingV(2); // Support L1 norm and L2 norm
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["mu"] = SettingV(); // Loss weight

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(), "Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ActivationNormLossLayer:top size problem.");

	norm = setting["norm"].iVal();
    mu = setting["mu"].fVal();

  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "ActivationNormLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ActivationNormLossLayer:top size problem.");

    top[0]->Resize(1, 1, 1, 1, true);
	if (show_info) {
      top[0]->PrintShape("top0");
	}
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data   = bottom[0]->data_d2();

    float loss = 0.f;
	if (norm == 2) {
      for (int i = 0; i < bottom0_data.size(0); ++i) {
        for (int j = 0; j < bottom0_data.size(1); ++j) {
          loss += bottom0_data[i][j] * bottom0_data[i][j]; 
		}
	  }
	} else if (norm == 1) {
      for (int i = 0; i < bottom0_data.size(0); ++i) {
        for (int j = 0; j < bottom0_data.size(1); ++j) {
		  loss += fabs(bottom0_data[i][j]);
		}
      }
    } else if (norm == -1) {
      for (int i = 0; i < bottom0_data.size(0); ++i) {
        for (int j = 0; j < bottom0_data.size(1); ++j) {
          loss += bottom0_data[i][j] * bottom0_data[i][j] - bottom0_data[i][j] + 0.25; 
		}
	  }
	}
    top[0]->data[0][0][0][0] = loss/float(bottom0_data.size(0));
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> bottom0_diff = bottom[0]->diff_d2();

	if (norm == 2) {
      bottom0_diff += bottom0_data;
	} else if (norm == 1) {
      for (int i = 0; i < bottom0_data.size(0); ++i) {
        for (int j = 0; j < bottom0_data.size(1); ++j) {
		  if (bottom0_data[i][j] > 0) {
			bottom0_diff[i][j] += 1;
		  } else if (bottom0_data[i][j] < 0) {
			bottom0_diff[i][j] += -1;
		  } 
        }
      }
	} else if (norm == -1) {
	  bottom0_diff += 2*bottom0_data-1.0;
	}
	bottom0_diff *= mu / float(bottom0_data.size(0));
  }

protected:
  float mu;
  int norm;
};
}  // namespace layer
}  // namespace textnet

#endif
