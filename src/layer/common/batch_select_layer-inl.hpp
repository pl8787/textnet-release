#ifndef TEXTNET_LAYER_BATCH_SELECT_LAYER_INL_HPP_
#define TEXTNET_LAYER_BATCH_SELECT_LAYER_INL_HPP_

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
class BatchSelectLayer : public Layer<xpu>{
 public:
  BatchSelectLayer(LayerType type) { this->layer_type = type; }
  virtual ~BatchSelectLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
	
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["step"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    step = setting["step"].iVal();

    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchSelectLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchSelectLayer:top size problem.");

  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchSelectLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchSelectLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    utils::Check(nbatch % step == 0, 
                  "BatchSelectLayer: nbatch div step.");
	out_nbatch = nbatch / step;
                  
    top[0]->Resize(out_nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), bottom[0]->data.size(3), out_nbatch, bottom[0]->length.size(1), true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (nbatch != bottom[0]->data.size(0)) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom_data2 = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> top_data2 = top[0]->data_d2();
	mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
	mshadow::Tensor<xpu, 2> top_len = top[0]->length;

	for (int i = 0, j = 0; i < nbatch; i += step, j += 1) {
		top_data2[j] = F<op::identity>(bottom_data2[i]);
		top_len[j] = F<op::identity>(bottom_len[i]);
	}

  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom_diff2 = bottom[0]->diff_d2();
    mshadow::Tensor<xpu, 2> top_diff2 = top[0]->diff_d2();

	for (int i = 0, j = 0; i < nbatch; i += step, j += 1) {
		bottom_diff2[i] = F<op::identity>(top_diff2[j]);
	}	

  }
  
 protected:
  int nbatch;
  int out_nbatch;
  int step;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_BATCH_SELECT_LAYER_INL_HPP_

