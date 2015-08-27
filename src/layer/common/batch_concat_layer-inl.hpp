#ifndef TEXTNET_LAYER_BATCH_CONCAT_LAYER_INL_HPP_
#define TEXTNET_LAYER_BATCH_CONCAT_LAYER_INL_HPP_

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
class BatchConcatLayer : public Layer<xpu>{
 public:
  BatchConcatLayer(LayerType type) { this->layer_type = type; }
  virtual ~BatchConcatLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
	
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["batch_step"] = SettingV();
	this->defaults["batch_count"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    batch_step = setting["batch_step"].iVal();
	batch_count = setting["batch_count"].iVal();

	utils::Check(batch_step > batch_count, 
			      "BatchConcatLayer: batch_step must greater than batch_count");

    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchConcatLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchConcatLayer:top size problem.");

  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchConcatLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchConcatLayer:top size problem.");
                  
    bottom0_nbatch = bottom[0]->data.size(0); 
    bottom1_nbatch = bottom[1]->data.size(0); 

    utils::Check(bottom0_nbatch % batch_count == 0, 
                  "BatchConcatLayer: bottom0_nbatch div step.");
    utils::Check(bottom1_nbatch % (batch_step - batch_count) == 0, 
                  "BatchConcatLayer: bottom1_nbatch div step.");

	top_nbatch = bottom0_nbatch / batch_count * batch_step;
                  
    top[0]->Resize(top_nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), bottom[0]->data.size(3), top_nbatch, bottom[0]->length.size(1), true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        bottom[1]->PrintShape("bottom1");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (bottom0_nbatch != bottom[0]->data.size(0) || bottom1_nbatch != bottom[1]->data.size(0)) {
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
    mshadow::Tensor<xpu, 2> bottom0_data2 = bottom[0]->data_d2();
	mshadow::Tensor<xpu, 2> bottom0_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> bottom1_data2 = bottom[1]->data_d2();
	mshadow::Tensor<xpu, 2> bottom1_len = bottom[1]->length;
    mshadow::Tensor<xpu, 2> top_data2 = top[0]->data_d2();
	mshadow::Tensor<xpu, 2> top_len = top[0]->length;

	int bottom0_ptr = 0;
	int bottom1_ptr = 0;
	int top_ptr = 0;

	while (top_ptr < top_nbatch) {
		for (int i = 0; i < batch_count; ++i) {
			top_data2[top_ptr] = F<op::identity>(bottom0_data2[bottom0_ptr]);
			top_len[top_ptr] = F<op::identity>(bottom0_len[bottom0_ptr]);
			++top_ptr;
			++bottom0_ptr;
		}
		for (int i = 0; i < batch_step - batch_count; ++i) {
			top_data2[top_ptr] = F<op::identity>(bottom1_data2[bottom1_ptr]);
			top_len[top_ptr] = F<op::identity>(bottom1_len[bottom1_ptr]);
			++top_ptr;
			++bottom1_ptr;
		}
	}
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_diff2 = bottom[0]->diff_d2();
    mshadow::Tensor<xpu, 2> bottom1_diff2 = bottom[1]->diff_d2();
    mshadow::Tensor<xpu, 2> top_diff2 = top[0]->diff_d2();

	int bottom0_ptr = 0;
	int bottom1_ptr = 0;

	for (int i = 0; i < top_nbatch; i += batch_step) {
		for (int j = 0; j < batch_count; ++j) {
			bottom0_diff2[bottom0_ptr] = F<op::identity>(top_diff2[i + j]);
			++bottom0_ptr;
		}
		for (int j = 0; j < batch_step - batch_count; ++j) {
			bottom1_diff2[bottom1_ptr] = F<op::identity>(top_diff2[i + j + batch_count]);
			++bottom1_ptr;
		}
	}
  }
  
 protected:
  int top_nbatch;
  int bottom0_nbatch;
  int bottom1_nbatch;
  int batch_step;
  int batch_count;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_BATCH_CONCAT_LAYER_INL_HPP_

