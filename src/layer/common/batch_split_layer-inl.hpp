#ifndef TEXTNET_LAYER_BATCH_SPLIT_LAYER_INL_HPP_
#define TEXTNET_LAYER_BATCH_SPLIT_LAYER_INL_HPP_

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
class BatchSplitLayer : public Layer<xpu>{
 public:
  BatchSplitLayer(LayerType type) { this->layer_type = type; }
  virtual ~BatchSplitLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 2; }
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
			      "BatchSplitLayer: batch_step must greater than batch_count");

    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchSplitLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchSplitLayer:top size problem.");

  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchSplitLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchSplitLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    utils::Check(nbatch % batch_step == 0, 
                  "BatchSplitLayer: nbatch div step.");
	top0_nbatch = nbatch / batch_step * batch_count;
	top1_nbatch = nbatch - top0_nbatch;
                  
    top[0]->Resize(top0_nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), bottom[0]->data.size(3), top0_nbatch, bottom[0]->length.size(1), true);
    top[1]->Resize(top1_nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), bottom[0]->data.size(3), top1_nbatch, bottom[0]->length.size(1), true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
		top[1]->PrintShape("top1");
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
	mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top0_data2 = top[0]->data_d2();
	mshadow::Tensor<xpu, 2> top0_len = top[0]->length;
    mshadow::Tensor<xpu, 2> top1_data2 = top[1]->data_d2();
	mshadow::Tensor<xpu, 2> top1_len = top[1]->length;

	int top0_ptr = 0;
	int top1_ptr = 0;

	for (int i = 0; i < nbatch; i += batch_step) {
		for (int j = 0; j < batch_count; ++j) {
			top0_data2[top0_ptr] = F<op::identity>(bottom_data2[i + j]);
			top0_len[top0_ptr] = F<op::identity>(bottom_len[i + j]);
			++top0_ptr;
		}
		for (int j = 0; j < batch_step - batch_count; ++j) {
			top1_data2[top1_ptr] = F<op::identity>(bottom_data2[i + j + batch_count]);
			top1_len[top1_ptr] = F<op::identity>(bottom_len[i + j + batch_count]);
			++top1_ptr;
		}
	}

  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom_diff2 = bottom[0]->diff_d2();
    mshadow::Tensor<xpu, 2> top0_diff2 = top[0]->diff_d2();
    mshadow::Tensor<xpu, 2> top1_diff2 = top[1]->diff_d2();

	int top0_ptr = 0;
	int top1_ptr = 0;
	int bottom_ptr = 0;
	
	while (bottom_ptr < nbatch) {
		for (int i = 0; i < batch_count; ++i) {
			bottom_diff2[bottom_ptr] += F<op::identity>(top0_diff2[top0_ptr]);
			++bottom_ptr;
			++top0_ptr;
		}
		for (int i = 0; i < batch_step - batch_count; ++i) {
			bottom_diff2[bottom_ptr] += F<op::identity>(top1_diff2[top1_ptr]);
			++bottom_ptr;
			++top1_ptr;
		}
	}	

  }
  
 protected:
  int nbatch;
  int top0_nbatch;
  int top1_nbatch;
  int batch_step;
  int batch_count;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_BATCH_SPLIT_LAYER_INL_HPP_

