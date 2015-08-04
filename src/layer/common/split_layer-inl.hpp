#ifndef TEXTNET_LAYER_SPLIT_LAYER_INL_HPP_
#define TEXTNET_LAYER_SPLIT_LAYER_INL_HPP_

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
class SplitLayer : public Layer<xpu>{
 public:
  SplitLayer(LayerType type) { this->layer_type = type; }
  virtual ~SplitLayer(void) {}
  
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
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "SplitLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "SplitLayer:top size problem.");    
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "SplitLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "SplitLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    doc_count = bottom[0]->data.size(1);
    doc_len = bottom[0]->data.size(2);  
    feat_size = bottom[0]->data.size(3);
                  
    top[0]->Resize(nbatch, 1, doc_len, feat_size, true);
    top[1]->Resize(nbatch, 1, doc_len, feat_size, true);

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
	utils::Check(doc_len == bottom[0]->data.size(2), 
			"Split Layer: doc_len.");
	utils::Check(feat_size == bottom[0]->data.size(3),
			"Split Layer: feat_size.");
	if (nbatch != bottom[0]->data.size(0)) {
		need_reshape = true;
		nbatch = bottom[0]->data.size(0);
	}

	// Do reshape 
	if (need_reshape) {
		top[0]->Resize(nbatch, 1, doc_len, feat_size, true);
		top[1]->Resize(nbatch, 1, doc_len, feat_size, true);
		utils::Printf(".");
	}
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
	mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top0_data = top[0]->data;
    mshadow::Tensor<xpu, 4> top1_data = top[1]->data;
	mshadow::Tensor<xpu, 2> top0_len = top[0]->length;
	mshadow::Tensor<xpu, 2> top1_len = top[1]->length;
    
    for (int i = 0; i < nbatch; i++) {
      top0_data[i] = F<op::identity>(bottom_data[i].Slice(0, 1));
      top1_data[i] = F<op::identity>(bottom_data[i].Slice(1, 2));
	  top0_len[i] = F<op::identity>(bottom_len[i].Slice(0, 1));
	  top1_len[i] = F<op::identity>(bottom_len[i].Slice(1, 2));
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top0_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> top1_diff = top[1]->diff;
    if (this->prop_error[0]) {
      for (int i = 0; i < nbatch; i++) {
        bottom_diff[i].Slice(0, 1) += top0_diff[i]; 
        bottom_diff[i].Slice(1, 2) += top1_diff[i];
      }
    }
  }
  
 protected:
  int nbatch;
  int doc_count;
  int doc_len;
  int feat_size;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_SPLIT_LAYER_INL_HPP_

