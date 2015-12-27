#ifndef TEXTNET_LAYER_CHANNEL_DUPLICATE_LAYER_INL_HPP_
#define TEXTNET_LAYER_CHANNEL_DUPLICATE_LAYER_INL_HPP_

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
class ChannelDuplicateLayer : public Layer<xpu>{
 public:
  ChannelDuplicateLayer(LayerType type) { this->layer_type = type; }
  virtual ~ChannelDuplicateLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
	
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["dup_count"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    dup_count = setting["dup_count"].iVal();

	utils::Check(dup_count > 0, 
			      "ChannelDuplicateLayer: dup_count need > 0.");

    utils::Check(bottom.size() == BottomNodeNum(),
                  "ChannelDuplicateLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ChannelDuplicateLayer:top size problem.");

  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ChannelDuplicateLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ChannelDuplicateLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
	nchannel = bottom[0]->data.size(1);

	top_nchannel = nchannel * dup_count;
                  
    top[0]->Resize(nbatch, top_nchannel, bottom[0]->data.size(2), bottom[0]->data.size(3), nbatch, top_nchannel, true);

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
    mshadow::Tensor<xpu, 3> bottom_data3 = bottom[0]->data_d3();
	mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 3> top_data3 = top[0]->data_d3();
	mshadow::Tensor<xpu, 2> top_len = top[0]->length;

	int bottom_ptr = 0;
	int top_ptr = 0;

	for (int i = 0; i < nbatch; ++i) {
		bottom_ptr = 0;
		top_ptr = 0;
		while (top_ptr < top_nchannel) {
			for (int c = 0; c < dup_count; ++c) {
				top_data3[i][top_ptr] = F<op::identity>(bottom_data3[i][bottom_ptr]);
				top_len[i][top_ptr] = bottom_len[i][bottom_ptr];
				++top_ptr;
			}
			++bottom_ptr;
		}
	}
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> bottom_diff3 = bottom[0]->diff_d3();
    mshadow::Tensor<xpu, 3> top_diff3 = top[0]->diff_d3();

	int bottom_ptr = 0;
	int top_ptr = 0;

	for (int i = 0; i < nbatch; ++i) {
		bottom_ptr = 0;
		top_ptr = 0;
		while (top_ptr < top_nchannel) {
			for (int c = 0; c < dup_count; ++c) {
				bottom_diff3[i][bottom_ptr] += F<op::identity>(top_diff3[i][top_ptr]);
				++top_ptr;
			}
			++bottom_ptr;
		}
	}
  }
  
 protected:
  int nchannel;
  int top_nchannel;
  int nbatch;
  int dup_count;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CHANNEL_DUPLICATE_LAYER_INL_HPP_

