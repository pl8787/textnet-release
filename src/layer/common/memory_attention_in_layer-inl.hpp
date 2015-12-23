#ifndef TEXTNET_LAYER_MEMORY_ATTENTION_IN_LAYER_INL_HPP_
#define TEXTNET_LAYER_MEMORY_ATTENTION_IN_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class MemoryAttentionInLayer : public Layer<xpu> {
 public:
  MemoryAttentionInLayer(LayerType type) { this->layer_type = type; }
  virtual ~MemoryAttentionInLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "MemoryAttentionInLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MemoryAttentionInLayer:top size problem.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "MemoryAttentionInLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MemoryAttentionInLayer:top size problem.");
    
    mshadow::Shape<4> shape_memory = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_query = bottom[1]->data.shape_;

    utils::Check(shape_memory[0] == 1, "MemoryAttentionInLayer: Shared memory batch size should equal to 1.");
    utils::Check(shape_memory[3] == shape_query[3], "MemoryAttentionInLayer: feat size need equal.");

    top[0]->Resize(shape_query[0], shape_memory[1], 1, 1, 1, 1, true);

	if (show_info) {
      bottom[0]->PrintShape("bottom0");
      bottom[1]->PrintShape("bottom1");
      top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2_middle();
    mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 2> top_data     = top[0]->data_d2();

	top_data = dot(bottom1_data, bottom0_data.T());
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> top_diff     = top[0]->diff_d2();
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2_middle();
    mshadow::Tensor<xpu, 2> bottom0_diff = bottom[0]->diff_d2_middle();
    mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 2> bottom1_diff = bottom[1]->diff_d2();

	if (this->prop_error[0]) {
      bottom0_diff = dot(top_diff.T(), bottom1_data);
	}
	if (this->prop_error[1]) {
      bottom1_diff = dot(top_diff, bottom0_data);
	}

  }
};
}  // namespace layer
}  // namespace textnet
#endif
