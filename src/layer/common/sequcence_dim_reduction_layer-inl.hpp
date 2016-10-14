#ifndef TEXTNET_LAYER_SEQUENCE_DIM_REDUCTION_LAYER_INL_HPP_
#define TEXTNET_LAYER_SEQUENCE_DIM_REDUCTION_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class SequenceDimReductionLayer : public Layer<xpu> {
 public:
  SequenceDimReductionLayer(LayerType type) { this->layer_type = type; }
  virtual ~SequenceDimReductionLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 1; }
  
  virtual void Require() {
    // default value, just set the value you want
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["num_hidden"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "SequenceDimReductionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SequenceDimReductionLayer:top size problem.");
                            
    this->param_file = setting["param_file"].sVal();
    num_hidden = setting["num_hidden"].iVal();
    num_input = bottom[0]->data.size(3);

    this->params.resize(1);
    this->params[0].Resize(num_hidden, num_input, 1, 1, true);
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
          w_setting, this->prnd_);
    this->params[0].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
          w_updater, this->prnd_);
    if (!this->param_file.empty()) {
      this->LoadParams();
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "SequenceDimReductionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SequenceDimReductionLayer:top size problem.");
    
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    // utils::Check(bottom_data.size(1) == 1, "SequenceDimReductionLayer:bottom size problem.");
    
    top[0]->Resize(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), num_hidden, true);

	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2_reverse();

    top[0]->length = bottom[0]->length;
    top[0]->data_d2_reverse() = dot(bottom_data, this->params[0].data_d2().T());
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> top_diff = top[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom_diff = bottom[0]->diff_d2_reverse();
    
    // if (this->prop_grad[0]) {
    this->params[0].diff_d2() += dot(top_diff.T(), bottom_data);
    // }
    bottom_diff += dot(top_diff, this->params[0].data_d2());
  }

 protected:
  /*! \brief random number generator */
  int num_input, num_hidden;
};
}  // namespace layer
}  // namespace textnet
#endif 
