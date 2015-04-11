#ifndef TEXTNET_LAYER_FULLC_LAYER_INL_HPP_
#define TEXTNET_LAYER_FULLC_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class TensorFullConnectLayer : public Layer<xpu> {
 public:
  TensorFullConnectLayer(LayerType type) { this->layer_type = type; }
  virtual ~TensorFullConnectLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 3; }
  
  virtual void Require() {
    // default value, just set the value you want
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["mode"] = SettingV(); // string value, 't_*_w_*_b_*', * is 1 or 0
    this->defaults["num_hidden"] = SettingV();
    this->defaults["t_filler"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["t_updater"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "TensorFullConnectionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "TensorFullConnectionLayer:top size problem.");
                            
    num_hidden = setting["num_hidden"].iVal();

    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2();
    num_input = bottom_data.size(1);

    this->params.resize(2);
    this->params[0].Resize(num_hidden, num_input, num_input, 1, true);
    this->params[1].Resize(num_hidden, num_input, 1, 1, true);
    this->params[2].Resize(num_hidden, 1, 1, 1, true);
    
    std::map<std::string, SettingV> &t_setting = *setting["t_filler"].mVal();
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(t_setting["init_type"].iVal(),
          t_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
          w_setting, this->prnd_);
    this->params[2].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), 
          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    this->params[2].Init();
    
    std::map<std::string, SettingV> &t_updater = *setting["t_updater"].mVal();
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(t_updater["updater_type"].iVal(),
          t_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
          w_updater, this->prnd_);
    this->params[2].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
          b_updater, this->prnd_);
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "TensorFullConnectionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "TensorFullConnectionLayer:top size problem.");
    
    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2();
    
    top[0]->Resize(bottom_data.size(0), num_hidden, 1, 1, true);

	bottom[0]->PrintShape("bottom0");
	top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2();

    top[0]->data_d2() = dot(bottom_data, this->params[0].data_d2().T());
    if (!no_bias) {
      int nbatch = bottom_data.size(0);
      top[0]->data_d2() += repmat(this->params[1].data_d1(), nbatch);
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> top_diff = top[0]->diff_d2();
    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> bottom_diff = bottom[0]->diff_d2();
    
    if (this->prop_grad[0]) {
      this->params[0].diff_d2() += dot(top_diff.T(), bottom_data);
    }
    if (!no_bias && this->prop_grad[1]) {
      this->params[1].diff_d1() += sum_rows(top_diff);
    }
    
    if (this->prop_error[0]) {
      bottom_diff += dot(top_diff, this->params[0].data_d2());
    }
  }

 protected:
  /*! \brief random number generator */
  int num_input;
  int num_hidden;
  bool is_t, is_w, is_b;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_FULLC_LAYER_INL_HPP_
