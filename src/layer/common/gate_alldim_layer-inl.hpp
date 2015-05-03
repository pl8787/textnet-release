#ifndef TEXTNET_LAYER_GATE_ALLDIM_LAYER_INL_HPP_
#define TEXTNET_LAYER_GATE_ALLDIM_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class GateAlldimLayer : public Layer<xpu> {
 public:
  GateAlldimLayer(LayerType type) { this->layer_type = type; }
  virtual ~GateAlldimLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "GateAlldimLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GateAlldimLayer:top size problem.");
                            
    no_bias = setting["no_bias"].bVal();
    num_input = bottom[0]->data.size(3);

    this->params.resize(2);
    this->params[0].Resize(num_input, num_input, 1, 1, true);
    this->params[1].Resize(num_input, 1, 1, 1, true);
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
          w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), 
          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
          w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
          b_updater, this->prnd_);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "GateAlldimLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GateAlldimLayer:top size problem.");
    
    int batch_size = bottom[0]->data.size(0);
    int num_seq    = bottom[0]->data.size(1);
    int max_length = bottom[0]->data.size(2);
    int feat_size  = bottom[0]->data.size(3);
    
    top[0]->Resize(batch_size, num_seq, max_length, feat_size, true);
    gate.Resize(batch_size, num_seq, max_length, feat_size, true), 

	bottom[0]->PrintShape("bottom0");
	top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    top[0]->length = F<op::identity>(bottom[0]->length);

    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> gate_data   = gate.data_d2_reverse();
    mshadow::Tensor<xpu, 2> top_data    = top[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> w_data      = this->params[0].data_d2();
    
    gate_data  = dot(bottom_data,  w_data);
    if (!no_bias) {
      utils::Check(false, "Gate Layer: compile error.");
      // gate_data += repmat(b_data, gate_data.size(0));
    }
    gate_data = F<op::sigmoid>(gate_data);

    // orc, this layer dose not deal variable length
    for (int i = 0; i < bottom_data.size(0); ++i) {
        top_data[i] = bottom_data[i] * gate_data[i] * 2;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> top_diff    = top[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> gate_data   = gate.data_d2_reverse();
    mshadow::Tensor<xpu, 2> gate_diff   = gate.diff_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom_diff = bottom[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> w_data      = this->params[0].data_d2();

    gate_diff = top_diff * bottom_data * 2;
    gate_diff *= F<op::sigmoid_grad>(gate_data);
    
    bottom_diff += top_diff * gate_data * 2;
    
    this->params[0].diff_d2() += dot(bottom_data.T(), gate_diff);
    if (!no_bias) {
      utils::Check(false, "Gate Layer: compile error.");
      // this->params[1].diff_d1() += sum_rows(gate_diff);
    }
    
    bottom_diff += dot(gate_diff, w_data.T());
  }

 protected:
  /*! \brief random number generator */
  int num_input;
  bool no_bias;
  Node<xpu> gate;
};
}  // namespace layer
}  // namespace textnet
#endif
