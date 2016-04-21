#ifndef TEXTNET_LAYER_PARAMETER_LAYER_INL_HPP_
#define TEXTNET_LAYER_PARAMETER_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

// this layer elem-wise products bottom representations on the last 2 dimensions
template<typename xpu>
class ParameterLayer : public Layer<xpu> {
 public:
  ParameterLayer(LayerType type) { this->layer_type = type; }
  virtual ~ParameterLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 1; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["d0"] = SettingV(1);  
    this->defaults["d1"] = SettingV(1);  
    this->defaults["d2"] = SettingV(1);  
    this->defaults["l0"] = SettingV(1);
    this->defaults["l1"] = SettingV(1);
    this->defaults["l_val"] = SettingV("0");
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["batch_size"] = SettingV();  
    this->defaults["w_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);

    d0 = setting["d0"].iVal();
    d1 = setting["d1"].iVal();
    d2 = setting["d2"].iVal();
    l0 = setting["l0"].iVal();
    l1 = setting["l1"].iVal();
    l_val = setting["l_val"].sVal();
    batch_size = setting["batch_size"].iVal();
    
    utils::Check(bottom.size() == BottomNodeNum(), "ParameterLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ParameterLayer:top size problem.");

    int s = 0;
    istringstream iss;
    iss.str(l_val);
    for (int i = 0; i < l1; ++i) {
      iss >> s;
      len_val.push_back(s);
    }

    this->params.resize(1);
    this->params[0].Resize(d0, d1, d2, 1, 1, 1, true);

    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
          w_setting, this->prnd_);
    this->params[0].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
          w_updater, this->prnd_);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "ParameterLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ParameterLayer:top size problem.");
    
    top[0]->Resize(batch_size, d0, d1, d2, l0, l1, true);

	if (show_info) {
      top[0]->PrintShape("top0");
	}
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;
    mshadow::Tensor<xpu, 3> param_data = this->params[0].data_d3();

    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < l1; ++j) {
        top_len[i][j] = len_val[j];
      }
      top_data[i] = F<op::identity>(param_data);
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 3> param_diff = this->params[0].diff_d3();

    for (int i = 0; i < batch_size; ++i) {
      param_diff += top_diff[i];
    }
  }

 protected:
  int d0, d1, d2;
  int l0, l1;
  string l_val;
  vector<int> len_val;
  int batch_size;

};
}  // namespace layer
}  // namespace textnet
#endif
