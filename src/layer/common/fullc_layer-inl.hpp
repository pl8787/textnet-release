#ifndef TEXTNET_LAYER_FULLC_LAYER_INL_HPP_
#define TEXTNET_LAYER_FULLC_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class FullConnectLayer : public Layer<xpu> {
 public:
  FullConnectLayer(LayerType type) { this->layer_type = type; }
  virtual ~FullConnectLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "FullConnectionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "FullConnectionLayer:top size problem.");
                            
    num_hidden = setting["num_hidden"].i_val;
    no_bias = setting["no_bias"].b_val;

    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2();
    num_input = bottom_data.size(1);

    this->params.resize(2);
    this->params[0].Resize(num_hidden, num_input, 1, 1);
    this->params[1].Resize(num_hidden, 1, 1, 1);
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].m_val;
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].m_val;
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].i_val,
          w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].i_val, 
          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].m_val;
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].m_val;
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].i_val,
          w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].i_val,
          b_updater, this->prnd_);
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "FullConnectionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "FullConnectionLayer:top size problem.");
    
    mshadow::Tensor<xpu, 2> bottom_data = bottom[0]->data_d2();
    
    top[0]->Resize(bottom_data.size(0), num_hidden, 1, 1);
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
      this->params[0].diff_d2() = dot(top_diff.T(), bottom_data);
    }
    if (!no_bias && this->prop_grad[1]) {
      this->params[1].diff_d1() = sum_rows(top_diff);
    }
    
    if (this->prop_error[0]) {
      bottom_diff = dot(top_diff, this->params[0].data_d2());
    }
  }
  
  virtual void SaveModel(utils::IStream &fo) const {

  }
  virtual void LoadModel(utils::IStream &fi) {

  }

 protected:
  /*! \brief random number generator */
  int num_input;
  int num_hidden;
  bool no_bias;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_FULLC_LAYER_INL_HPP_
