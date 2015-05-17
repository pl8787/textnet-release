#ifndef TEXTNET_LAYER_LR2SOFTMAX_LAYER_INL_HPP_
#define TEXTNET_LAYER_LR2SOFTMAX_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

// sum aross one axis
template<typename xpu>
class Lr2softmaxLayer : public Layer<xpu> {
 public:
  Lr2softmaxLayer(LayerType type) { this->layer_type = type; }
  virtual ~Lr2softmaxLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 1; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["score_class"] = SettingV(); // the other class will be pad as zero
    this->defaults["rescale"]     = SettingV(); // this is to rescale the score value
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "Lr2softmaxLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "Lr2softmaxLayer:top size problem.");
    score_class = setting["score_class"].iVal();
    rescale = setting["rescale"].fVal();
    utils::Check(0 == score_class || 1 == score_class, "Lr2softmaxLayer: score class setting error.");

    this->params.resize(1);
    this->params[0].Resize(1, 1, 1, 1, true);
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), 
          b_setting, this->prnd_);
    this->params[0].Init();
    
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
          b_updater, this->prnd_);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "Lr2softmaxLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "Lr2softmaxLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    utils::Check(1 == shape_in[1] && 1 == shape_in[2] && 1 == shape_in[3], "Lr2softmaxLayer: input size error.");
    mshadow::Shape<4> shape_out= shape_in;
    shape_out[1] = 2;

    top[0]->Resize(shape_out, true);

	bottom[0]->PrintShape("bottom0");
	top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    mshadow::Tensor<xpu, 1> bottom_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 2> top_data    = top[0]->data_d2();
    mshadow::Tensor<xpu, 1> b_data      = this->params[0].data_d1();
    top_data = 0.f;
    int bias_class = score_class == 0 ? 1 : 0;
    for (int i = 0; i < bottom_data.size(0); ++i) {
      top_data[i][score_class] = bottom_data[i] * rescale; 
      top_data[i][bias_class]  = b_data[0];
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    mshadow::Tensor<xpu, 1> bottom_diff = bottom[0]->diff_d1();
    mshadow::Tensor<xpu, 2> top_diff    = top[0]->diff_d2();
    mshadow::Tensor<xpu, 1> b_diff      = this->params[0].diff_d1();
    int bias_class = score_class == 0 ? 1 : 0;
    for (int i = 0; i < bottom_diff.size(0); ++i) {
      bottom_diff[i] += top_diff[i][score_class];
      b_diff[0] += top_diff[i][bias_class];
    }
  }
 protected:
  int score_class;
  float rescale;
};
}  // namespace layer
}  // namespace textnet
#endif
