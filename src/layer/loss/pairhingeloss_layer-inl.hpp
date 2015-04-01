#ifndef TEXTNET_LAYER_PAIRHINGELOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_PAIRHINGELOSS_LAYER_INL_HPP_

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
class PairHingeLossLayer : public Layer<xpu>{
 public:
  PairHingeLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~PairHingeLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["delta"] = SettingV(1.0f);
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
                  "PairHingeLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PairHingeLossLayer:top size problem.");
    delta = setting["delta"].f_val;
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "PairHingeLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PairHingeLossLayer:top size problem.");
    nbatch = bottom[0]->data.size(0);    
    utils::Check(nbatch % 2 == 0, "nBatch must be even.");              
    top[0]->Resize(1, 1, 1, 1, true);
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();
    
    for (int i = 0; i < nbatch; i += 2) {
      utils::Check(bottom1_data[i] == 1 && bottom1_data[i+1] == 0, 
                    "Instances come like 1 0 1 0 ...");
      top_data[0] += std::max(0.0f, delta + bottom0_data[i+1] - bottom0_data[i]);
    }
    
    top_data[0] /= (nbatch/2);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom0_diff = bottom[0]->diff_d1();
    
    if (this->prop_error[0]) {
      for (int i = 0; i < nbatch; i+=2) {
        float gate = (delta + bottom0_data[i+1] - bottom0_data[i]) > 0 ? 1 : 0;
        bottom0_diff[i] = -gate;
        bottom0_diff[i+1] = gate;
      }
    }
  }
  
 protected:
  int nbatch;
  float delta;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_PAIRHINGELOSS_LAYER_INL_HPP_

