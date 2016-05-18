#ifndef TEXTNET_LAYER_LISTHINGELOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_LISTHINGELOSS_LAYER_INL_HPP_

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
class ListHingeLossLayer : public Layer<xpu>{
 public:
  ListHingeLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~ListHingeLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["delta"] = SettingV(1.0f);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["list_size"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ListHingeLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ListHingeLossLayer:top size problem.");
    delta = setting["delta"].fVal();
    list_size = setting["list_size"].iVal();
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ListHingeLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ListHingeLossLayer:top size problem.");
    nbatch = bottom[0]->data.size(0);    
    utils::Check(nbatch % list_size == 0, "ListHingeLossLayer:nBatch must be mod up list_size. Batch size: %d.", nbatch);              
    top[0]->Resize(1, 1, 1, 1, true);
    if (show_info) {
        top[0]->PrintShape("top0");
    }
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    nbatch = bottom[0]->data.size(0);
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();
    
    for (int p_idx = 0; p_idx < nbatch; p_idx += list_size) {
      for (int n_idx = p_idx+1; n_idx < p_idx + list_size; ++n_idx) {
        utils::Check(bottom1_data[p_idx] > bottom1_data[n_idx], 
                    "Instances come like x > y");
        top_data[0] += std::max(0.0f, delta + bottom0_data[n_idx] - bottom0_data[p_idx]);
      }
    }
    
    top_data[0] /= (list_size - 1) * (nbatch / list_size);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom0_diff = bottom[0]->diff_d1();
    
    if (this->prop_error[0]) {
      float factor = (list_size - 1) * (nbatch / list_size);
      for (int p_idx = 0; p_idx < nbatch; p_idx += list_size) {
        for (int n_idx = p_idx+1; n_idx < p_idx + list_size; ++n_idx) {
          float gate = (delta + bottom0_data[n_idx] - bottom0_data[p_idx]) > 0 ? 1 : 0;
          bottom0_diff[p_idx] += -gate / factor;
          bottom0_diff[n_idx] +=  gate / factor;
        }
      }
    }

  }
  
 protected:
  int nbatch;
  float delta;
  int list_size;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LISTHINGELOSS_LAYER_INL_HPP_

