#ifndef TEXTNET_LAYER_BATCH_MAX_LAYER_INL_HPP_
#define TEXTNET_LAYER_BATCH_MAX_LAYER_INL_HPP_

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
class BatchMaxLayer : public Layer<xpu>{
 public:
  BatchMaxLayer(LayerType type) { this->layer_type = type; }
  virtual ~BatchMaxLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["step"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    step = setting["step"].iVal();

    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchMaxLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchMaxLayer:top size problem.");

  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchMaxLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchMaxLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    utils::Check(bottom[0]->data.size(1) == 1 && bottom[0]->data.size(2) == 1 && bottom[0]->data.size(3) == 1, 
                 "BatchMaxLayer: only support one element.");
    utils::Check(nbatch % step == 0, 
                  "BatchMaxLayer: nbatch div step.");
    out_nbatch = nbatch / step;
                  
    top[0]->Resize(out_nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), bottom[0]->data.size(3), out_nbatch, bottom[0]->length.size(1), true);
    mask = vector<int>(out_nbatch);

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
    mshadow::Tensor<xpu, 1> bottom_data1 = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> top_data1 = top[0]->data_d1();
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;

    for (int i = 0, j = 0; i < nbatch; i += step, j += 1) {
      float max_val = -FLT_MAX; 
      int max_idx = -1;
      for (int k = 0; k < step; ++k) {
        if (max_val < bottom_data1[i+k]) {
          max_val = bottom_data1[i+k];
          max_idx = i+k;
        }
        top_data1[j] = max_val;
        mask[j] = max_idx;
        top_len[j] = F<op::identity>(bottom_len[max_idx]);
      }
    }

  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom_diff1 = bottom[0]->diff_d1();
    mshadow::Tensor<xpu, 1> top_diff1 = top[0]->diff_d1();

    for (int j = 0; j < out_nbatch; ++j) {
        bottom_diff1[mask[j]] += top_diff1[j];
    }    

  }
  
 protected:
  int nbatch;
  int out_nbatch;
  int step;
  vector<int> mask;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_BATCH_MAX_LAYER_INL_HPP_

