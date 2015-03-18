#ifndef TEXTNET_LAYER_CROSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_CROSS_LAYER_INL_HPP_

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
class CrossLayer : public Layer<xpu>{
 public:
  CrossLayer(LayerType type) { this->layer_type = type; }
  virtual ~CrossLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "CrossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "CrossLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    doc_len = bottom[0]->data.size(3);    
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "CrossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "CrossLayer:top size problem.");
                  
    top[0]->Resize(nbatch, 1, doc_len, doc_len, true);
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 3> top_data = top[0]->data_d3();
    
    for (int i = 0; i < nbatch; i++) {
      top_data[i] = broadcast<0>(bottom0_data[i]) + 
                    broadcast<1>(bottom1_data[i]);
    }
    
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_diff = bottom[0]->diff_d2();
    mshadow::Tensor<xpu, 2> bottom1_diff = bottom[1]->diff_d2();
    mshadow::Tensor<xpu, 3> top_diff = top[0]->diff_d3();
    
    for (int i = 0; i < nbatch; i++) {
      bottom0_diff[i] = sum_all_except<1>(bottom0_data[i]); 
      bottom1_diff[i] = sum_all_except<0>(bottom1_data[i]);
    }
  }
  
 protected:
  int doc_len;
  int nbatch;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CROSS_LAYER_INL_HPP_

