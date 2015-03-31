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
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "CrossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "CrossLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    channel = bottom[0]->data.size(1);
    doc_len = bottom[0]->data.size(2);    
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "CrossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "CrossLayer:top size problem.");
                  
    top[0]->Resize(nbatch, channel, doc_len, doc_len, true);

	bottom[0]->PrintShape("bottom0");
	bottom[1]->PrintShape("bottom1");
	top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> bottom0_data = bottom[0]->data_d3();
    mshadow::Tensor<xpu, 3> bottom1_data = bottom[1]->data_d3();
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    
    for (int i = 0; i < nbatch; i++) {
        for (int j = 0; j < channel; j++) {
        top_data[i][j] = broadcast<0>(bottom0_data[i][j], top_data[i][j].shape_) + 
                         broadcast<1>(bottom1_data[i][j], top_data[i][j].shape_);
      }
    }
    
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> bottom0_diff = bottom[0]->diff_d3();
    mshadow::Tensor<xpu, 3> bottom1_diff = bottom[1]->diff_d3();
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    
    for (int i = 0; i < nbatch; i++) {
      for (int j = 0; j < channel; j++) {
        bottom0_diff[i][j] = sumall_except_dim<0>(top_diff[i][j]); 
        bottom1_diff[i][j] = sumall_except_dim<1>(top_diff[i][j]);
      }
    }
  }
  
 protected:
  int doc_len;
  int nbatch;
  int channel;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CROSS_LAYER_INL_HPP_

