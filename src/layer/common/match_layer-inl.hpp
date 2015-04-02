#ifndef TEXTNET_LAYER_MATCH_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_LAYER_INL_HPP_

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
class MatchLayer : public Layer<xpu>{
 public:
  MatchLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
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
                  "MatchLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MatchLayer:top size problem.");
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MatchLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MatchLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    doc_len = bottom[0]->data.size(3);    
                  
    top[0]->Resize(nbatch, 1, doc_len, doc_len, true);
	  bottom[0]->PrintShape("bottom0");
	  top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    
    for (int i = 0; i < nbatch; i++) {
      for (int j = 0; j < doc_len; j++) {
        for (int k = 0; k < doc_len; k++) {
          if (bottom0_data[i][j]==-1 || bottom1_data[i][k]==-1) {
            top_data[i][0][j][k] = 0;
          } else {
            top_data[i][0][j][k] = (bottom0_data[i][j] == bottom1_data[i][k]) ? 1 : 0;
          }
        }
      }
    }
    
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    
  }
  
 protected:
  int doc_len;
  int nbatch;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MATCH_LAYER_INL_HPP_

