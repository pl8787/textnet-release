#ifndef TEXTNET_LAYER_ACCURACY_LAYER_INL_HPP_
#define TEXTNET_LAYER_ACCURACY_LAYER_INL_HPP_

#include <algorithm>
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
class AccuracyLayer : public Layer<xpu>{
 public:
  AccuracyLayer(LayerType type) { this->layer_type = type; }
  virtual ~AccuracyLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["topk"] = SettingV(1);
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
                  "AccuracyLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "AccuracyLayer:top size problem.");
       
    topk = setting["topk"].iVal();
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "AccuracyLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "AccuracyLayer:top size problem.");
    nbatch = bottom[0]->data.size(0);  
    ncategory = bottom[0]->data.size(1);              
    top[0]->Resize(1, 1, 1, 1, true);
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 2> bottom1_len  = bottom[1]->length;
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();

	top_data[0] = 0.0f;
    
    for (int i = 0; i < nbatch; ++i) {
      std::vector<std::pair<float, int> > bottom_data_vector;
      for (int j = 0; j < ncategory; ++j) {
        bottom_data_vector.push_back(std::make_pair(
            bottom0_data[i][j], j));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + topk,
          bottom_data_vector.end(), std::greater<std::pair<float, int> >());
      for (int j = 0; j < topk; ++j) {
        int len = bottom1_len[i][0];
        if (len == -1) len = bottom1_data.size(1); // support multi label, however, softmax layer does not support
        for (int k = 0; k < len; ++k) {
          int lable_value = static_cast<int>(bottom1_data[i][k]);
          if (bottom_data_vector[j].second == lable_value) {
            ++top_data[0];
            break;
          }
        }
      }
    }
    top_data[0] /= (float)(nbatch);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

  }
  
 protected:
  int nbatch;
  int ncategory;
  int topk;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_HINGELOSS_LAYER_INL_HPP_

