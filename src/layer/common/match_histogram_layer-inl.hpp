#ifndef TEXTNET_LAYER_MATCH_HISTOGRAM_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_HISTOGRAM_LAYER_INL_HPP_

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
class MatchHistogramLayer : public Layer<xpu>{
 public:
  MatchHistogramLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchHistogramLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["hist_size"] = SettingV(3); 
    this->defaults["base_val"] = SettingV(1.f);
    this->defaults["axis"] = SettingV(); 
    // it will force custom to set in config
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MatchHistogramLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MatchHistogramLayer:top size problem.");
    hist_size  = setting["hist_size"].iVal();
    base_val  = setting["base_val"].fVal();
    axis = setting["axis"].iVal();
    utils::Check( 1 <= axis && axis <= 3,
                  "MatchHistogramLayer:axis should be 1|2|3.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MatchHistogramLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MatchHistogramLayer:top size problem.");
                  
    int nbatch = bottom[0]->data.size(0); 
    if(axis == 1){
      top[0]->Resize(nbatch, hist_size, bottom[0]->data.size(2), bottom[0]->data.size(3), nbatch, bottom[0]->length.size(1), true);
    }else if(axis == 2){
      top[0]->Resize(nbatch, bottom[0]->data.size(1), hist_size, bottom[0]->data.size(3), nbatch, bottom[0]->length.size(1), true);
    }else if(axis == 3){
      utils::Check(bottom[0]->length.size(1) == 2, "MatchHistogramLayer: bottom length should be equal to 2.");
      top[0]->Resize(nbatch, bottom[0]->data.size(1), bottom[0]->data.size(2), hist_size, nbatch, bottom[0]->length.size(1), true);
    }

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (top[0]->data.size(0) != bottom[0]->data.size(0) ) {
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
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
	mshadow::Tensor<xpu, 2> top_len = top[0]->length;

    top_data = 0.0f;
    if(axis == 1){
      for(int i = 0 ; i < bottom[0]->data.size(0); ++ i){
        for(int j = 0 ; j < bottom[0]->data.size(1); ++ j){
          for(int k = 0 ; k < bottom[0]->data.size(2); ++ k){
            for(int l = 0 ; l < bottom[0]->data.size(3); ++ l){
                int idx = int(((bottom[0]->data[i][j][k][l] + 1.0) / 2.0 ) * (hist_size - 1));
                top[0]->data[i][idx][k][l] += base_val;
            }
          }
        }
      }
      top_len = F<op::identity>(bottom_len);
    }else if(axis == 2){
      for(int i = 0 ; i < bottom[0]->data.size(0); ++ i){
        for(int j = 0 ; j < bottom[0]->data.size(1); ++ j){
          for(int k = 0 ; k < bottom[0]->data.size(2); ++ k){
            for(int l = 0 ; l < bottom[0]->data.size(3); ++ l){
                int idx = int(((bottom[0]->data[i][j][k][l] + 1.0) / 2.0 ) * (hist_size - 1));
                top[0]->data[i][j][idx][l] += base_val;
            }
          }
        }
      }
      top_len = F<op::identity>(bottom_len);
    }else if(axis == 3){
      /*
      mshadow::Tensor<xpu,2> bottom_data2 = bottom[0]->data_d2_reverse();
      mshadow::Tensor<xpu,2> top_data2 = top[0]->data_d2_reverse();
      for(int i = 0 ; i < bottom_data2.size(0); ++ i){
        for(int j = 0 ; j < bottom_data2.size(1); ++ j){
          int idx = int(((bottom_data2[i][j] + 1.0) / 2.0 ) * (hist_size - 1));
          top_data2[i][idx] += base_val;
        }
        top_data2[i] /= bottom_data2.size(1); //distribution
        //for(int j = 0 ; j < hist_size; ++ j){ // rescale
          //top_data2[i][j] = log10(1.0 + top_data2[i][j]);
        //}
      }
      top_len = F<op::identity>(bottom_len);
      */
      for(int i = 0 ; i < bottom[0]->data.size(0); ++ i){
        for(int j = 0 ; j < bottom[0]->data.size(1); ++ j){
          for(int k = 0 ; k < bottom[0]->data.size(2); ++ k){
            int doc_len = bottom_len[i][1];
            utils::Check(doc_len <= bottom[0]->data.size(3),"doc-len:%d less than size:%d.", doc_len, bottom[0]->data.size(3));
            for(int l = 0 ; l < bottom[0]->data.size(3); ++ l){
              int idx = int(((bottom[0]->data[i][j][k][l] + 1.0) / 2.0 ) * (hist_size - 1));
              top[0]->data[i][j][k][idx] += base_val;
            }
            for(int l = 0 ; l < hist_size; ++ l){
              top[0]->data[i][j][k][l] = log10(1 + top[0]->data[i][j][k][l]);
            }
            top[0]->length[i][0] = bottom[0]->length[i][0];
            top[0]->length[i][1] = hist_size;
          }
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {

    if (!this->prop_error[0]) return;
      
    using namespace mshadow::expr;
    // this layer is a counting layer, there is no gradient.
  }
  
 protected:
  float base_val;
  int axis;
  int hist_size;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MATCH_LAYER_INL_HPP_

