#ifndef TEXTNET_LAYER_MATCH_COMBINE_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_COMBINE_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include <algorithm>

namespace textnet {
namespace layer {

template<typename xpu>
class MatchCombineLayer : public Layer<xpu>{
 public:
  MatchCombineLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchCombineLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; } 
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    // this->defaults["dim"] = SettingV(2);
    // this->defaults["is_constraint"] = SettingV(false);

    // require value, set to SettingV(),
    // it will force custom to set in config
    //this->defaults["k"] = SettingV();

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    //k = setting["k"].iVal();
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "MatchCombineLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchCombineLayer: top size problem.");
    utils::Check(bottom[0]->data.shape_ == bottom[1]->data.shape_, "MatchCombineLayer: bottom 0 size should be equal to bottom 1 size.");

    mshadow::Shape<4> shape_out  = bottom[0]->data.shape_;
    mshadow::Shape<2> shape_length = bottom[0]->length.shape_;
    top[0]->Resize(shape_out, shape_length, true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        bottom[1]->PrintShape("bottom1");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;

    if (top[0]->data.shape_[0] != bottom[0]->data.shape_[0]) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }
 
    
  typedef mshadow::Tensor<xpu,2> Tensor2D;
  typedef mshadow::Tensor<xpu,2,int> Tensor2DInt;

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom0_data  = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom1_data  = bottom[1]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    top[0]->length = F<op::identity>(bottom[0]->length);

    top_data = 0;
    for (index_t batch_idx = 0; batch_idx < bottom0_data.size(0); ++batch_idx) {
      for (index_t channel_idx = 0; channel_idx < bottom0_data.size(1); ++channel_idx) {
        for( index_t i = 0 ; i < bottom0_data.size(2); ++ i){
          for( index_t j = 0 ; j < bottom0_data.size(3); ++ j){
            if( abs(bottom1_data[batch_idx][channel_idx][i][j] - 1.0) < 1e-4 ){
              top_data[batch_idx][channel_idx][i][j] = bottom1_data[batch_idx][channel_idx][i][j];
            }else{
              top_data[batch_idx][channel_idx][i][j] = bottom0_data[batch_idx][channel_idx][i][j];
            }
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
};
}  // namespace layer
}  // namespace textnet
#endif  
