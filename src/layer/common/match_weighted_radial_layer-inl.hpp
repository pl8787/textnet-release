#ifndef TEXTNET_LAYER_MATCH_WEIGHTED_RADIAL_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_WEIGHTED_RADIAL_LAYER_INL_HPP_

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
class MatchWeightedRadialLayer : public Layer<xpu>{
 public:
  MatchWeightedRadialLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchWeightedRadialLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 1; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["interval"] = SettingV(1); 
    this->defaults["is_var_len"] = SettingV(true); 
    
    // require value, set to SettingV(),
    // it will force custom to set in config

    this->defaults["w_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;

  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "MatchWeightedRadialLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchWeightedRadialLayer:top size problem.");

    is_var_len = setting["is_var_len"].bVal();
    interval = setting["interval"].iVal();
	  feat_size = bottom[0]->data.size(3);
    this->params.resize(1);
    this->params[0].Resize(feat_size, 1, 1, 1, true); 

    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(), w_setting, this->prnd_);
    this->params[0].Init();

    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    this->params[0].updater_ = updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
                                                              w_updater, this->prnd_);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "MatchWeightedRadialLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchWeightedRadialLayer:top size problem.");
    utils::Check(bottom[0]->data.size(0) == bottom[1]->data.size(0) &&
        bottom[0]->data.size(3) == bottom[1]->data.size(3), "MatchWeightedRadialLayer:batch_size or feat_size problem.");
                  
    batch_size = bottom[0]->data.size(0); 
    doc_len0 = bottom[0]->data.size(2);
    doc_len1 = bottom[1]->data.size(2);
                  
    top[0]->Resize(batch_size, 1, doc_len0, doc_len1, batch_size, 2, true);

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
    if (top[0]->data.size(0) != bottom[0]->data.size(0)) {
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
    Tensor4D bottom0_data = bottom[0]->data;
    Tensor4D bottom1_data = bottom[1]->data;
	  Tensor1D bottom0_len  = bottom[0]->length_d1();
    Tensor1D bottom1_len  = bottom[1]->length_d1();
    Tensor4D top_data     = top[0]->data;
	  Tensor2D top_len      = top[0]->length;
    Tensor1D w_data       = this->params[0].data_d1();

	  top_data = 0.f;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int len_0 = -1, len_1 = -1;
      if (is_var_len) {
        len_0 = bottom0_len[batch_idx];
        len_1 = bottom1_len[batch_idx];
		    top_len[batch_idx][0] = len_0;
		    top_len[batch_idx][1] = len_1;
      } else {
        len_0 = doc_len0;
        len_1 = doc_len1;
      }
      for (int i = 0; i < len_0; i+=interval) {
        Tensor1D rep_0 = bottom0_data[batch_idx][0][i];
        for (int j = 0; j < len_1; j+=interval) {
          Tensor1D rep_1 = bottom1_data[batch_idx][0][j];

          float sigma = 0.f;
          float sub_elem_square = 0.f;
          for(int f = 0 ; f < feat_size; ++ f){
            sigma += w_data[f] * rep_0[f];
            float sub_elem = rep_0[f] - rep_1[f];
            sub_elem_square += sub_elem * sub_elem;
          }
          sigma *= sigma; 
          top_data[batch_idx][0][i][j] = - sub_elem_square / sigma / 4.f;
        }
      }
    }
    top_data = F<op::exp_lookup>(top_data);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_diff     = top[0]->diff;
    Tensor4D top_data     = top[0]->data;
    Tensor4D bottom0_data = bottom[0]->data;
    Tensor4D bottom1_data = bottom[1]->data;
    Tensor4D bottom0_diff = bottom[0]->diff;
    Tensor4D bottom1_diff = bottom[1]->diff;
    Tensor1D bottom0_len  = bottom[0]->length_d1();
    Tensor1D bottom1_len  = bottom[1]->length_d1();
    Tensor1D w_data       = this->params[0].data_d1();
    Tensor1D w_diff       = this->params[0].diff_d1();

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int len_0 = -1, len_1 = -1;
      if (is_var_len) {
        len_0 = bottom0_len[batch_idx];
        len_1 = bottom1_len[batch_idx];
      } else {
        len_0 = doc_len0;
        len_1 = doc_len1;
      }
      for (int i = 0; i < len_0; i+=interval) {
        Tensor1D rep_data_0 = bottom0_data[batch_idx][0][i];
        Tensor1D rep_diff_0 = bottom0_diff[batch_idx][0][i];
        for (int j = 0; j < len_1; j+=interval) {
          Tensor1D rep_data_1 = bottom1_data[batch_idx][0][j];
          Tensor1D rep_diff_1 = bottom1_diff[batch_idx][0][j];

          float sigma = 0.f;
          float sub_elem_square = 0.f;
          float simi = top_data[batch_idx][0][i][j];
          for(int f = 0 ; f < feat_size; ++ f){
            sigma += w_data[f] * rep_data_0[f];
            float sub_elem = rep_data_0[f] - rep_data_1[f];
            sub_elem_square += sub_elem * sub_elem;
          }

          for(int f = 0 ; f < feat_size; ++ f){
            w_diff[f] += top_diff[batch_idx][0][i][j] * simi * sub_elem_square / 2.f / sigma / sigma / sigma * rep_data_0[f];
            rep_diff_0[f] += top_diff[batch_idx][0][i][j] * simi / sigma / sigma / 2.f * (sub_elem_square / sigma * w_data[f] - (rep_data_0[f] - rep_data_1[f]));
            rep_diff_1[f] += top_diff[batch_idx][0][i][j] * simi / sigma / sigma / 2.f * ( rep_data_0[f] - rep_data_1[f]);
          }
        }
      }
    }
  }
  
 protected:
  int doc_len0, doc_len1, feat_size, batch_size, interval;
  bool is_var_len;
  // Node<xpu> out_product, top_data_swap;
};
}  // namespace layer
}  // namespace textnet
#endif 

