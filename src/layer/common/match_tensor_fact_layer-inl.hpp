#ifndef TEXTNET_LAYER_MATCH_TENSOR_FACT_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_TENSOR_FACT_LAYER_INL_HPP_

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
class MatchTensorFactLayer : public Layer<xpu>{
 public:
  MatchTensorFactLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchTensorFactLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 3; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["interval"] = SettingV(1); 
    this->defaults["is_var_len"] = SettingV(true); 
    this->defaults["is_init_as_I"] = SettingV(true);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["d_hidden"] = SettingV();
    this->defaults["d_factor"] = SettingV();
    this->defaults["t_l2"] = SettingV();

    this->defaults["t_filler"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["t_updater"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "MatchTensorFactLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchTensorFactLayer:top size problem.");

    d_hidden = setting["d_hidden"].iVal();
    d_factor = setting["d_factor"].iVal();
    is_var_len = setting["is_var_len"].bVal();
    is_init_as_I = setting["is_init_as_I"].bVal();
    interval = setting["interval"].iVal();
    t_l2 = setting["t_l2"].fVal();
	feat_size = bottom[0]->data.size(3);

    diag_4_reg.Resize(feat_size, d_hidden, d_factor, 1, true);
    this->params.resize(3);
    this->params[0].Resize(feat_size, d_hidden, d_factor, 1, true); // t
    this->params[1].Resize(feat_size, d_hidden, 1,        1, true); // w
    this->params[2].Resize(d_hidden, 1,         1,        1, true); // b

    std::map<std::string, SettingV> &t_setting = *setting["t_filler"].mVal();
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(t_setting["init_type"].iVal(), t_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(), w_setting, this->prnd_);
    this->params[2].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    this->params[2].Init();
    if (is_init_as_I) {
      InitAsDiag();
    }

    std::map<std::string, SettingV> &t_updater = *setting["t_updater"].mVal();
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = updater::CreateUpdater<xpu, 4>(t_updater["updater_type"].iVal(),
                                                              t_updater, this->prnd_);
    this->params[1].updater_ = updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
                                                              w_updater, this->prnd_);
    this->params[2].updater_ = updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
                                                              b_updater, this->prnd_);
  }

  // init t nearly with I
  void InitAsDiag(void) {
    utils::Printf("Init As Diag.");
    utils::Check(d_factor == feat_size, "MatchTensorFactLayer: init as diag error.");
    for (int i = 0; i < d_factor; ++i) {
      for (int j = 0; j < d_hidden; ++j) {
        this->params[0].data[i][j][i][0] = 1.f;
        diag_4_reg.data[i][j][i][0] = 1.f;
      }
    }
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "MatchTensorFactLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchTensorFactLayer:top size problem.");
                  
    batch_size = bottom[0]->data.size(0); 
    doc_len = bottom[0]->data.size(2);
                  
    bottom_0_transform.Resize(batch_size, doc_len, d_hidden, d_factor);
    bottom_1_transform.Resize(batch_size, doc_len, d_hidden, d_factor);
    top[0]->Resize(batch_size, d_hidden, doc_len, doc_len, true);

    bottom[0]->PrintShape("bottom0");
    bottom[1]->PrintShape("bottom1");
    top[0]->PrintShape("top0");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
	Tensor1D bottom0_len = bottom[0]->length_d1();
	Tensor1D bottom1_len = bottom[1]->length_d1();
    Tensor4D top_data = top[0]->data;

	top_data = 0.f;

    Tensor2D bottom0_data_d2 = bottom[0]->data_d2_reverse();
    Tensor2D bottom1_data_d2 = bottom[1]->data_d2_reverse();
    Tensor2D t_data = this->params[0].data_d2();

    // without consider var len
    bottom_0_transform.data_d2_middle() = dot(bottom0_data_d2, t_data);
    bottom_1_transform.data_d2_middle() = dot(bottom1_data_d2, t_data);

    // compute dot
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int len_0 = -1, len_1 = -1;
      if (is_var_len) {
        len_0 = bottom0_len[batch_idx];
        len_1 = bottom1_len[batch_idx];
      } else {
        len_0 = doc_len;
        len_1 = doc_len;
      }
      for (int i = 0; i < len_0; i+=interval) {
        for (int j = 0; j < len_1; j+=interval) {
          for (int d = 0; d < d_hidden; ++d) {
            for (int f = 0; f < d_factor; ++f) {
              top_data[batch_idx][d][i][j] += bottom_0_transform.data[batch_idx][i][d][f] * \
                                              bottom_1_transform.data[batch_idx][j][d][f];
            }
          }
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_diff = top[0]->diff;
	Tensor1D bottom0_len = bottom[0]->length_d1();
	Tensor1D bottom1_len = bottom[1]->length_d1();

    bottom_0_transform.diff = 0.f, bottom_1_transform.diff = 0.f;
    // compute dot
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int len_0 = -1, len_1 = -1;
      if (is_var_len) {
        len_0 = bottom0_len[batch_idx];
        len_1 = bottom1_len[batch_idx];
      } else {
        len_0 = doc_len;
        len_1 = doc_len;
      }
      for (int i = 0; i < len_0; i+=interval) {
        for (int j = 0; j < len_1; j+=interval) {
          for (int d = 0; d < d_hidden; ++d) {
            for (int f = 0; f < d_factor; ++f) {
              bottom_0_transform.diff[batch_idx][i][d][f] += top_diff[batch_idx][d][i][j] * \
                                                             bottom_1_transform.data[batch_idx][j][d][f];
              bottom_1_transform.diff[batch_idx][j][d][f] += top_diff[batch_idx][d][i][j] * \
                                                             bottom_0_transform.data[batch_idx][i][d][f];
            }
          }
        }
      }
    }

    Tensor2D bottom0_data_d2 = bottom[0]->data_d2_reverse();
    Tensor2D bottom0_diff_d2 = bottom[0]->diff_d2_reverse();
    Tensor2D bottom1_data_d2 = bottom[1]->data_d2_reverse();
    Tensor2D bottom1_diff_d2 = bottom[1]->diff_d2_reverse();
    Tensor2D t_data = this->params[0].data_d2();
    Tensor2D t_diff = this->params[0].diff_d2();
    t_diff += dot(bottom0_data_d2.T(), bottom_0_transform.diff_d2_middle());
    t_diff += dot(bottom1_data_d2.T(), bottom_1_transform.diff_d2_middle());
    bottom0_diff_d2 += dot(bottom_0_transform.diff_d2_middle(), t_data.T());
    bottom1_diff_d2 += dot(bottom_1_transform.diff_d2_middle(), t_data.T());

    // l2 by diag_4_reg
    if (t_l2 > 0.) {
      this->params[0].diff += (this->params[0].data - diag_4_reg.data) * t_l2;
    }
  }
  
 protected:
  int doc_len, feat_size, batch_size, interval, d_hidden, d_factor;
  bool is_var_len, is_init_as_I;
  float t_l2;
  Node<xpu> bottom_0_transform, bottom_1_transform, diag_4_reg; // tensor layer is essentially a transform layer followed by a dot producttion
};
}  // namespace layer
}  // namespace textnet
#endif  

