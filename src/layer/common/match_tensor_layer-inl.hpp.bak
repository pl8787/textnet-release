#ifndef TEXTNET_LAYER_MATCH_TENSOR_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_TENSOR_LAYER_INL_HPP_

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
class MatchTensorLayer : public Layer<xpu>{
 public:
  MatchTensorLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchTensorLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 3; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["interval"] = SettingV(1); 
    this->defaults["is_var_len"] = SettingV(true); 
    this->defaults["is_use_linear"] = SettingV(true); 
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["d_hidden"] = SettingV();

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
    
    utils::Check(bottom.size() == BottomNodeNum(), "MatchTensorLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchTensorLayer:top size problem.");

    d_hidden = setting["d_hidden"].iVal();
    is_var_len = setting["is_var_len"].bVal();
    is_use_linear = setting["is_use_linear"].bVal();
    interval = setting["interval"].iVal();
	feat_size = bottom[0]->data.size(3);

    this->params.resize(3);
    this->params[0].Resize(   d_hidden, feat_size, feat_size, 1, true); // t
    this->params[1].Resize(2*feat_size,  d_hidden,         1, 1, true); // w
    this->params[2].Resize(   d_hidden,         1,         1, 1, true); // b

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
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "MatchTensorLayer:bottom size problem."); 
    utils::Check(top.size()    == TopNodeNum(),    "MatchTensorLayer:top size problem.");
                  
    batch_size = bottom[0]->data.size(0); 
    doc_len = bottom[0]->data.size(2);
                  
    left_product.Resize(batch_size, doc_len, d_hidden, feat_size, true);
    bottom_0_transform_linear.Resize(batch_size, 1, doc_len, d_hidden, true);
    bottom_1_transform_linear.Resize(batch_size, 1, doc_len, d_hidden, true);
    top[0]->Resize(batch_size, d_hidden, doc_len, doc_len, true);

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
    if (batch_size != bottom[0]->data.size(0)) {
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
    Tensor4D bottom0_data    = bottom[0]->data;
    Tensor4D bottom1_data    = bottom[1]->data;
    Tensor2D bottom0_data_d2 = bottom[0]->data_d2_reverse();
    Tensor2D bottom1_data_d2 = bottom[1]->data_d2_reverse();
	Tensor1D bottom0_len     = bottom[0]->length_d1();
	Tensor1D bottom1_len     = bottom[1]->length_d1();
    Tensor4D top_data        = top[0]->data;
    Tensor3D t_data          = this->params[0].data_d3();

	top_data = 0.f, left_product = 0.f;

    if (is_use_linear) {
      Tensor2D w_data = this->params[1].data_d2();
      bottom_0_transform_linear.data_d2_reverse() = dot(bottom0_data_d2, w_data.Slice(0,feat_size));
      bottom_1_transform_linear.data_d2_reverse() = dot(bottom1_data_d2, w_data.Slice(feat_size, 2*feat_size));
    }

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
        Tensor2D rep_0 = bottom0_data[batch_idx][0].Slice(i,i+1);
        for (int k = 0; k < d_hidden; ++k) {
          Tensor2D rep_trans = left_product.data[batch_idx][i].Slice(k, k+1);
          rep_trans = dot(rep_0, t_data[k]);
          for (int j = 0; j < len_1; j+=interval) {
            Tensor2D rep_1 = bottom1_data[batch_idx][0].Slice(j,j+1);
            // top_data[batch_idx][k].Slice(i,i+1).Slice(j,j+1) += dot(rep_trans, rep_1.T());
            for (int f = 0; f < feat_size; ++f) {
              top_data[batch_idx][k][i][j] += rep_trans[0][f]*rep_1[0][f];
            }
            if (is_use_linear) {
              top_data[batch_idx][k][i][j] += bottom_0_transform_linear.data[batch_idx][0][i][k];
              top_data[batch_idx][k][i][j] += bottom_1_transform_linear.data[batch_idx][0][j][k];
            }
          }
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_data        = top[0]->data;
    Tensor4D top_diff        = top[0]->diff;
	Tensor4D bottom0_data    = bottom[0]->data;
	Tensor4D bottom1_data    = bottom[1]->data;
	Tensor4D bottom0_diff    = bottom[0]->diff;
	Tensor4D bottom1_diff    = bottom[1]->diff;
    Tensor2D bottom0_data_d2 = bottom[0]->data_d2_reverse();
    Tensor2D bottom1_data_d2 = bottom[1]->data_d2_reverse();
    Tensor2D bottom0_diff_d2 = bottom[0]->diff_d2_reverse();
    Tensor2D bottom1_diff_d2 = bottom[1]->diff_d2_reverse();
	Tensor1D bottom0_len     = bottom[0]->length_d1();
	Tensor1D bottom1_len     = bottom[1]->length_d1();
    Tensor3D t_data          = this->params[0].data_d3();
    Tensor3D t_diff          = this->params[0].diff_d3();

    left_product.diff = 0.f;
    bottom_0_transform_linear.diff = 0.f, bottom_1_transform_linear.diff = 0.f;

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
        Tensor2D rep_0_data = bottom0_data[batch_idx][0].Slice(i,i+1);
        Tensor2D rep_0_diff = bottom0_diff[batch_idx][0].Slice(i,i+1);
        for (int k = 0; k < d_hidden; ++k) {
          Tensor2D rep_trans_data = left_product.data[batch_idx][i].Slice(k, k+1);
          Tensor2D rep_trans_diff = left_product.diff[batch_idx][i].Slice(k, k+1);
          for (int j = 0; j < len_1; j+=interval) {
            if (is_use_linear) {
              bottom_0_transform_linear.diff[batch_idx][0][i][k] += top_diff[batch_idx][k][i][j];
              bottom_1_transform_linear.diff[batch_idx][0][j][k] += top_diff[batch_idx][k][i][j];
            }

            Tensor2D rep_1_data = bottom1_data[batch_idx][0].Slice(j,j+1);
            Tensor2D rep_1_diff = bottom1_diff[batch_idx][0].Slice(j,j+1);
            rep_1_diff += top_diff[batch_idx][k][i][j] * rep_trans_data;
            rep_trans_diff += top_diff[batch_idx][k][i][j] * rep_1_data;
          }
          t_diff[k] += dot(rep_0_data.T(), rep_trans_diff);
          rep_0_diff += dot(rep_trans_diff, t_data[k].T());
        }
      }
    }
    if (is_use_linear) {
      Tensor2D w_data = this->params[1].data_d2();
      Tensor2D w_diff = this->params[1].diff_d2();
      w_diff.Slice(0,feat_size) += dot(bottom0_data_d2.T(), bottom_0_transform_linear.diff_d2_reverse());
      w_diff.Slice(feat_size,2*feat_size) += dot(bottom1_data_d2.T(), bottom_1_transform_linear.diff_d2_reverse());
      bottom0_diff_d2 += dot(bottom_0_transform_linear.diff_d2_reverse(), w_data.Slice(0,feat_size).T());
      bottom1_diff_d2 += dot(bottom_1_transform_linear.diff_d2_reverse(), w_data.Slice(feat_size, 2*feat_size).T());
    }
  }
  
 protected:
  int doc_len, feat_size, batch_size, interval, d_hidden;
  bool is_var_len, is_use_linear;
  Node<xpu> left_product;
  Node<xpu> bottom_0_transform_linear, bottom_1_transform_linear; // this is for w in tensor layer
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MATCH_LAYER_INL_HPP_

