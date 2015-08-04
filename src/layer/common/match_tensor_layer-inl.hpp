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
    interval = setting["interval"].iVal();
	feat_size = bottom[0]->data.size(3);

    this->params.resize(3);
    this->params[0].Resize(d_hidden, feat_size,   feat_size, 1, true); // t
    this->params[1].Resize(d_hidden, 2*feat_size, 1,         1, true); // w
    this->params[2].Resize(d_hidden, 1,           1,         1, true); // b

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
    utils::Check(top.size() == TopNodeNum(), "MatchTensorLayer:top size problem.");
                  
    batch_size = bottom[0]->data.size(0); 
    doc_len = bottom[0]->data.size(2);
                  
    out_product.Resize(batch_size*doc_len*doc_len, feat_size, feat_size, 1, true);
    // top_data_swap.Resize(batch_size, doc_len, doc_len, d_hidden, true);
    top[0]->Resize(batch_size, d_hidden, doc_len, doc_len, true);

	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		bottom[1]->PrintShape("bottom1");
		top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D bottom0_data = bottom[0]->data;
    Tensor4D bottom1_data = bottom[1]->data;
	Tensor1D bottom0_len = bottom[0]->length_d1();
	Tensor1D bottom1_len = bottom[1]->length_d1();
    Tensor4D top_data = top[0]->data;

	top_data = 0.f, out_product.data = 0.f; // top_data_swap.data = 0.f;

    // comput for out_product for every pair
    Tensor3D out_prod = out_product.data_d3();
    idxes.clear();
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
        for (int j = 0; j < len_1; j+=interval) {
          Tensor2D rep_1 = bottom1_data[batch_idx][0].Slice(j,j+1);
          int idx = batch_idx*doc_len*doc_len + i*doc_len + j;
          out_prod[idx] = dot(rep_0.T(), rep_1);
          idxes.push_back(idx);
        }
      }
    }
    out_product_dense.Resize(idxes.size(), feat_size, feat_size, 1, true);
    Tensor3D out_prod_dense = out_product_dense.data_d3();
    for (size_t i = 0; i < idxes.size(); ++i) {
        out_prod_dense[i] = F<op::identity>(out_prod[idxes[i]]);
    }

    // product with tensor
    Tensor2D t = this->params[0].data_d2();
    Tensor2D input = out_product_dense.data_d2();
    top_data_dense.Resize(1, 1, idxes.size(), d_hidden, true);
    top_data_dense.data_d2_reverse() += dot(input, t.T());

    // need to swap axis
    for (size_t i = 0; i < idxes.size(); ++i) {
      int idx = idxes[i];
      int batch_idx = idx / (doc_len*doc_len);
      int pos = idx % (doc_len*doc_len);
      int row = pos / doc_len; 
      int col = pos % doc_len;

      for (int f = 0; f < d_hidden; ++f) {
        top_data[batch_idx][f][row][col] = top_data_dense.data[0][0][i][f];
      }
    }

    // top_data_swap.data_d2_reverse() += dot(input, t.T());

    // swap axis 1->2, 2->3, 3->1
    // for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    //   for (int i = 0; i < doc_len; ++i) {
    //     for (int j = 0; j < doc_len; ++j) {
    //       for (int f = 0; f < d_hidden; ++f) {
    //         top_data[batch_idx][f][i][j] = top_data_swap.data[batch_idx][i][j][f];
    //       }
    //     }
    //   }
    // }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_diff = top[0]->diff;
    Tensor4D top_data = top[0]->data;
	Tensor4D bottom0_data = bottom[0]->data;
	Tensor4D bottom1_data = bottom[1]->data;
	Tensor4D bottom0_diff = bottom[0]->diff;
	Tensor4D bottom1_diff = bottom[1]->diff;
	Tensor1D bottom0_len = bottom[0]->length_d1();
	Tensor1D bottom1_len = bottom[1]->length_d1();

    top_data_dense.diff = 0.f, out_product.diff = 0.f, out_product_dense.diff = 0.f;

    // need to swap axis
    for (size_t i = 0; i < idxes.size(); ++i) {
      int idx = idxes[i];
      int batch_idx = idx / (doc_len*doc_len);
      int pos = idx % (doc_len*doc_len);
      int row = pos / doc_len; 
      int col = pos % doc_len;

      for (int f = 0; f < d_hidden; ++f) {
        top_data_dense.diff[0][0][i][f] = top_diff[batch_idx][f][row][col];
      }
    }

    // for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    //   for (int i = 0; i < doc_len; ++i) {
    //     for (int j = 0; j < doc_len; ++j) {
    //       for (int f = 0; f < d_hidden; ++f) {
    //         top_data_swap.diff[batch_idx][i][j][f] = top_diff[batch_idx][f][i][j];
    //       }
    //     }
    //   }
    // }

    // Tensor3D out_prod_dense = out_product_dense.diff_d3();
    this->params[0].diff_d2() += dot(top_data_dense.diff_d2_reverse().T(), out_product_dense.data_d2());
    out_product_dense.diff_d2() = dot(top_data_dense.diff_d2_reverse(), this->params[0].data_d2());

         
    Tensor3D out_prod = out_product.diff_d3();
    Tensor3D out_prod_dense = out_product_dense.diff_d3();
    for (size_t i = 0; i < idxes.size(); ++i) {
      out_prod[idxes[i]] = F<op::identity>(out_prod_dense[i]);
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
        Tensor2D rep_data_0 = bottom0_data[batch_idx][0].Slice(i,i+1);
        Tensor2D rep_diff_0 = bottom0_diff[batch_idx][0].Slice(i,i+1);
        for (int j = 0; j < len_1; j+=interval) {
          Tensor2D rep_data_1 = bottom1_data[batch_idx][0].Slice(j,j+1);
          Tensor2D rep_diff_1 = bottom1_diff[batch_idx][0].Slice(j,j+1);

          int idx = batch_idx*doc_len*doc_len + i*doc_len + j;
          rep_diff_1 += dot(rep_data_0, out_prod[idx]);
          rep_diff_0 += dot(rep_data_1, out_prod[idx].T());
        }
      }
    }
  }
  
 protected:
  int doc_len, feat_size, batch_size, interval, d_hidden;
  bool is_var_len;
  Node<xpu> out_product, out_product_dense, top_data_dense;
  vector<int> idxes; // out_product maybe sparse, this is the index of non 0 values
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MATCH_LAYER_INL_HPP_

