#ifndef TEXTNET_LAYER_GATE_WHOLE_POOLING_D2_LAYER_INL_HPP_
#define TEXTNET_LAYER_GATE_WHOLE_POOLING_D2_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {


// this is a 2D gate average whole pooling layer
// input size (batch_size, x_length, y_length, dim_rep)
// output size (batch_size, 1, 1, dim_rep)
// the output representation is a weighted
// sum of all input representations
// the weights are normalized by softmax
template<typename xpu>
class GateWholePoolingD2Layer : public Layer<xpu> {
 public:
  GateWholePoolingD2Layer(LayerType type) { this->layer_type = type; }
  virtual ~GateWholePoolingD2Layer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["is_var_len"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "GateWholePoolingD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GateWholePoolingD2Layer:top size problem.");
                            
    no_bias = setting["no_bias"].bVal();
    is_var_len = setting["is_var_len"].bVal();
    dim_rep = bottom[0]->data.size(3);

    this->params.resize(2);
    this->params[0].Resize(dim_rep, 1, 1, 1, true);
    this->params[1].Resize(1, 1, 1, 1, true);
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
                                                                          w_setting, this->prnd_);
    this->params[1].initializer_ = initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), 
                                                                          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
                                                              w_updater, this->prnd_);
    this->params[1].updater_ = updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
                                                              b_updater, this->prnd_);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "GateWholePoolingD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GateWholePoolingD2Layer:top size problem.");
    
    int batch_size = bottom[0]->data.size(0);
    int x_max_len  = bottom[0]->data.size(1);
    int y_max_len  = bottom[0]->data.size(2);
    int feat_size  = bottom[0]->data.size(3);
    
    top[0]->Resize(batch_size, 1, 1, feat_size, true);
    gate_score.Resize(batch_size,  x_max_len, y_max_len, 1, 1, true); 
    gate_score_consec.Resize(batch_size,  x_max_len*y_max_len, 1, 1, true); 
    gate_prob.Resize(batch_size, x_max_len*y_max_len, 1, 1, true); 

	if (show_info) {
	  bottom[0]->PrintShape("bottom0");
	  top[0]->PrintShape("top0");
	}
  }

  void MoveDataToConsec(mshadow::Tensor<xpu,2> bottom_len){
    for (int k = 0; k < bottom_len.size(0); ++k) {
      int cnt = 0;
      int x_len = bottom_len[k][0];
      int y_len = bottom_len[k][1];
      for (int i = 0; i < x_len; ++i) {
        for (int j = 0; j < y_len; ++j) {
          gate_score_consec.data[k][cnt][0][0] = gate_score.data[k][i][j][0];
          cnt += 1;
        }
      }
    }
  }

  void MoveDiffToUnConsec(mshadow::Tensor<xpu,2> bottom_len){
    for (int k = 0; k < bottom_len.size(0); ++k) {
      int cnt = 0;
      int x_len = bottom_len[k][0];
      int y_len = bottom_len[k][1];
      for (int i = 0; i < x_len; ++i) {
        for (int j = 0; j < y_len; ++j) {
          gate_score.diff[k][i][j][0] = gate_score_consec.diff[k][cnt][0][0];
          cnt += 1;
        }
      }
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_data_d2 = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;

    mshadow::Tensor<xpu, 2> gate_score_data_d2     = gate_score.data_d2_reverse();

    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    mshadow::Tensor<xpu, 2> w_data      = this->params[0].data_d2();
    mshadow::Tensor<xpu, 1> b_data      = this->params[1].data_d1();

    top_data = 0.f, gate_score.data = 0.f, gate_score_consec.data = 0.f, gate_prob.data = 0.f;
    
    gate_score_data_d2 = dot(bottom_data_d2,  w_data);
    if (!no_bias) {
      gate_score_data_d2 += repmat(b_data, gate_score_data_d2.size(0));
    }

    if (is_var_len) {
      MoveDataToConsec(bottom_len);
    } else {
      gate_score_consec.data_d1() = F<op::identity>(gate_score.data_d1());
    }

    for (int i = 0; i < bottom_data.size(0); ++i) {
      int x_len = bottom_data.size(1);
      int y_len = bottom_data.size(2);
      if (is_var_len) {
        utils::Check(bottom_len[i][0] > 0, "GateWholePoolingD2Layer:length should be unset.");
        utils::Check(bottom_len[i][1] > 0, "GateWholePoolingD2Layer:length should be unset.");
        x_len = bottom_len[i][0];
        y_len = bottom_len[i][1];
      } else {
        utils::Check(bottom_len[i][0] == -1, "GateWholePoolingD2Layer:length should be unset.");
      }

      mshadow::Tensor<xpu, 3> gate_score_seq = gate_score_consec.data[i].Slice(0, x_len*y_len);
      mshadow::Tensor<xpu, 3> gate_prob_seq  = gate_prob.data[i].Slice(0, x_len*y_len);
      mshadow::Tensor<xpu, 1> gate_score_seq_1d(gate_score_seq.dptr_, mshadow::Shape1(gate_score_seq.size(0)));
      mshadow::Tensor<xpu, 1> gate_prob_seq_1d(gate_prob_seq.dptr_, mshadow::Shape1(gate_prob_seq.size(0)));
      mshadow::Softmax(gate_prob_seq_1d, gate_score_seq_1d);
      // for (int i = 0; i < output.size(0); ++i) {
      //   if (output[i] < crop) {
      //     std::cout << "SoftmaxFuncVarLenLayer: WARNING, prob too small, crop." << std::endl;
      //     output[i] = crop;
      //   }
      // }

      int cnt = 0;
      for (int m = 0; m < x_len; ++m) {
        for (int n = 0; n < y_len; ++n) {
          for (int k = 0; k < dim_rep; ++k) {
            top_data[i][0][0][k] += bottom_data[i][m][n][k] * gate_prob_seq[cnt][0][0];
          }
          cnt += 1;
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff               = top[0]->diff;
    mshadow::Tensor<xpu, 2> gate_score_diff_d2     = gate_score.diff_d2_reverse();

    mshadow::Tensor<xpu, 4> bottom_data    = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff    = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> bottom_len     = bottom[0]->length;
    mshadow::Tensor<xpu, 2> bottom_data_d2 = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom_diff_d2 = bottom[0]->diff_d2_reverse();

    mshadow::Tensor<xpu, 2> w_data         = this->params[0].data_d2();
    mshadow::Tensor<xpu, 2> w_diff         = this->params[0].diff_d2();
    mshadow::Tensor<xpu, 1> b_diff         = this->params[1].diff_d1();

    gate_score.diff = 0.f, gate_score_consec.diff = 0.f, gate_prob.diff = 0.f;

    for (int i = 0; i < bottom_data.size(0); ++i) {
      int x_len = bottom_data.size(1);
      int y_len = bottom_data.size(2);
      if (is_var_len) {
        x_len = bottom_len[i][0];
        y_len = bottom_len[i][1];
      } 
      int cnt = 0;
      for (int m = 0; m < x_len; ++m) {
        for (int n = 0; n < y_len; ++n) {
          for (int k = 0; k < dim_rep; ++k) {
            bottom_diff[i][m][n][k] += top_diff[i][0][0][k] * gate_prob.data[i][cnt][0][0];
            gate_prob.diff[i][cnt][0][0] += top_diff[i][0][0][k] * bottom_data[i][m][n][k];
          }
          cnt += 1;
        }
      }
      
      // diff softmax
      mshadow::Tensor<xpu,3> sm_out_data = gate_prob.data[i].Slice(0, x_len*y_len);
      mshadow::Tensor<xpu,3> sm_in_diff  = gate_score_consec.diff[i].Slice(0, x_len*y_len);
      mshadow::Tensor<xpu,3> sm_out_diff = gate_prob.diff[i].Slice(0, x_len*y_len);
      for (int col_idx = 0; col_idx < x_len*y_len; ++col_idx) {
        for (int jacobi_row_idx = 0; jacobi_row_idx < x_len*y_len; ++jacobi_row_idx) {
          float top = sm_out_diff[jacobi_row_idx][0][0];
          if (top == 0.f) continue;
          float p_0 = sm_out_data[col_idx][0][0];
          float p_1 = sm_out_data[jacobi_row_idx][0][0];
          if (jacobi_row_idx == col_idx) {
            sm_in_diff[col_idx][0][0] += top * (-(p_0*p_0) + p_0);
          } else {
            sm_in_diff[col_idx][0][0] += top * (-(p_0*p_1));
          }
        }
      }
    }

    if (is_var_len) {
      MoveDiffToUnConsec(bottom_len);
    } else {
      gate_score.diff_d1() = F<op::identity>(gate_score_consec.diff_d1());
    }

    w_diff += dot(bottom_data_d2.T(), gate_score_diff_d2);
    bottom_diff_d2 += dot(gate_score_diff_d2, w_data.T());
    if (!no_bias) {
      b_diff += sum_rows(gate_score_diff_d2);
    }
  }

 protected:
  /*! \brief random number generator */
  int dim_rep;
  bool no_bias, is_var_len;
  Node<xpu> gate_score, gate_score_consec, gate_prob; // gate_score is before softmax, gate_prob is after softmax
};
}  // namespace layer
}  // namespace textnet
#endif
