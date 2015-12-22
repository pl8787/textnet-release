#ifndef TEXTNET_LAYER_GATE_WHOLE_POOLING_LAYER_INL_HPP_
#define TEXTNET_LAYER_GATE_WHOLE_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {


// this layer can be composed by other layers (convolution, softmax, product)
// however, 2D version can not, thus, we rewrite this 1D layer,
// and then upgrade to 2D version

// this is a 1D gate average whole pooling layer
// input size (batch_size, sentence_num, length, dim_rep)
// output size (batch_size, sentence_num, 1, dim_rep)
// the output representation for a sentence is a weighted
// sum of all positional sentence representations
// the weights are normalized by softmax
template<typename xpu>
class GateWholePoolingLayer : public Layer<xpu> {
 public:
  GateWholePoolingLayer(LayerType type) { this->layer_type = type; }
  virtual ~GateWholePoolingLayer(void) {}
  
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "GateWholePoolingLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GateWholePoolingLayer:top size problem.");
                            
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
    utils::Check(bottom.size() == BottomNodeNum(), "GateWholePoolingLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GateWholePoolingLayer:top size problem.");
    
    int batch_size = bottom[0]->data.size(0);
    int num_seq    = bottom[0]->data.size(1);
    int max_length = bottom[0]->data.size(2);
    int feat_size  = bottom[0]->data.size(3);
    
    top[0]->Resize(batch_size, num_seq, 1, feat_size, true);
    gate_score.Resize(batch_size, num_seq, 1, max_length, true); 
    gate_prob.Resize(batch_size, num_seq, 1, max_length, true); 

	if (show_info) {
	  bottom[0]->PrintShape("bottom0");
	  top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_data_d2 = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;

    mshadow::Tensor<xpu, 4> gate_score_data    = gate_score.data;
    mshadow::Tensor<xpu, 4> gate_prob_data     = gate_prob.data;
    mshadow::Tensor<xpu, 2> gate_score_data_d2 = gate_score.data_d2_reverse();

    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    mshadow::Tensor<xpu, 2> w_data      = this->params[0].data_d2();
    mshadow::Tensor<xpu, 1> b_data      = this->params[1].data_d1();

    top_data = 0.f, gate_score_data = 0.f, gate_prob_data = 0.f;
    
    gate_score_data_d2 = dot(bottom_data_d2,  w_data);
    if (!no_bias) {
      gate_score_data_d2 += repmat(b_data, gate_score_data_d2.size(0));
    }

    for (int i = 0; i < bottom_data.size(0); ++i) {
      for (int j = 0; j < bottom_data.size(1); ++j) {
        int len = bottom_data.size(2);
        if (is_var_len) {
          utils::Check(bottom_len[i][j] > 0, "GateWholePoolingLayer:length should be unset.");
          len = bottom_len[i][j];
        } else {
          utils::Check(bottom_len[i][j] == -1, "GateWholePoolingLayer:length should be unset.");
        }
        mshadow::Tensor<xpu, 1> gate_score_seq = gate_score_data[i][j][0].Slice(0, len);
        mshadow::Tensor<xpu, 1> gate_prob_seq  = gate_prob_data[i][j][0].Slice(0, len);
        mshadow::Softmax(gate_score_seq, gate_prob_seq);
        // for (int i = 0; i < output.size(0); ++i) {
        //   if (output[i] < crop) {
        //     std::cout << "SoftmaxFuncVarLenLayer: WARNING, prob too small, crop." << std::endl;
        //     output[i] = crop;
        //   }
        // }

        for (int m = 0; m < len; ++m) {
          for (int n = 0; n < dim_rep; ++n) {
            top_data[i][j][0][n] += bottom_data[i][j][m][n] * gate_prob_seq[m];
          }
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff          = top[0]->diff;
    mshadow::Tensor<xpu, 4> gate_score_diff   = gate_score.diff;
    mshadow::Tensor<xpu, 4> gate_prob_data    = gate_prob.data;
    mshadow::Tensor<xpu, 4> gate_prob_diff    = gate_prob.diff;
    mshadow::Tensor<xpu, 2> gate_score_diff_d2= gate_score.diff_d2_reverse();

    mshadow::Tensor<xpu, 4> bottom_data    = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff    = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> bottom_data_d2 = bottom[0]->data_d2_reverse();

    mshadow::Tensor<xpu, 2> w_data         = this->params[0].data_d2();
    mshadow::Tensor<xpu, 2> w_diff         = this->params[0].data_d2();
    mshadow::Tensor<xpu, 1> b_diff         = this->params[1].data_d1();

    gate_score_diff = 0.f, gate_prob_diff = 0.f;

    for (int i = 0; i < bottom_data.size(0); ++i) {
      for (int j = 0; j < bottom_data.size(1); ++j) {
        int len = bottom_data.size(2);
        if (is_var_len) {
          len = bottom_len[i][j];
        } 
        for (int m = 0; m < len; ++m) {
          for (int n = 0; n < dim_rep; ++n) {
            bottom_diff[i][j][m][n] += top_diff[i][j][0][n] * gate_prob_data[i][j][0][m];
            gate_prob_diff[i][j][0][m] += top_diff[i][j][0][n] * bottom_data[i][j][m][n];
          }
        }
        
        // diff softmax
        mshadow::Tensor<xpu,1> sm_out_data = gate_prob_data[i][j][0].Slice(0, len);
        mshadow::Tensor<xpu,1> sm_in_diff  = gate_score_diff[i][j][0].Slice(0, len);
        mshadow::Tensor<xpu,1> sm_out_diff = gate_prob_diff[i][j][0].Slice(0, len);
        for (int col_idx = 0; col_idx < len; ++col_idx) {
          for (int jacobi_row_idx = 0; jacobi_row_idx < len; ++jacobi_row_idx) {
            float top = sm_out_diff[jacobi_row_idx];
            if (top == 0.f) continue;
            float p_0 = sm_out_data[col_idx];
            float p_1 = sm_out_data[jacobi_row_idx];
            if (jacobi_row_idx == col_idx) {
              sm_in_diff[col_idx] += top * (-(p_0*p_0) + p_0);
            } else {
              sm_in_diff[col_idx] += top * (-(p_0*p_1));
            }
          }
        }
      }
    }

    w_diff += dot(bottom_data_d2.T(), gate_score_diff_d2);
    bottom_diff += dot(gate_score_diff_d2, w_data.T());
    if (!no_bias) {
      b_diff += sum_rows(gate_score_diff_d2);
    }
  }

 protected:
  /*! \brief random number generator */
  int dim_rep;
  bool no_bias, is_var_len;
  Node<xpu> gate_score, gate_prob; // gate_score is before softmax, gate_prob is after softmax
};
}  // namespace layer
}  // namespace textnet
#endif
