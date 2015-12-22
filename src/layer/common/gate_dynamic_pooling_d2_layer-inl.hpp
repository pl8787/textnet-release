#ifndef TEXTNET_LAYER_GATE_DYNAMIC_POOLING_D2_LAYER_INL_HPP_
#define TEXTNET_LAYER_GATE_DYNAMIC_POOLING_D2_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {


// this is a 2D gate average whole pooling layer
// input size (batch_size, x_length, y_length, dim_rep)
// output size (batch_size, row, col, dim_rep)
// the output representation is a weighted
// sum of all input representations in a sub matrix
// the weights are normalized by softmax
// NOTE: the axises are different with DynamicPoolingLayer
template<typename xpu>
class GateDynamicPoolingD2Layer : public Layer<xpu> {
 public:
  GateDynamicPoolingD2Layer(LayerType type) { this->layer_type = type; }
  virtual ~GateDynamicPoolingD2Layer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["row"] = SettingV();
    this->defaults["col"] = SettingV();
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "GateDynamicPoolingD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GateDynamicPoolingD2Layer:top size problem.");
                            
    no_bias = setting["no_bias"].bVal();
    row = setting["row"].iVal();
    col = setting["col"].iVal();
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
    utils::Check(bottom.size() == BottomNodeNum(), "GateDynamicPoolingD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GateDynamicPoolingD2Layer:top size problem.");
    
    int batch_size = bottom[0]->data.size(0);
    int x_max_len  = bottom[0]->data.size(1);
    int y_max_len  = bottom[0]->data.size(2);
    int feat_size  = bottom[0]->data.size(3);
    
    top[0]->Resize(batch_size, row, col, feat_size, true);
    gate_score.Resize(batch_size,  x_max_len, y_max_len, 1, true); 
    // gate_score_consec.Resize(batch_size,  x_max_len*y_max_len, 1, 1, true); 
    gate_prob.Resize(batch_size, x_max_len, y_max_len, 1, true); 

	if (show_info) {
	  bottom[0]->PrintShape("bottom0");
	  top[0]->PrintShape("top0");
	}
  }

  // this functions return the split spans of each dynamic window
  // NOTE: if the input size is smaller than output size
  // this functions will return split spans larger than input size
  // and it is required to pad the input in other functions
  void dynamic_split(int input_row, int pool_row, vector<int> &pos) {
    utils::Check(input_row >= pool_row, "GateDynamicPoolingD2Layer: padding has not been implenmented yet.");
    pos.clear();
    int pad_input_row = input_row < pool_row ? pool_row : input_row;
    int margin = pad_input_row / pool_row;
    int mod    = pad_input_row % pool_row;
    pos.push_back(0);
    for (size_t i = 0; i < pool_row; ++i) {
      if (i < (pool_row-mod)) { 
        pos.push_back(pos[pos.size()-1]+margin);
      } else {
        pos.push_back(pos[pos.size()-1]+margin+1);
      }
    }
    
    utils::Check(pos[pos.size()-1] == pad_input_row, "GateDynamicPoolingD2Layer: split error.");
    
    for (size_t i = 1; i < pos.size(); ++i) {
      utils::Check(pos[i-1] < pos[i], "GateDynamicPoolingD2Layer: split error.");
      utils::Check((pos[i] - pos[i-1]) <= ((pad_input_row-1)/pool_row) + 1, "GateDynamicPoolingD2Layer: split error.");
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

    mshadow::Tensor<xpu, 4> bottom_data    = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_data_d2 = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom_len     = bottom[0]->length;

    mshadow::Tensor<xpu, 2> gate_score_data_d2 = gate_score.data_d2_reverse();

    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;
    mshadow::Tensor<xpu, 2> w_data      = this->params[0].data_d2();
    mshadow::Tensor<xpu, 1> b_data      = this->params[1].data_d1();

    top_data = 0.f, gate_score.data = 0.f, gate_prob.data = 0.f;
    
    gate_score_data_d2 = dot(bottom_data_d2, w_data);
    if (!no_bias) {
      gate_score_data_d2 += repmat(b_data, gate_score_data_d2.size(0));
    }

    // for softmax
    gate_prob.data = mshadow::expr::F<op::orc_exp>(gate_score.data);

    for (int i = 0; i < bottom_data.size(0); ++i) {
      int x_len = bottom_data.size(1);
      int y_len = bottom_data.size(2);
      if (is_var_len) {
        utils::Check(bottom_len[i][0] > 0, "GateDynamicPoolingD2Layer:length should be unset.");
        utils::Check(bottom_len[i][1] > 0, "GateDynamicPoolingD2Layer:length should be unset.");
        x_len = bottom_len[i][0];
        y_len = bottom_len[i][1];
      } else {
        utils::Check(bottom_len[i][0] == -1, "GateDynamicPoolingD2Layer:length should be unset.");
      }

      vector<int> begin_pos_row, begin_pos_col;
      dynamic_split(x_len, row, begin_pos_row);
      dynamic_split(y_len, col, begin_pos_col);
      
      // normalize softmax in a pooling sub matrix
      // then get the top data
      for (int r = 0; r < row; ++r) {
        for (int c = 0; c < col; ++c) {
          float sum = 0.f;
          for (int m = begin_pos_row[r]; m < begin_pos_row[r+1]; ++m) {
            for (int n = begin_pos_col[c]; n < begin_pos_col[c+1]; ++n) {
              sum += gate_prob.data[i][m][n][0];
            }
          }
          for (int m = begin_pos_row[r]; m < begin_pos_row[r+1]; ++m) {
            for (int n = begin_pos_col[c]; n < begin_pos_col[c+1]; ++n) {
              gate_prob.data[i][m][n][0] /= sum;
              for (int d = 0; d < dim_rep; ++d) {
                top_data[i][r][c][d] += bottom_data[i][m][n][d] * gate_prob.data[i][m][n][0];
              }
            }
          }
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

    gate_score.diff = 0.f, gate_prob.diff = 0.f;

    for (int i = 0; i < bottom_data.size(0); ++i) {
      int x_len = bottom_data.size(1);
      int y_len = bottom_data.size(2);
      if (is_var_len) {
        x_len = bottom_len[i][0];
        y_len = bottom_len[i][1];
      } 

      vector<int> begin_pos_row, begin_pos_col;
      dynamic_split(x_len, row, begin_pos_row);
      dynamic_split(y_len, col, begin_pos_col);

      for (int r = 0; r < row; ++r) {
        for (int c = 0; c < col; ++c) {
          // bp product
          for (int m = begin_pos_row[r]; m < begin_pos_row[r+1]; ++m) {
            for (int n = begin_pos_col[c]; n < begin_pos_col[c+1]; ++n) {
              for (int d = 0; d < dim_rep; ++d) {
                bottom_diff[i][m][n][d] += top_diff[i][r][c][d] * gate_prob.data[i][m][n][0];
                gate_prob.diff[i][m][n][0] += top_diff[i][r][c][d] * bottom_data[i][m][n][d];
              }
            }
          }
          float error_sum = 0.0f;
          // bp softmax
          for (int m = begin_pos_row[r]; m < begin_pos_row[r+1]; ++m) {
            for (int n = begin_pos_col[c]; n < begin_pos_col[c+1]; ++n) {
              error_sum += gate_prob.diff[i][m][n][0] * gate_prob.data[i][m][n][0];
            }
          }
          for (int m = begin_pos_row[r]; m < begin_pos_row[r+1]; ++m) {
            for (int n = begin_pos_col[c]; n < begin_pos_col[c+1]; ++n) {
              gate_score.diff[i][m][n][0] += (gate_prob.diff[i][m][n][0] - error_sum) * gate_prob.data[i][m][n][0];
            }
          }
        }
      }
    }

    w_diff += dot(bottom_data_d2.T(), gate_score_diff_d2);
    bottom_diff_d2 += dot(gate_score_diff_d2, w_data.T());
    if (!no_bias) {
      b_diff += sum_rows(gate_score_diff_d2);
    }
  }

 protected:
  /*! \brief random number generator */
  int dim_rep, row, col;
  bool no_bias, is_var_len;
  Node<xpu> gate_score, gate_prob; // gate_score is before softmax, gate_prob is after softmax
};
}  // namespace layer
}  // namespace textnet
#endif
