#ifndef TEXTNET_LAYER_DIAG_RECURRENT_LAYER_INL_HPP_
#define TEXTNET_LAYER_DIAG_RECURRENT_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"
#include <cassert>

namespace textnet {
namespace layer {

// The input  format is (batch_size, height, width, input_feat_num).
// The output format is (batch_size, height, width, output_feat_num).
// For that textnet doesnot support 2D variable length, 
// this layer need another 2 nodes to give the length.
template<typename xpu>
class DiagRecurrentLayer : public Layer<xpu> {
 public:
  DiagRecurrentLayer(LayerType type) { this->layer_type = type; }
  virtual ~DiagRecurrentLayer(void) {}
  
  virtual int BottomNodeNum() { return 3; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 3; }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);
    this->defaults["reverse"] = SettingV(false);
    this->defaults["input_transform"] = SettingV(true);
    this->defaults["nonlinear_type"] = SettingV("tanh"); // other options "sigmoid", "rectifier"
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["d_mem"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["u_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["u_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(), "DiagRecurrentLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "DiagRecurrentLayer:top size problem.");
                  
    d_mem   = setting["d_mem"].iVal();
    d_input = bottom[0]->data.size(3);
    no_bias = setting["no_bias"].bVal();
    reverse = setting["reverse"].bVal();
    input_transform = setting["input_transform"].bVal();
    nonlinear_type = setting["nonlinear_type"].sVal();
    if (!input_transform) {
        utils::Check(d_input == d_mem, "DiagRecurrentLayer:input does not match with memory, need transform.");
    }

    begin_h.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_h_er.Resize(mshadow::Shape2(1, d_mem), 0.f);

    this->params.resize(3);
    this->params[0].Resize(d_input, d_mem, 1, 1, true); // w
    this->params[1].Resize(d_mem,   d_mem, 1, 1, true); // u
    this->params[2].Resize(d_mem,       1, 1, 1, true); // b
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &u_setting = *setting["u_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(), w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(u_setting["init_type"].iVal(), u_setting, this->prnd_);
    this->params[2].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    this->params[2].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &u_updater = *setting["u_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();

    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(), w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(u_updater["updater_type"].iVal(), u_updater, this->prnd_);
    this->params[2].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(), b_updater, this->prnd_);
  }
  
  // bottom should be padded with only one zero on both sides
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "DiagRecurrentLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "DiagRecurrentLayer:top size problem.");
    
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);
    top[0]->Resize(shape_out, true);

	bottom[0]->PrintShape("bottom0");
	bottom[1]->PrintShape("bottom1");
	bottom[2]->PrintShape("bottom2");
	top[0]->PrintShape("top0");
  }

  void checkNan(float *p, int l) {
      for (int i = 0; i < l; ++i) {
          assert(!isnan(p[i]));
      }
  }

  void checkNanParams() {
      Tensor2D w_data = this->params[0].data[0][0];
      Tensor2D u_data = this->params[1].data[0][0];
      Tensor2D w_diff = this->params[0].diff[0][0];
      Tensor2D u_diff = this->params[1].diff[0][0];
      checkNan(w_data.dptr_, w_data.size(0) * w_data.size(1));
      checkNan(u_data.dptr_, u_data.size(0) * u_data.size(1));
      checkNan(w_diff.dptr_, w_diff.size(0) * w_diff.size(1));
      checkNan(u_diff.dptr_, u_diff.size(0) * u_diff.size(1));
  }

  virtual void ForwardOneStep(Tensor2D pre_h,
                              Tensor2D x,
                              Tensor2D cur_h) {
    Tensor2D w_data = this->params[0].data_d2();
    Tensor2D u_data = this->params[1].data_d2();
    Tensor1D b_data = this->params[2].data_d1();

    cur_h = dot(pre_h, u_data);
    if (input_transform) {
      cur_h += dot(x, w_data);
    } else {
      cur_h += x;
    }
    if (!no_bias) {
      cur_h += repmat(b_data, 1);
    }
    if (nonlinear_type == "sigmoid") {
      cur_h = mshadow::expr::F<op::sigmoid>(cur_h); // sigmoid_grad
    } else if (nonlinear_type == "tanh") {         
      cur_h = mshadow::expr::F<op::tanh>(cur_h);    // tanh_grad
    } else if (nonlinear_type == "rectifier") {
      cur_h = mshadow::expr::F<op::relu>(cur_h);    // relu_grad
    } else {
      utils::Check(false, "DiagRecurrentLayer:nonlinear type error.");
    }
  }
  // the tensor must be sliced for var len 
  void ForwardLeftTop2RightBottom(Tensor3D in, Tensor3D out, 
                                  int begin_row, int begin_col, 
                                  int max_row, int max_col) {
    utils::Check(begin_row == 0 || begin_col == 0, "DiagRecurrentLayer: ff input error.");
    utils::Check(begin_row < max_row && begin_col < max_col, "DiagRecurrentLayer: ff input error.");
    utils::Check(out.size(0) == in.size(0) && out.size(1) == in.size(1), "DiagRecurrentLayer: ff input error.");
    utils::Check(max_row <= in.size(0) && max_col <= in.size(1), "DiagRecurrentLayer: ff input error.");
    
    Tensor2D pre_h;
    // not need any padding, begin h and c are set to 0
    for (index_t row_idx = begin_row, col_idx = begin_col; 
         row_idx < max_row && col_idx < max_col;
         ++row_idx, ++col_idx) {
      if (row_idx == 0 || col_idx == 0) {
        pre_h = begin_h;
      } else {
        pre_h = out[row_idx-1].Slice(col_idx-1, col_idx);
      }
      ForwardOneStep(pre_h,
                     in[row_idx].Slice(col_idx, col_idx+1),
                     out[row_idx].Slice(col_idx, col_idx+1));
    }
  }
  // the tensor must be sliced for var len 
  void ForwardRightBottom2LeftTop(Tensor3D in, Tensor3D out, 
                                  int begin_row, int begin_col, 
                                  int max_row, int max_col) {
    utils::Check(begin_row == max_row-1 || begin_col == max_col-1, "DiagRecurrentLayer: ff input error.");
    utils::Check(begin_row < max_row && begin_col < max_col, "DiagRecurrentLayer: ff input error.");
    utils::Check(out.size(0) == in.size(0) && out.size(1) == in.size(1), "DiagRecurrentLayer: ff input error.");
    utils::Check(max_row <= in.size(0) && max_col <= in.size(1), "DiagRecurrentLayer: ff input error.");
    
    Tensor2D pre_h;
    // not need any padding, begin h and c are set to 0
    for (int row_idx = begin_row, col_idx = begin_col; 
         row_idx >= 0 && col_idx >= 0;
         --row_idx, --col_idx) {
      if (row_idx == max_row-1 || col_idx == max_col-1) {
        pre_h = begin_h;
      } else {
        pre_h = out[row_idx+1].Slice(col_idx+1, col_idx+2);
      }
      ForwardOneStep(pre_h,
                     in[row_idx].Slice(col_idx, col_idx+1),
                     out[row_idx].Slice(col_idx, col_idx+1));
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    // checkNanParams();
    Tensor4D bottom_data = bottom[0]->data;
    Tensor4D top_data = top[0]->data;
    Tensor2D l_sen_len = bottom[1]->length;
    Tensor2D r_sen_len = bottom[2]->length;

    top_data = 0.f;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      int l_len = l_sen_len[batch_idx][0];
      int r_len = r_sen_len[batch_idx][0];
      utils::Assert(l_len >= 0 && r_len >= 0, "DiagRecurrentLayer: sequence length error.");
      if (!reverse) {
        for (index_t begin_col = 0; begin_col < r_len; ++begin_col) {
          ForwardLeftTop2RightBottom(bottom_data[batch_idx],
                                     top_data[batch_idx],
                                     0, begin_col,
                                     l_len, r_len);
        }
        for (index_t begin_row = 1; begin_row < l_len; ++begin_row) {
          ForwardLeftTop2RightBottom(bottom_data[batch_idx],
                                     top_data[batch_idx],
                                     begin_row, 0,
                                     l_len, r_len);
        }
      } else {
        for (index_t begin_col = 0; begin_col < r_len; ++begin_col) {
          ForwardRightBottom2LeftTop(bottom_data[batch_idx],
                                     top_data[batch_idx],
                                     l_len-1, begin_col,
                                     l_len, r_len);
        }
        for (index_t begin_row = 0; begin_row < l_len-1; ++begin_row) {
          ForwardRightBottom2LeftTop(bottom_data[batch_idx],
                                     top_data[batch_idx],
                                     begin_row, r_len-1,
                                     l_len, r_len);
        }
      }
    }
    // checkNanParams();
  }
  void BpOneStep(Tensor2D cur_h_er,
                 Tensor2D cur_h,
                 Tensor2D pre_h,
                 Tensor2D x,
                 Tensor2D pre_h_er,
                 Tensor2D x_er) {
    Tensor2D w_er = this->params[0].diff_d2();
    Tensor2D u_er = this->params[1].diff_d2();
    Tensor1D b_er = this->params[2].diff_d1();

    Tensor2D w_data = this->params[0].data_d2();
    Tensor2D u_data = this->params[1].data_d2();

    if (nonlinear_type == "sigmoid") {
      cur_h_er *= mshadow::expr::F<op::sigmoid_grad>(cur_h); // sigmoid_grad
    } else if (nonlinear_type == "tanh") {         
      cur_h_er *= mshadow::expr::F<op::tanh_grad>(cur_h);    // tanh_grad
    } else if (nonlinear_type == "rectifier") {
      cur_h_er *= mshadow::expr::F<op::relu_grad>(cur_h);    // relu_grad
    } else {
      utils::Check(false, "DiagRecurrentLayer:nonlinear type error.");
    }

    pre_h_er += dot(cur_h_er, u_data.T());
    u_er += dot(pre_h.T(), cur_h_er);

    if (input_transform) {
      x_er += dot(cur_h_er, w_data.T());
      w_er += dot(x.T(), cur_h_er); 
    } else {
      x_er += cur_h_er;
    }
    if (!no_bias) {
      b_er += sum_rows(cur_h_er);
    }
  }

  void BackpropForLeftTop2RightBottomRnn(Tensor3D top_data, Tensor3D top_diff, 
                                         Tensor3D bottom_data, Tensor3D bottom_diff,
                                         int begin_row, int begin_col,
                                         int max_row, int max_col) {
    utils::Check(begin_row == 0 || begin_col == 0, "DiagRecurrentLayer: bp input error.");
    utils::Check(begin_row < max_row && begin_col < max_col, "DiagRecurrentLayer: bp input error.");
    utils::Check(max_row <= top_data.size(0) && max_col <= top_data.size(1), "DiagRecurrentLayer: bp input error.");

    int step = -1;
    if ((max_row-begin_row) > (max_col-begin_col)) {
      step = max_col-begin_col;
    } else {
      step = max_row-begin_row;
    }
    int end_row_idx = begin_row + step - 1; 
    int end_col_idx = begin_col + step - 1;

    Tensor2D pre_h, pre_h_er;
    for (int row_idx = end_row_idx, col_idx = end_col_idx; 
         row_idx >= 0 && col_idx >= 0; 
         --row_idx, --col_idx) {
      if (row_idx == 0 || col_idx == 0) {
          pre_h = begin_h;
          pre_h_er = begin_h_er;
      } else {
          pre_h = top_data[row_idx-1].Slice(col_idx-1, col_idx);
          pre_h_er = top_diff[row_idx-1].Slice(col_idx-1, col_idx);
      }
      BpOneStep(top_diff[row_idx].Slice(col_idx, col_idx+1), 
                top_data[row_idx].Slice(col_idx, col_idx+1),
                pre_h,
                bottom_data[row_idx].Slice(col_idx, col_idx+1), 
                pre_h_er,
                bottom_diff[row_idx].Slice(col_idx, col_idx+1));
    }
  }
  void BackpropForRightBottom2LeftTopRnn(Tensor3D top_data, Tensor3D top_diff, 
                                         Tensor3D bottom_data, Tensor3D bottom_diff,
                                         int begin_row, int begin_col,
                                         int max_row, int max_col) {
    utils::Check(begin_row == max_row-1 || begin_col == max_col-1, "DiagRecurrentLayer: bp input error.");
    utils::Check(begin_row < max_row && begin_col < max_col, "DiagRecurrentLayer: bp input error.");
    utils::Check(max_row <= top_data.size(0) && max_col <= top_data.size(1), "DiagRecurrentLayer: bp input error.");

    int step = -1;
    if (begin_row > begin_col) {
      step = begin_col+1;
    } else {
      step = begin_row+1;
    }
    int end_row_idx = begin_row - step + 1; 
    int end_col_idx = begin_col - step + 1;
    utils::Check(end_row_idx == 0 || end_col_idx == 0, "DiagRecurrentLayer: bp input error.");
    utils::Check(end_row_idx >= 0 && end_col_idx >= 0, "DiagRecurrentLayer: bp input error.");

    Tensor2D pre_h, pre_h_er;
    for (int row_idx = end_row_idx, col_idx = end_col_idx; 
         row_idx <= begin_row && col_idx <= begin_col;
         ++row_idx, ++col_idx) {
      if (row_idx == begin_row || col_idx == begin_col) {
          pre_h = begin_h;
          pre_h_er = begin_h_er;
      } else {
          pre_h = top_data[row_idx+1].Slice(col_idx+1, col_idx+2);
          pre_h_er = top_diff[row_idx+1].Slice(col_idx+1, col_idx+2);
      }
      BpOneStep(top_diff[row_idx].Slice(col_idx, col_idx+1), 
                top_data[row_idx].Slice(col_idx, col_idx+1),
                pre_h,
                bottom_data[row_idx].Slice(col_idx, col_idx+1), 
                pre_h_er,
                bottom_diff[row_idx].Slice(col_idx, col_idx+1));
    }
  }

  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    // checkNanParams();
    Tensor4D top_diff = top[0]->diff;
    Tensor4D top_data = top[0]->data;
    Tensor4D bottom_data = bottom[0]->data;
    Tensor4D bottom_diff = bottom[0]->diff;
    Tensor2D l_sen_len = bottom[1]->length;
    Tensor2D r_sen_len = bottom[2]->length;

    begin_h_er = 0.; 
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      int l_len = l_sen_len[batch_idx][0];
      int r_len = r_sen_len[batch_idx][0];
      utils::Assert(l_len >= 0 && r_len >= 0, "DiagRecurrentLayer: sequence length error.");
      if (!reverse) {
        for (index_t begin_col = 0; begin_col < r_len; ++begin_col) {
          BackpropForLeftTop2RightBottomRnn(top_data[batch_idx],
                                            top_diff[batch_idx],
                                            bottom_data[batch_idx],
                                            bottom_diff[batch_idx],
                                            0, begin_col,
                                            l_len, r_len);
        }
        for (index_t begin_row = 1; begin_row < l_len; ++begin_row) {
          BackpropForLeftTop2RightBottomRnn(top_data[batch_idx],
                                            top_diff[batch_idx],
                                            bottom_data[batch_idx],
                                            bottom_diff[batch_idx],
                                            begin_row, 0,
                                            l_len, r_len);
        }
      } else {
        for (index_t begin_col = 0; begin_col < r_len; ++begin_col) {
          BackpropForRightBottom2LeftTopRnn(top_data[batch_idx],
                                            top_diff[batch_idx],
                                            bottom_data[batch_idx],
                                            bottom_diff[batch_idx],
                                            l_len-1, begin_col,
                                            l_len, r_len);
        }
        for (index_t begin_row = 0; begin_row < l_len-1; ++begin_row) {
          BackpropForRightBottom2LeftTopRnn(top_data[batch_idx],
                                            top_diff[batch_idx],
                                            bottom_data[batch_idx],
                                            bottom_diff[batch_idx],
                                            begin_row, r_len-1,
                                            l_len, r_len);
        }
      }
    }
  }

 protected:
  int d_mem, d_input;
  bool no_bias, reverse, input_transform; 
  mshadow::TensorContainer<xpu, 2> begin_h, begin_h_er;
  std::string nonlinear_type;
};
}  // namespace layer
}  // namespace textnet
#endif
