#ifndef TEXTNET_LAYER_LSTM_D2_LAYER_INL_HPP_
#define TEXTNET_LAYER_LSTM_D2_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"
#include "../../io/json/json.h"
#include <ctime>
#include <chrono>
#include <cassert>

namespace textnet {
namespace layer {


using namespace std::chrono;

// this implement is different with (Graves 2009)
// we add a diagonal connnection 
template<typename xpu>
class LstmD2Layer : public Layer<xpu> {
 public:
  LstmD2Layer(LayerType type) { this->layer_type = type; }
  virtual ~LstmD2Layer(void) { }
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);
    // this->defaults["no_out_tanh"] = SettingV(false);
    // this->defaults["param_file"] = SettingV("");
    this->defaults["o_gate_bias_init"] = SettingV(0.f);
    this->defaults["f_gate_bias_init"] = SettingV(0.f);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    // this->defaults["max_norm2"] = SettingV();
    // this->defaults["grad_norm2"] = SettingV();
    this->defaults["d_mem"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    this->defaults["reverse"] = SettingV();
    // this->defaults["grad_cut_off"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(), "LstmD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LstmD2Layer:top size problem.");

    d_mem   = setting["d_mem"].iVal();
    d_input = bottom[0]->data.size(3);
    no_bias = setting["no_bias"].bVal();
    // no_out_tanh = setting["no_out_tanh"].bVal();
    reverse = setting["reverse"].bVal();
    // grad_norm2 = setting["grad_norm2"].fVal();
    // param_file = setting["param_file"].sVal();
    o_gate_bias_init = setting["o_gate_bias_init"].fVal();
    f_gate_bias_init = setting["f_gate_bias_init"].fVal();
    // grad_cut_off = setting["grad_cut_off"].fVal();
    // max_norm2 = setting["max_norm2"].fVal();

    begin_h.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_c.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_h_er.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_c_er.Resize(mshadow::Shape2(1, d_mem), 0.f);

    this->params.resize(2);
    this->params[0].Resize(1, 1, d_input+3*d_mem, 6*d_mem, true); // w and u is in one matrix
    this->params[1].Resize(1, 1, 1,               6*d_mem, true); // b
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(), w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    // if (f_gate_bias_init != 0.f) {
    //     init_f_gate_bias(); // this must be after init()
    // }
    // if (o_gate_bias_init != 0.f) {
    //     init_o_gate_bias(); // this must be after init()
    // }

    // if (!param_file.empty()) {
    //   LoadParam();
    // }
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();

    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(), w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(), b_updater, this->prnd_);
  }

  // if want to capture long term dependency, should init as a positive value
  // void init_f_gate_bias() {
  //   Tensor1D bias_data = this->params[2].data_d1();
  //   Tensor1D f_bias = Tensor1D(bias_data.dptr_ + 1*d_mem, mshadow::Shape1(d_mem));
  //   f_bias = f_gate_bias_init;
  // }

  // void init_o_gate_bias() {
  //   Tensor1D bias_data = this->params[2].data_d1();
  //   Tensor1D o_bias = Tensor1D(bias_data.dptr_ + 2*d_mem, mshadow::Shape1(d_mem));
  //   o_bias = o_gate_bias_init;
  // }
  
  // bottom should be padded with only one zero on both sides
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "LstmD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LstmD2Layer:top size problem.");
    
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    // (batch size, x_len, y_len, d_mem)
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);
    // input, forget * 3, output, candidate
    mshadow::Shape<4> shape_gate= mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem*6); 

    top[0]->Resize(shape_out, mshadow::Shape2(shape_out[0],2), true);
    c.Resize(shape_out, 0.f);
    g.Resize(shape_gate, 0.f);
    c_er.Resize(shape_out, 0.f);
    g_er.Resize(shape_gate, 0.f);

	if (show_info) {
      bottom[0]->PrintShape("bottom0");
	  top[0]->PrintShape("top0");
	}
  }

  void checkNan(float *p, int l) {
    for (int i = 0; i < l; ++i) {
      assert(!isnan(p[i]));
    }
  }

  void checkNanParams() {
    Tensor2D w_data = this->params[0].data_d2_reverse();
    Tensor2D w_diff = this->params[0].diff_d2_reverse();
    checkNan(w_data.dptr_, w_data.size());
    checkNan(w_diff.dptr_, w_diff.size());
  }

  void concat_input(Tensor2D &x, Tensor2D &h_l, Tensor2D &h_m, Tensor2D &h_t, Tensor2D &input) {
    utils::Check(x.size(0)  ==1 && h_l.size(0)==1 && h_m.size(0)==1 &&
                 h_t.size(0)==1 && input.size(0)  ==1, "LstmD2Layer: size error.");
    utils::Check(x.size(1)+h_l.size(1)+h_m.size(1)+h_t.size(1) == input.size(1), "LstmD2Layer: size error.");

    int cnt = 0;
    input[0].Slice(cnt, cnt+x.size(1))   = mshadow::expr::F<op::identity>(x[0]);
    cnt += x.size(1);
    input[0].Slice(cnt, cnt+h_l.size(1)) = mshadow::expr::F<op::identity>(h_l[0]);
    cnt += h_l.size(1);
    input[0].Slice(cnt, cnt+h_m.size(1)) = mshadow::expr::F<op::identity>(h_m[0]);
    cnt += h_m.size(1);
    input[0].Slice(cnt, cnt+h_t.size(1)) = mshadow::expr::F<op::identity>(h_t[0]);
  }

  // this is different with concat_input(), this re-use the same memory
  void split_input(Tensor2D t, Tensor2D &x, Tensor2D &h_l, Tensor2D &h_m, Tensor2D &h_t) {
    utils::Check(t.size(0)==1 && t.size(1)==(d_input+3*d_mem), "LstmD2Layer: size error.");

    x   = Tensor2D(t.dptr_,                   mshadow::Shape2(1, d_input));
    h_l = Tensor2D(t.dptr_ + d_input,         mshadow::Shape2(1, d_mem));
    h_m = Tensor2D(t.dptr_ + d_input+1*d_mem, mshadow::Shape2(1, d_mem));
    h_t = Tensor2D(t.dptr_ + d_input+2*d_mem, mshadow::Shape2(1, d_mem));
  }


  virtual void ForwardOneStep(Tensor2D pre_c_l, // left
                              Tensor2D pre_c_m, // left top
                              Tensor2D pre_c_t, // top
                              Tensor2D pre_h_l,
                              Tensor2D pre_h_m,
                              Tensor2D pre_h_t,
                              Tensor2D cur_x,
                              Tensor2D cur_g,
                              Tensor2D cur_c,
                              Tensor2D cur_h) {
      Tensor2D w_data = this->params[0].data_d2_reverse();
      Tensor2D b_data = this->params[1].data_d2_reverse();

      mshadow::TensorContainer<xpu, 2> input(mshadow::Shape2(1, d_input + 3*d_mem));
      concat_input(cur_x, pre_h_l, pre_h_m, pre_h_t, input);
      cur_g = dot(input, w_data);
      if (!no_bias) {
        cur_g += b_data;
      }

      high_resolution_clock::time_point b_time_2 = high_resolution_clock::now();
      Tensor2D i, f_l, f_m, f_t, o, cc;
      SplitGate(cur_g, i, f_l, f_m, f_t, o, cc);
      i   = mshadow::expr::F<op::sigmoid>(i);   // logi
      f_l = mshadow::expr::F<op::sigmoid>(f_l); // logi
      f_m = mshadow::expr::F<op::sigmoid>(f_m); // logi
      f_t = mshadow::expr::F<op::sigmoid>(f_t); // logi
      o   = mshadow::expr::F<op::sigmoid>(o);   // logi
      cc  = mshadow::expr::F<op::tanh>(cc);     // tanh 
      high_resolution_clock::time_point e_time_2 = high_resolution_clock::now();
      time_2 += duration_cast<duration<double>>(e_time_2 - b_time_2);

      cur_c = f_l * pre_c_l + f_m * pre_c_m + f_t * pre_c_t + i * cc;
      // if (!no_out_tanh) {
        cur_h = o * mshadow::expr::F<op::tanh>(cur_c); // tanh
      // } else {
      //   cur_h = o * cur_c; 
      // }
      high_resolution_clock::time_point e_time_3 = high_resolution_clock::now();
      time_3 += duration_cast<duration<double>>(e_time_3 - e_time_2);
  }

  // x: (x_max_len, y_max_len, d_input)
  void ForwardLeftTop2RightBottom(Tensor3D x, int x_len, int y_len, 
                                  Tensor3D g, Tensor3D c, Tensor3D h) {
    utils::Check(x_len > 0 && y_len > 0 && x_len <= x.size(0) && y_len <= x.size(1), "LstmD2Layer: input size error.");
    Tensor2D pre_c_l, pre_c_m, pre_c_t;
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    // not need any padding, begin h and c are set to 0
    for (index_t row_idx = 0; row_idx < x_len; ++row_idx) {
      for (index_t col_idx = 0; col_idx < y_len; ++col_idx) {
        if (row_idx == 0) {
          pre_c_t = begin_c;
          pre_h_t = begin_h;
        } else {
          pre_c_t = c[row_idx-1].Slice(col_idx, col_idx+1);
          pre_h_t = h[row_idx-1].Slice(col_idx, col_idx+1);
        }
        if (col_idx == 0) {
          pre_c_l = begin_c;
          pre_h_l = begin_h;
        } else {
          pre_c_l = c[row_idx].Slice(col_idx-1, col_idx);
          pre_h_l = h[row_idx].Slice(col_idx-1, col_idx);
        }
        if (row_idx == 0 || col_idx == 0) {
          pre_c_m = begin_c;
          pre_h_m = begin_h;
        } else {
          pre_c_m = c[row_idx-1].Slice(col_idx-1, col_idx);
          pre_h_m = h[row_idx-1].Slice(col_idx-1, col_idx);
        }

        ForwardOneStep(pre_c_l,
                       pre_c_m,
                       pre_c_t,
                       pre_h_l,
                       pre_h_m,
                       pre_h_t,
                       x[row_idx].Slice(col_idx, col_idx+1),
                       g[row_idx].Slice(col_idx, col_idx+1), 
                       c[row_idx].Slice(col_idx, col_idx+1), 
                       h[row_idx].Slice(col_idx, col_idx+1));
      }
    }
  }

  // x: (x_max_len, y_max_len, d_input)
  void ForwardRightBottom2LeftTop(Tensor3D x, int x_len, int y_len, 
                                  Tensor3D g, Tensor3D c, Tensor3D h) {
    utils::Check(x_len > 0 && y_len > 0 && x_len <= x.size(0) && y_len <= x.size(1), "LstmD2Layer: input size error.");
    Tensor2D pre_c_l, pre_c_m, pre_c_t;
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    // not need any padding, begin h and c are set to 0
    for (int row_idx = x_len-1; row_idx >= 0; --row_idx) {
      for (int col_idx = y_len-1; col_idx >= 0; --col_idx) {
        if (row_idx == x_len-1) {
          pre_c_t = begin_c;
          pre_h_t = begin_h;
        } else {
          pre_c_t = c[row_idx+1].Slice(col_idx, col_idx+1);
          pre_h_t = h[row_idx+1].Slice(col_idx, col_idx+1);
        }
        if (col_idx == y_len-1) {
          pre_c_l = begin_c;
          pre_h_l = begin_h;
        } else {
          pre_c_l = c[row_idx].Slice(col_idx+1, col_idx+2);
          pre_h_l = h[row_idx].Slice(col_idx+1, col_idx+2);
        }
        if (row_idx == x_len-1 || col_idx == y_len-1) {
          pre_c_m = begin_c;
          pre_h_m = begin_h;
        } else {
          pre_c_m = c[row_idx+1].Slice(col_idx+1, col_idx+2);
          pre_h_m = h[row_idx+1].Slice(col_idx+1, col_idx+2);
        }

        ForwardOneStep(pre_c_l,
                       pre_c_m,
                       pre_c_t,
                       pre_h_l,
                       pre_h_m,
                       pre_h_t,
                       x[row_idx].Slice(col_idx, col_idx+1),
                       g[row_idx].Slice(col_idx, col_idx+1), 
                       c[row_idx].Slice(col_idx, col_idx+1), 
                       h[row_idx].Slice(col_idx, col_idx+1));
      }
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
// #if DEBUG
//     checkNanParams();
// #endif
    Tensor4D bottom_data = bottom[0]->data;
    Tensor2D bottom_len  = bottom[0]->length;
    Tensor4D top_data    = top[0]->data;

    utils::Check(bottom_len.size(0) == bottom_data.size(0) && 
                 bottom_len.size(1) == 2, "LstmD2Layer: input length error.");
    top[0]->length = mshadow::expr::F<op::identity>(bottom[0]->length);

    top_data = 0.f; c = 0.f, g = 0.f; c_er = 0.f; g_er = 0.f;
    high_resolution_clock::time_point b_time_1 = high_resolution_clock::now();
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      int x_len = bottom_len[batch_idx][0];
      int y_len = bottom_len[batch_idx][1];
      utils::Assert(x_len >= 0 && y_len >= 0, "LstmD2Layer: sequence length error.");
      if (!reverse) {
        ForwardLeftTop2RightBottom(bottom_data[batch_idx],
                                   x_len, y_len,
                                   g[batch_idx], 
                                   c[batch_idx],
                                   top_data[batch_idx]);
      } else {
        ForwardRightBottom2LeftTop(bottom_data[batch_idx],
                                   x_len, y_len,
                                   g[batch_idx], 
                                   c[batch_idx],
                                   top_data[batch_idx]);
      }
    }
    high_resolution_clock::time_point e_time_1 = high_resolution_clock::now();
    time_1 += duration_cast<duration<double>>(e_time_1 - b_time_1);
	// utils::Printf("\tLSTM D2 Time:%fs,%fs,%f\n", time_1.count(), time_2.count(), time_3.count()); 
// #if DEBUG
//     checkNanParams();
// #endif
  }

  // too tricky, may bring errors
  void SplitGate(Tensor2D &g, 
                 Tensor2D &i, 
                 Tensor2D &f_l, Tensor2D &f_m, Tensor2D &f_t, 
                 Tensor2D &o, 
                 Tensor2D &cc) {
    utils::Check(g.size(0) == 1, "LstmD2Layer: gate problem."); 

    i   = Tensor2D(g.dptr_,             mshadow::Shape2(1, d_mem));
    f_l = Tensor2D(g.dptr_ + 1 * d_mem, mshadow::Shape2(1, d_mem));
    f_m = Tensor2D(g.dptr_ + 2 * d_mem, mshadow::Shape2(1, d_mem));
    f_t = Tensor2D(g.dptr_ + 3 * d_mem, mshadow::Shape2(1, d_mem));
    o   = Tensor2D(g.dptr_ + 4 * d_mem, mshadow::Shape2(1, d_mem));
    cc  = Tensor2D(g.dptr_ + 5 * d_mem, mshadow::Shape2(1, d_mem));
  }

  float norm2(Tensor2D t) {
    float norm2 = 0.f;
    for (int i = 0; i < t.size(0); ++i) {
      for (int j = 0; j < t.size(1); ++j) {
        norm2 += t[i][j] * t[i][j];
      }
    }
    if (norm2 == 0.f) norm2 = 0.0000000001;
    return sqrt(norm2);
  }

  void BpOneStep(Tensor2D cur_h_er,
                 Tensor2D pre_c_l,
                 Tensor2D pre_c_m,
                 Tensor2D pre_c_t,
                 Tensor2D pre_h_l,
                 Tensor2D pre_h_m,
                 Tensor2D pre_h_t,
                 Tensor2D cur_x,
                 Tensor2D cur_g,
                 Tensor2D cur_c,
                 Tensor2D cur_h,
                 Tensor2D cur_c_er,
                 Tensor2D cur_g_er,
                 Tensor2D pre_c_er_l,
                 Tensor2D pre_c_er_m,
                 Tensor2D pre_c_er_t,
                 Tensor2D pre_h_er_l,
                 Tensor2D pre_h_er_m,
                 Tensor2D pre_h_er_t,
                 Tensor2D cur_x_er) {

    Tensor2D w_data = this->params[0].data_d2_reverse();
    Tensor2D w_er   = this->params[0].diff_d2_reverse();
    Tensor2D b_er   = this->params[1].diff_d2_reverse();

    // gradient normalization by norm 2
    // float n2 = norm2(cur_h_er);
    // if (n2 > grad_norm2) {
    //   // utils::Printf("LSTM: grad norm, %f,%f\n", n2, grad_norm2);
    //   cur_h_er *= (grad_norm2/n2);
    // }
    
    Tensor2D i, f_l, f_m, f_t, o, cc;
    Tensor2D i_er, f_er_l, f_er_m, f_er_t, o_er, cc_er;
    SplitGate(cur_g, i, f_l, f_m, f_t, o, cc);
    SplitGate(cur_g_er, i_er, f_er_l, f_er_m, f_er_t, o_er, cc_er);

    // if (!no_out_tanh) {
      mshadow::TensorContainer<xpu, 2> tanhc(cur_c.shape_);
      tanhc = mshadow::expr::F<op::tanh>(cur_c);
      o_er = mshadow::expr::F<op::sigmoid_grad>(o) * (cur_h_er * tanhc); // logi
      cur_c_er += mshadow::expr::F<op::tanh_grad>(tanhc) * (cur_h_er * o); // NOTICE: += 
    // } else {
    //   o_er = mshadow::expr::F<op::sigmoid_grad>(o) * (cur_h_er * cur_c); // logi
    //   cur_c_er += cur_h_er * o;
    // }

    i_er = mshadow::expr::F<op::sigmoid_grad>(i) * (cur_c_er * cc);    // logi
    cc_er = mshadow::expr::F<op::tanh_grad>(cc) * (cur_c_er * i);      // tanh
    pre_c_er_l += cur_c_er * f_l; // NOTICE: in 1D, use '+' is correct, in 2D, must use '+='
    pre_c_er_m += cur_c_er * f_m;
    pre_c_er_t += cur_c_er * f_t;
    f_er_l = mshadow::expr::F<op::sigmoid_grad>(f_l) * (cur_c_er * pre_c_l); // logi
    f_er_m = mshadow::expr::F<op::sigmoid_grad>(f_m) * (cur_c_er * pre_c_m); // logi
    f_er_t = mshadow::expr::F<op::sigmoid_grad>(f_t) * (cur_c_er * pre_c_t); // logi

    mshadow::TensorContainer<xpu, 2> input_er(mshadow::Shape2(1, d_input + 3*d_mem));
    input_er = dot(cur_g_er, w_data.T());
    Tensor2D x_er_, h_er_l_, h_er_m_, h_er_t_;
    split_input(input_er, x_er_, h_er_l_, h_er_m_, h_er_t_);
    pre_h_er_l += h_er_l_;
    pre_h_er_m += h_er_m_;
    pre_h_er_t += h_er_t_;
    cur_x_er   += x_er_;

    // grad
    if (!no_bias) {
      b_er += cur_g_er;
    }

    mshadow::TensorContainer<xpu, 2> input(mshadow::Shape2(1, d_input + 3*d_mem));
    concat_input(cur_x, pre_h_l, pre_h_m, pre_h_t, input);

    w_er += dot(input.T(), cur_g_er); 
  }

  void BackpropForLeftTop2RightBottomLstm(int x_len, int y_len,
                                          Tensor3D h, Tensor3D h_er, 
                                          Tensor3D c, Tensor3D c_er, 
                                          Tensor3D g, Tensor3D g_er, 
                                          Tensor3D x, Tensor3D x_er) {
    Tensor2D cur_x, cur_g, cur_c, cur_h;
    Tensor2D cur_x_er, cur_g_er, cur_c_er, cur_h_er; 
    Tensor2D pre_c_l, pre_c_m, pre_c_t;
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    Tensor2D pre_c_er_l, pre_c_er_m, pre_c_er_t;
    Tensor2D pre_h_er_l, pre_h_er_m, pre_h_er_t;
    for (int row_idx = x_len-1; row_idx >= 0; --row_idx) {
      for (int col_idx = y_len-1; col_idx >= 0; --col_idx) {
        if (row_idx == 0) {
          pre_c_t = begin_c;
          pre_h_t = begin_h;
          pre_c_er_t = begin_c_er;
          pre_h_er_t = begin_h_er;
        } else {
          pre_c_t = c[row_idx-1].Slice(col_idx, col_idx+1);
          pre_h_t = h[row_idx-1].Slice(col_idx, col_idx+1);
          pre_c_er_t = c_er[row_idx-1].Slice(col_idx, col_idx+1);
          pre_h_er_t = h_er[row_idx-1].Slice(col_idx, col_idx+1);
        }
        if (col_idx == 0) {
          pre_c_l = begin_c;
          pre_h_l = begin_h;
          pre_c_er_l = begin_c_er;
          pre_h_er_l = begin_h_er;
        } else {
          pre_c_l = c[row_idx].Slice(col_idx-1, col_idx);
          pre_h_l = h[row_idx].Slice(col_idx-1, col_idx);
          pre_c_er_l = c_er[row_idx].Slice(col_idx-1, col_idx);
          pre_h_er_l = h_er[row_idx].Slice(col_idx-1, col_idx);
        }
        if (row_idx == 0 || col_idx == 0) {
          pre_c_m = begin_c;
          pre_h_m = begin_h;
          pre_c_er_m = begin_c_er;
          pre_h_er_m = begin_h_er;
        } else {
          pre_c_m = c[row_idx-1].Slice(col_idx-1, col_idx);
          pre_h_m = h[row_idx-1].Slice(col_idx-1, col_idx);
          pre_c_er_m = c_er[row_idx-1].Slice(col_idx-1, col_idx);
          pre_h_er_m = h_er[row_idx-1].Slice(col_idx-1, col_idx);
        }
        cur_x    = x[row_idx].Slice(col_idx, col_idx+1);
        cur_g    = g[row_idx].Slice(col_idx, col_idx+1);
        cur_c    = c[row_idx].Slice(col_idx, col_idx+1);
        cur_h    = h[row_idx].Slice(col_idx, col_idx+1);
        cur_x_er = x_er[row_idx].Slice(col_idx, col_idx+1);
        cur_g_er = g_er[row_idx].Slice(col_idx, col_idx+1);
        cur_c_er = c_er[row_idx].Slice(col_idx, col_idx+1);
        cur_h_er = h_er[row_idx].Slice(col_idx, col_idx+1);
        BpOneStep(cur_h_er,
                  pre_c_l,
                  pre_c_m,
                  pre_c_t,
                  pre_h_l,
                  pre_h_m,
                  pre_h_t,
                  cur_x,
                  cur_g,
                  cur_c,
                  cur_h,
                  cur_c_er,
                  cur_g_er,
                  pre_c_er_l,
                  pre_c_er_m,
                  pre_c_er_t,
                  pre_h_er_l,
                  pre_h_er_m,
                  pre_h_er_t,
                  cur_x_er);
      }
    }
  }
  void BackpropForRightBottom2LeftTopLstm(int x_len, int y_len,
                                          Tensor3D h, Tensor3D h_er, 
                                          Tensor3D c, Tensor3D c_er, 
                                          Tensor3D g, Tensor3D g_er, 
                                          Tensor3D x, Tensor3D x_er) {
    Tensor2D cur_x, cur_g, cur_c, cur_h;
    Tensor2D cur_x_er, cur_g_er, cur_c_er, cur_h_er; 
    Tensor2D pre_c_l, pre_c_m, pre_c_t;
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    Tensor2D pre_c_er_l, pre_c_er_m, pre_c_er_t;
    Tensor2D pre_h_er_l, pre_h_er_m, pre_h_er_t;
    for (index_t row_idx = 0; row_idx < x_len; ++row_idx) {
      for (index_t col_idx = 0; col_idx < y_len; ++col_idx) {
        if (row_idx == x_len-1) {
          pre_c_t = begin_c;
          pre_h_t = begin_h;
          pre_c_er_t = begin_c_er;
          pre_h_er_t = begin_h_er;
        } else {
          pre_c_t = c[row_idx+1].Slice(col_idx, col_idx+1);
          pre_h_t = h[row_idx+1].Slice(col_idx, col_idx+1);
          pre_c_er_t = c_er[row_idx+1].Slice(col_idx, col_idx+1);
          pre_h_er_t = h_er[row_idx+1].Slice(col_idx, col_idx+1);
        }
        if (col_idx == y_len-1) {
          pre_c_l = begin_c;
          pre_h_l = begin_h;
          pre_c_er_l = begin_c_er;
          pre_h_er_l = begin_h_er;
        } else {
          pre_c_l = c[row_idx].Slice(col_idx+1, col_idx+2);
          pre_h_l = h[row_idx].Slice(col_idx+1, col_idx+2);
          pre_c_er_l = c_er[row_idx].Slice(col_idx+1, col_idx+2);
          pre_h_er_l = h_er[row_idx].Slice(col_idx+1, col_idx+2);
        }
        if (row_idx == x_len-1 || col_idx == y_len-1) {
          pre_c_m = begin_c;
          pre_h_m = begin_h;
          pre_c_er_m = begin_c_er;
          pre_h_er_m = begin_h_er;
        } else {
          pre_c_m = c[row_idx+1].Slice(col_idx+1, col_idx+2);
          pre_h_m = h[row_idx+1].Slice(col_idx+1, col_idx+2);
          pre_c_er_m = c_er[row_idx+1].Slice(col_idx+1, col_idx+2);
          pre_h_er_m = h_er[row_idx+1].Slice(col_idx+1, col_idx+2);
        }
        cur_x    = x[row_idx].Slice(col_idx, col_idx+1);
        cur_g    = g[row_idx].Slice(col_idx, col_idx+1);
        cur_c    = c[row_idx].Slice(col_idx, col_idx+1);
        cur_h    = h[row_idx].Slice(col_idx, col_idx+1);
        cur_x_er = x_er[row_idx].Slice(col_idx, col_idx+1);
        cur_g_er = g_er[row_idx].Slice(col_idx, col_idx+1);
        cur_c_er = c_er[row_idx].Slice(col_idx, col_idx+1);
        cur_h_er = h_er[row_idx].Slice(col_idx, col_idx+1);
        BpOneStep(cur_h_er,
                  pre_c_l,
                  pre_c_m,
                  pre_c_t,
                  pre_h_l,
                  pre_h_m,
                  pre_h_t,
                  cur_x,
                  cur_g,
                  cur_c,
                  cur_h,
                  cur_c_er,
                  cur_g_er,
                  pre_c_er_l,
                  pre_c_er_m,
                  pre_c_er_t,
                  pre_h_er_l,
                  pre_h_er_m,
                  pre_h_er_t,
                  cur_x_er);
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
// #if DEBUG
//     checkNanParams();
// #endif
    mshadow::Tensor<xpu, 4> h    = top[0]->data;
    mshadow::Tensor<xpu, 4> h_er = top[0]->diff;
    mshadow::Tensor<xpu, 4> x    = bottom[0]->data;
    mshadow::Tensor<xpu, 4> x_er = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> len  = bottom[0]->length;
        
    high_resolution_clock::time_point b_time_4 = high_resolution_clock::now();
    begin_c_er = 0.; begin_h_er = 0.; g_er = 0.; c_er = 0.;
    for (index_t batch_idx = 0; batch_idx < x.size(0); ++batch_idx) {
      int x_len = len[batch_idx][0];
      int y_len = len[batch_idx][1];
      if (!reverse) {
        BackpropForLeftTop2RightBottomLstm(x_len, y_len,
                                           h[batch_idx],
                                           h_er[batch_idx], 
                                           c[batch_idx],
                                           c_er[batch_idx],
                                           g[batch_idx],
                                           g_er[batch_idx],
                                           x[batch_idx],
                                           x_er[batch_idx]);
      } else {
        BackpropForRightBottom2LeftTopLstm(x_len, y_len,
                                           h[batch_idx],
                                           h_er[batch_idx], 
                                           c[batch_idx],
                                           c_er[batch_idx],
                                           g[batch_idx],
                                           g_er[batch_idx],
                                           x[batch_idx],
                                           x_er[batch_idx]);

      }
    }
    high_resolution_clock::time_point e_time_4 = high_resolution_clock::now();
    time_4 += duration_cast<duration<double>>(e_time_4 - b_time_4);
	// utils::Printf("\tLSTM D2 BP Time:%fs\n", time_4.count()); 
    // this->params[0].CutOffGradient(grad_cut_off);
    // this->params[1].CutOffGradient(grad_cut_off);
    // this->params[2].CutOffGradient(grad_cut_off);

    // this->params[0].PrintStatistic("LSTM W");
    // this->params[1].PrintStatistic("LSTM U");
    // this->params[2].PrintStatistic("LSTM b");
// #if DEBUG
//     checkNanParams();
// #endif
  }
  // void LoadTensor(Json::Value &tensor_root, mshadow::TensorContainer<xpu, 4> &t) {
  //   Json::Value data_root = tensor_root["data"];
  //   int s0 = data_root["shape"][0].asInt();
  //   int s1 = data_root["shape"][1].asInt();
  //   int s2 = data_root["shape"][2].asInt();
  //   int s3 = data_root["shape"][3].asInt();
  //   utils::Check(t.size(0) == s0 && t.size(1) == s1 && t.size(2) == s2 && t.size(3) == s3, 
  //                "LstmD2Layer: load tensor error.");
  //   int size = s0*s1*s2*s3;
  //   for (int i = 0; i < size; ++i) {
  //     t.dptr_[i] = data_root["value"][i].asFloat();
  //   }
  // }
  // void LoadParam() {
  //   utils::Printf("LstmD2Layer: load params...\n");
  //   Json::Value param_root;
  //   ifstream ifs(param_file.c_str());
  //   ifs >> param_root;
  //   ifs.close();
  //   LoadTensor(param_root[0], this->params[0].data);
  //   LoadTensor(param_root[1], this->params[1].data);
  //   LoadTensor(param_root[2], this->params[2].data);
  // }

 public:
// protected:
  // float max_norm2;
  int d_mem, d_input;
  bool no_bias, reverse; //, no_out_tanh; 
  // float grad_norm2;
  float o_gate_bias_init;
  float f_gate_bias_init;
  // float grad_cut_off;
  // string param_file;
  mshadow::TensorContainer<xpu, 4> c, g, c_er, g_er;
  mshadow::TensorContainer<xpu, 2> begin_h, begin_c, begin_c_er, begin_h_er;
  // clock_t time_1, time_2, time_3, time_4;
  duration<double> time_1, time_2, time_3, time_4;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LSTM_LAYER_INL_HPP_
