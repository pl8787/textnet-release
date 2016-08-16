#ifndef TEXTNET_LAYER_GRU_D2_LAYER_INL_HPP_
#define TEXTNET_LAYER_GRU_D2_LAYER_INL_HPP_

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
class GruD2Layer : public Layer<xpu> {
 public:
  GruD2Layer(LayerType type) { this->layer_type = type; }
  virtual ~GruD2Layer(void) { }
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 4; }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);
    this->defaults["is_use_reset_gate"] = SettingV(true);
    // this->defaults["no_out_tanh"] = SettingV(false);
    // this->defaults["param_file"] = SettingV("");
    // this->defaults["o_gate_bias_init"] = SettingV(0.f);
    // this->defaults["f_gate_bias_init"] = SettingV(0.f);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    // this->defaults["max_norm2"] = SettingV();
    // this->defaults["grad_norm2"] = SettingV();
    this->defaults["d_mem"] = SettingV();
    this->defaults["is_diag_connection"] = SettingV();
    this->defaults["w_g_filler"] = SettingV();
    this->defaults["b_g_filler"] = SettingV();
    this->defaults["w_g_updater"] = SettingV();
    this->defaults["b_g_updater"] = SettingV();
    this->defaults["w_c_filler"] = SettingV();
    this->defaults["b_c_filler"] = SettingV();
    this->defaults["w_c_updater"] = SettingV();
    this->defaults["b_c_updater"] = SettingV();
    this->defaults["reverse"] = SettingV();
    // this->defaults["grad_cut_off"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(), "GruD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GruD2Layer:top size problem.");

    d_mem   = setting["d_mem"].iVal();
    d_input = bottom[0]->data.size(3);
    no_bias = setting["no_bias"].bVal();
    is_use_reset_gate = setting["is_use_reset_gate"].bVal();
    // no_out_tanh = setting["no_out_tanh"].bVal();
    reverse = setting["reverse"].bVal();
    is_diag_connection = setting["is_diag_connection"].bVal();
    // grad_norm2 = setting["grad_norm2"].fVal();
    // param_file = setting["param_file"].sVal();
    // o_gate_bias_init = setting["o_gate_bias_init"].fVal();
    // f_gate_bias_init = setting["f_gate_bias_init"].fVal();
    // grad_cut_off = setting["grad_cut_off"].fVal();
    // max_norm2 = setting["max_norm2"].fVal();

    begin_h.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_h_er.Resize(mshadow::Shape2(1, d_mem), 0.f);

    this->params.resize(4);
    this->params[0].Resize(1, 1, d_input+3*d_mem, 7*d_mem, true); // w and u is in one matrix, gate 
    this->params[1].Resize(1, 1, 1,               7*d_mem, true); // b, gate
    this->params[2].Resize(1, 1, d_input+3*d_mem, 1*d_mem, true); // w and u is in one matrix, cc
    this->params[3].Resize(1, 1, 1,               1*d_mem, true); // b, cc
    
    std::map<std::string, SettingV> &w_g_setting = *setting["w_g_filler"].mVal();
    std::map<std::string, SettingV> &b_g_setting = *setting["b_g_filler"].mVal();
    std::map<std::string, SettingV> &w_c_setting = *setting["w_c_filler"].mVal();
    std::map<std::string, SettingV> &b_c_setting = *setting["b_c_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_g_setting["init_type"].iVal(), w_g_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_g_setting["init_type"].iVal(), b_g_setting, this->prnd_);
    this->params[2].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_c_setting["init_type"].iVal(), w_c_setting, this->prnd_);
    this->params[3].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_c_setting["init_type"].iVal(), b_c_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    this->params[2].Init();
    this->params[3].Init();
    // if (f_gate_bias_init != 0.f) {
    //     init_f_gate_bias(); // this must be after init()
    // }
    // if (o_gate_bias_init != 0.f) {
    //     init_o_gate_bias(); // this must be after init()
    // }

    // if (!param_file.empty()) {
    //   LoadParam();
    // }
    
    std::map<std::string, SettingV> &w_g_updater = *setting["w_g_updater"].mVal();
    std::map<std::string, SettingV> &b_g_updater = *setting["b_g_updater"].mVal();
    std::map<std::string, SettingV> &w_c_updater = *setting["w_c_updater"].mVal();
    std::map<std::string, SettingV> &b_c_updater = *setting["b_c_updater"].mVal();

    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_g_updater["updater_type"].iVal(), w_g_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_g_updater["updater_type"].iVal(), b_g_updater, this->prnd_);
    this->params[2].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_c_updater["updater_type"].iVal(), w_c_updater, this->prnd_);
    this->params[3].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_c_updater["updater_type"].iVal(), b_c_updater, this->prnd_);
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
    utils::Check(bottom.size() == BottomNodeNum(), "GruD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GruD2Layer:top size problem.");
    
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    // (batch size, x_len, y_len, d_mem)
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);
    mshadow::Shape<4> shape_reset_h = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem*3);
    // input, forget * 3, output, candidate
    mshadow::Shape<4> shape_gate = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem*7); 

    top[0]->Resize(shape_out, mshadow::Shape2(shape_out[0],2), true);
    hi.Resize(shape_out, 0.f); // h input
    hi_er.Resize(shape_out, 0.f);
    g.Resize(shape_gate, 0.f);
    g_er.Resize(shape_gate, 0.f);
    reset_h.Resize(shape_reset_h, 0.f);
    reset_h_er.Resize(shape_reset_h, 0.f);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
      top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (! (bottom[0]->data.size(0) == top[0]->data.size(0))) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }

  void checkNan(float *p, int l) {
    for (int i = 0; i < l; ++i) {
      assert(!std::isnan(p[i]));
    }
  }

  void checkNanParams() {
    Tensor2D w_data = this->params[0].data_d2_reverse();
    Tensor2D w_diff = this->params[0].diff_d2_reverse();
    checkNan(w_data.dptr_, w_data.size());
    checkNan(w_diff.dptr_, w_diff.size());
  }

  // this layer copy the input value to a continue memory
  void concat_input(Tensor2D x, Tensor2D h_l, Tensor2D h_m, Tensor2D h_t, Tensor2D input) {
    utils::Check(x.size(0)  ==1 && h_l.size(0)==1 && h_m.size(0)==1 &&
                 h_t.size(0)==1 && input.size(0)  ==1, "GruD2Layer: size error.");
    utils::Check(x.size(1)+h_l.size(1)+h_m.size(1)+h_t.size(1) == input.size(1), "GruD2Layer: size error.");

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
  // return several tensors with pointers to the inut tensor
  void split_input(Tensor2D t, Tensor2D &x, Tensor2D &h_l, Tensor2D &h_m, Tensor2D &h_t) {
    utils::Check(t.size(0)==1 && t.size(1)==(d_input+3*d_mem), "GruD2Layer: size error.");

    x   = Tensor2D(t.dptr_,                   mshadow::Shape2(1, d_input));
    h_l = Tensor2D(t.dptr_ + d_input,         mshadow::Shape2(1, d_mem));
    h_m = Tensor2D(t.dptr_ + d_input+1*d_mem, mshadow::Shape2(1, d_mem));
    h_t = Tensor2D(t.dptr_ + d_input+2*d_mem, mshadow::Shape2(1, d_mem));
  }

  void diff_softmax_z_no_diag(Tensor2D z_i,    Tensor2D z_l,    Tensor2D z_t, 
                              Tensor2D z_i_er, Tensor2D z_l_er, Tensor2D z_t_er) {
    int len = z_i_er.size(1);
    mshadow::TensorContainer<xpu, 2> z_i_er_tmp(mshadow::Shape2(1, len));
    mshadow::TensorContainer<xpu, 2> z_l_er_tmp(mshadow::Shape2(1, len));
    // mshadow::TensorContainer<xpu, 2> z_m_er_tmp(mshadow::Shape2(1, len));
    mshadow::TensorContainer<xpu, 2> z_t_er_tmp(mshadow::Shape2(1, len));
    // Tensor2D tmp_1 = z_i_er_tmp;
    // Tensor2D tmp_2 = z_l_er_tmp;
    // Tensor2D tmp_3 = z_m_er_tmp;
    // Tensor2D tmp_4 = z_t_er_tmp;
    // utils::Check(false, "tmp");
    z_i_er_tmp = mshadow::expr::F<op::identity>(z_i_er);
    z_l_er_tmp = mshadow::expr::F<op::identity>(z_l_er);
    // z_m_er_tmp = mshadow::expr::F<op::identity>(z_m_er);
    z_t_er_tmp = mshadow::expr::F<op::identity>(z_t_er);
    for (int i = 0; i < len; ++i) {
      float error_sum = 0.f;
      error_sum += z_i_er_tmp[0][i] * z_i[0][i];
      error_sum += z_l_er_tmp[0][i] * z_l[0][i];
      // error_sum += z_m_er_tmp[0][i] * z_m[0][i];
      error_sum += z_t_er_tmp[0][i] * z_t[0][i];

      z_i_er[0][i] = (z_i_er_tmp[0][i] - error_sum) * z_i[0][i];
      z_l_er[0][i] = (z_l_er_tmp[0][i] - error_sum) * z_l[0][i];
      // z_m_er[0][i] = (z_m_er_tmp[0][i] - error_sum) * z_m[0][i];
      z_t_er[0][i] = (z_t_er_tmp[0][i] - error_sum) * z_t[0][i];
    }
  }
  
  void softmax_z_no_diag(Tensor2D z_i, Tensor2D z_l, Tensor2D z_t) {
    z_i = mshadow::expr::F<op::orc_exp>(z_i);
    z_l = mshadow::expr::F<op::orc_exp>(z_l);
    // z_m = mshadow::expr::F<op::orc_exp>(z_m);
    z_t = mshadow::expr::F<op::orc_exp>(z_t);

    // normalize by col
    for (int i = 0; i < z_i.size(1); ++i) {
      float sum = 0.f;
      sum += z_i[0][i];
      sum += z_l[0][i];
      // sum += z_m[0][i];
      sum += z_t[0][i];

      z_i[0][i] /= sum;
      z_l[0][i] /= sum;
      // z_m[0][i] /= sum;
      z_t[0][i] /= sum;
    }
  }


  void diff_softmax_z(Tensor2D z_i, Tensor2D z_l, Tensor2D z_m, Tensor2D z_t, 
                      Tensor2D z_i_er, Tensor2D z_l_er, Tensor2D z_m_er, Tensor2D z_t_er) {
    int len = z_i_er.size(1);
    mshadow::TensorContainer<xpu, 2> z_i_er_tmp(mshadow::Shape2(1, len));
    mshadow::TensorContainer<xpu, 2> z_l_er_tmp(mshadow::Shape2(1, len));
    mshadow::TensorContainer<xpu, 2> z_m_er_tmp(mshadow::Shape2(1, len));
    mshadow::TensorContainer<xpu, 2> z_t_er_tmp(mshadow::Shape2(1, len));
    // Tensor2D tmp_1 = z_i_er_tmp;
    // Tensor2D tmp_2 = z_l_er_tmp;
    // Tensor2D tmp_3 = z_m_er_tmp;
    // Tensor2D tmp_4 = z_t_er_tmp;
    // utils::Check(false, "tmp");
    z_i_er_tmp = mshadow::expr::F<op::identity>(z_i_er);
    z_l_er_tmp = mshadow::expr::F<op::identity>(z_l_er);
    z_m_er_tmp = mshadow::expr::F<op::identity>(z_m_er);
    z_t_er_tmp = mshadow::expr::F<op::identity>(z_t_er);
    for (int i = 0; i < len; ++i) {
      float error_sum = 0.f;
      error_sum += z_i_er_tmp[0][i] * z_i[0][i];
      error_sum += z_l_er_tmp[0][i] * z_l[0][i];
      error_sum += z_m_er_tmp[0][i] * z_m[0][i];
      error_sum += z_t_er_tmp[0][i] * z_t[0][i];

      z_i_er[0][i] = (z_i_er_tmp[0][i] - error_sum) * z_i[0][i];
      z_l_er[0][i] = (z_l_er_tmp[0][i] - error_sum) * z_l[0][i];
      z_m_er[0][i] = (z_m_er_tmp[0][i] - error_sum) * z_m[0][i];
      z_t_er[0][i] = (z_t_er_tmp[0][i] - error_sum) * z_t[0][i];
    }
  }
  
  void softmax_z(Tensor2D z_i, Tensor2D z_l, Tensor2D z_m, Tensor2D z_t) {
    z_i = mshadow::expr::F<op::orc_exp>(z_i);
    z_l = mshadow::expr::F<op::orc_exp>(z_l);
    z_m = mshadow::expr::F<op::orc_exp>(z_m);
    z_t = mshadow::expr::F<op::orc_exp>(z_t);

    // normalize by col
    for (int i = 0; i < z_i.size(1); ++i) {
      float sum = 0.f;
      sum += z_i[0][i];
      sum += z_l[0][i];
      sum += z_m[0][i];
      sum += z_t[0][i];

      z_i[0][i] /= sum;
      z_l[0][i] /= sum;
      z_m[0][i] /= sum;
      z_t[0][i] /= sum;
    }
  }

  virtual void ForwardOneStep(Tensor2D pre_h_l, // left
                              Tensor2D pre_h_m, // middle
                              Tensor2D pre_h_t, // top
                              Tensor2D cur_x,
                              Tensor2D cur_g,
                              Tensor2D reset_h, 
                              Tensor2D cur_hi,
                              Tensor2D cur_h) {
    utils::Check(cur_x.size(0) == 1, "GruD2Layer: ForwardOneStep(): input size error.");
    Tensor2D w_g_data = this->params[0].data_d2_reverse();
    Tensor2D b_g_data = this->params[1].data_d2_reverse();
    Tensor2D w_c_data = this->params[2].data_d2_reverse();
    Tensor2D b_c_data = this->params[3].data_d2_reverse();

    mshadow::TensorContainer<xpu, 2> input(mshadow::Shape2(1, d_input + 3*d_mem));
    concat_input(cur_x, pre_h_l, pre_h_m, pre_h_t, input);

    cur_g = dot(input, w_g_data);
    if (!no_bias) {
      cur_g += b_g_data;
    }

    Tensor2D r_l, r_m, r_t, z_i, z_l, z_m, z_t;
    SplitGate(cur_g, r_l, r_m, r_t, z_i, z_l, z_m, z_t);
    r_l = mshadow::expr::F<op::sigmoid>(r_l); // logi
    r_m = mshadow::expr::F<op::sigmoid>(r_m); // logi
    r_t = mshadow::expr::F<op::sigmoid>(r_t); // logi

    Tensor2D reset_h_l(reset_h.dptr_,         mshadow::Shape2(1,d_mem));
    Tensor2D reset_h_m(reset_h.dptr_+d_mem,   mshadow::Shape2(1,d_mem));
    Tensor2D reset_h_t(reset_h.dptr_+d_mem*2, mshadow::Shape2(1,d_mem));
    if (is_use_reset_gate) {
      reset_h_l = r_l * pre_h_l;
      reset_h_m = r_m * pre_h_m;
      reset_h_t = r_t * pre_h_t;
    } else {
      reset_h_l = mshadow::expr::F<op::identity>(pre_h_l);
      reset_h_m = mshadow::expr::F<op::identity>(pre_h_m);
      reset_h_t = mshadow::expr::F<op::identity>(pre_h_t);
    }

    concat_input(cur_x, 
                 reset_h_l,
                 reset_h_m,
                 reset_h_t,
                 input);
    cur_hi = dot(input, w_c_data);
    if (!no_bias) {
      cur_hi += b_c_data;
    }
    cur_hi = mshadow::expr::F<op::tanh>(cur_hi);

    // here we use a softmax layer for input and forget gate on each dimension
    // NOTE: on each dimension
    if (is_diag_connection) {
      softmax_z(z_i, z_l, z_m, z_t);
      cur_h = cur_hi * z_i + pre_h_l * z_l + pre_h_m * z_m + pre_h_t * z_t;
    } else {
      softmax_z_no_diag(z_i, z_l, z_t);
      cur_h = cur_hi * z_i + pre_h_l * z_l + pre_h_t * z_t;
    }
  }

  void BpOneStep(Tensor2D cur_h_er,
                 Tensor2D pre_h_l,
                 Tensor2D pre_h_m,
                 Tensor2D pre_h_t,
                 Tensor2D cur_x,
                 Tensor2D cur_reset_h,
                 Tensor2D cur_g,
                 Tensor2D cur_hi,
                 Tensor2D cur_hi_er,
                 Tensor2D cur_g_er,
                 Tensor2D pre_h_l_er,
                 Tensor2D pre_h_m_er,
                 Tensor2D pre_h_t_er,
                 Tensor2D cur_x_er) {
    Tensor2D w_g_data = this->params[0].data_d2_reverse();
    Tensor2D w_g_er   = this->params[0].diff_d2_reverse();
    Tensor2D b_g_er   = this->params[1].diff_d2_reverse();
    Tensor2D w_c_data = this->params[2].data_d2_reverse();
    Tensor2D w_c_er   = this->params[2].diff_d2_reverse();
    Tensor2D b_c_er   = this->params[3].diff_d2_reverse();
    Tensor2D r_l, r_m, r_t, z_i, z_l, z_m, z_t;
    Tensor2D r_l_er, r_m_er, r_t_er, z_i_er, z_l_er, z_m_er, z_t_er;
    SplitGate(cur_g, r_l, r_m, r_t, z_i, z_l, z_m, z_t);
    SplitGate(cur_g_er, r_l_er, r_m_er, r_t_er, z_i_er, z_l_er, z_m_er, z_t_er);
    
    z_i_er = cur_h_er * cur_hi; 
    z_l_er = cur_h_er * pre_h_l; 
    if (is_diag_connection) {
    z_m_er = cur_h_er * pre_h_m; 
    }
    z_t_er = cur_h_er * pre_h_t; 

    cur_hi_er  = mshadow::expr::F<op::tanh_grad>(cur_hi) * (cur_h_er * z_i);
    pre_h_l_er += cur_h_er * z_l; // NOTE: += 
    if (is_diag_connection) {
    pre_h_m_er += cur_h_er * z_m; // NOTE: +=
    }
    pre_h_t_er += cur_h_er * z_t; // NOTE: +=

    if (is_diag_connection) {
      diff_softmax_z(z_i, z_l, z_m, z_t, z_i_er, z_l_er, z_m_er, z_t_er);
    } else {
      diff_softmax_z_no_diag(z_i, z_l, z_t, z_i_er, z_l_er, z_t_er);
    }

    mshadow::TensorContainer<xpu, 2> input(mshadow::Shape2(1, d_input + 3*d_mem));
    mshadow::TensorContainer<xpu, 2> input_er(mshadow::Shape2(1, d_input + 3*d_mem));
    Tensor2D reset_h_l(cur_reset_h.dptr_,         mshadow::Shape2(1,d_mem));
    Tensor2D reset_h_m(cur_reset_h.dptr_+d_mem,   mshadow::Shape2(1,d_mem));
    Tensor2D reset_h_t(cur_reset_h.dptr_+d_mem*2, mshadow::Shape2(1,d_mem));
    concat_input(cur_x, 
                 reset_h_l,
                 reset_h_m,
                 reset_h_t,
                 input);

    w_c_er += dot(input.T(), cur_hi_er);
    if (!no_bias) {
      b_c_er += cur_hi_er;
    }
    input_er = dot(cur_hi_er, w_c_data.T());
    Tensor2D cur_x_er_tmp, reset_h_l_er, reset_h_m_er, reset_h_t_er;
    split_input(input_er,
                cur_x_er_tmp,
                reset_h_l_er,
                reset_h_m_er,
                reset_h_t_er);

    cur_x_er += cur_x_er_tmp;


    if (is_use_reset_gate) {
      r_l_er = mshadow::expr::F<op::sigmoid_grad>(r_l) * (reset_h_l_er * pre_h_l);
      r_m_er = mshadow::expr::F<op::sigmoid_grad>(r_m) * (reset_h_m_er * pre_h_m);
      r_t_er = mshadow::expr::F<op::sigmoid_grad>(r_t) * (reset_h_t_er * pre_h_t);
      pre_h_l_er += reset_h_l_er * r_l;
      pre_h_m_er += reset_h_m_er * r_m;
      pre_h_t_er += reset_h_t_er * r_t;
    } else {
      pre_h_l_er += reset_h_l_er;
      pre_h_m_er += reset_h_m_er;
      pre_h_t_er += reset_h_t_er;
    }

    concat_input(cur_x, pre_h_l, pre_h_m, pre_h_t, input);

    w_g_er += dot(input.T(), cur_g_er);
    if (!no_bias) {
      b_g_er += cur_g_er;
    }
    input_er = dot(cur_g_er, w_g_data.T());
    Tensor2D pre_h_l_er_tmp, pre_h_m_er_tmp, pre_h_t_er_tmp;
    split_input(input_er, 
                cur_x_er_tmp,
                pre_h_l_er_tmp,
                pre_h_m_er_tmp,
                pre_h_t_er_tmp);

    cur_x_er   += cur_x_er_tmp;
    pre_h_l_er += pre_h_l_er_tmp;
    pre_h_m_er += pre_h_m_er_tmp;
    pre_h_t_er += pre_h_t_er_tmp;
  }

  // x: (x_max_len, y_max_len, d_input)
  void ForwardLeftTop2RightBottom(Tensor3D x, int x_len, int y_len, 
                                  Tensor3D g, Tensor3D reset_h, 
                                  Tensor3D hi, Tensor3D h) {
    utils::Check(x_len > 0 && y_len > 0 && x_len <= x.size(0) && y_len <= x.size(1), "GruD2Layer: input size error. x_len: %d, y_len: %d, x.size: %d x %d", x_len, y_len, x.size(0), x.size(1));
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    // not need any padding, begin h and c are set to 0
    for (index_t row_idx = 0; row_idx < x_len; ++row_idx) {
      for (index_t col_idx = 0; col_idx < y_len; ++col_idx) {
        if (row_idx == 0) {
          pre_h_t = begin_h;
        } else {
          pre_h_t = h[row_idx-1].Slice(col_idx, col_idx+1);
        }
        if (col_idx == 0) {
          pre_h_l = begin_h;
        } else {
          pre_h_l = h[row_idx].Slice(col_idx-1, col_idx);
        }
        if (row_idx == 0 || col_idx == 0) {
          pre_h_m = begin_h;
        } else {
          pre_h_m = h[row_idx-1].Slice(col_idx-1, col_idx);
        }
        ForwardOneStep(pre_h_l, 
                       pre_h_m,
                       pre_h_t,
                       x[row_idx].Slice(col_idx, col_idx+1),
                       g[row_idx].Slice(col_idx, col_idx+1),
                       reset_h[row_idx].Slice(col_idx, col_idx+1),
                       hi[row_idx].Slice(col_idx, col_idx+1),
                       h[row_idx].Slice(col_idx, col_idx+1));
      }
    }
  }

  // x: (x_max_len, y_max_len, d_input)
  void ForwardRightBottom2LeftTop(Tensor3D x, int x_len, int y_len, 
                                  Tensor3D g, Tensor3D reset_h,
                                  Tensor3D hi, Tensor3D h) {
    utils::Check(x_len > 0 && y_len > 0 && x_len <= x.size(0) && y_len <= x.size(1), "GruD2Layer: input size error. x_len: %d, y_len: %d, x.size: %d x %d", x_len, y_len, x.size(0), x.size(1));
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    // not need any padding, begin h and c are set to 0
    for (int row_idx = x_len-1; row_idx >= 0; --row_idx) {
      for (int col_idx = y_len-1; col_idx >= 0; --col_idx) {
        if (row_idx == x_len-1) {
          pre_h_t = begin_h;
        } else {
          pre_h_t = h[row_idx+1].Slice(col_idx, col_idx+1);
        }
        if (col_idx == y_len-1) {
          pre_h_l = begin_h;
        } else {
          pre_h_l = h[row_idx].Slice(col_idx+1, col_idx+2);
        }
        if (row_idx == x_len-1 || col_idx == y_len-1) {
          pre_h_m = begin_h;
        } else {
          pre_h_m = h[row_idx+1].Slice(col_idx+1, col_idx+2);
        }

        ForwardOneStep(pre_h_l, 
                       pre_h_m,
                       pre_h_t,
                       x[row_idx].Slice(col_idx, col_idx+1),
                       g[row_idx].Slice(col_idx, col_idx+1),
                       reset_h[row_idx].Slice(col_idx, col_idx+1),
                       hi[row_idx].Slice(col_idx, col_idx+1),
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
                 bottom_len.size(1) == 2, "GruD2Layer: input length error.");
    top[0]->length = mshadow::expr::F<op::identity>(bottom[0]->length);

    top_data = 0.f; g = 0.f; reset_h = 0.f; hi = 0.f;
    high_resolution_clock::time_point b_time_1 = high_resolution_clock::now();
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      int x_len = bottom_len[batch_idx][0];
      int y_len = bottom_len[batch_idx][1];
      utils::Assert(x_len >= 0 && y_len >= 0, "GruD2Layer: sequence length error.");
      if (!reverse) {
        ForwardLeftTop2RightBottom(bottom_data[batch_idx],
                                   x_len, y_len,
                                   g[batch_idx], 
                                   reset_h[batch_idx],
                                   hi[batch_idx],
                                   top_data[batch_idx]);
      } else {
        ForwardRightBottom2LeftTop(bottom_data[batch_idx],
                                   x_len, y_len,
                                   g[batch_idx], 
                                   reset_h[batch_idx],
                                   hi[batch_idx],
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
                 Tensor2D &r_l, 
                 Tensor2D &r_m, 
                 Tensor2D &r_t, 
                 Tensor2D &z_i, 
                 Tensor2D &z_l, 
                 Tensor2D &z_m, 
                 Tensor2D &z_t) {
    utils::Check(g.size(0) == 1, "GruD2Layer: gate problem."); 

    r_l = Tensor2D(g.dptr_,             mshadow::Shape2(1, d_mem));
    r_m = Tensor2D(g.dptr_ + 1 * d_mem, mshadow::Shape2(1, d_mem));
    r_t = Tensor2D(g.dptr_ + 2 * d_mem, mshadow::Shape2(1, d_mem));
    z_i = Tensor2D(g.dptr_ + 3 * d_mem, mshadow::Shape2(1, d_mem));
    z_l = Tensor2D(g.dptr_ + 4 * d_mem, mshadow::Shape2(1, d_mem));
    z_m = Tensor2D(g.dptr_ + 5 * d_mem, mshadow::Shape2(1, d_mem));
    z_t = Tensor2D(g.dptr_ + 6 * d_mem, mshadow::Shape2(1, d_mem));
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
  
  void BackpropForLeftTop2RightBottomGru(int x_len,        int y_len,
                                         Tensor3D h,       Tensor3D h_er, 
                                         Tensor3D hi,      Tensor3D hi_er, 
                                         Tensor3D reset_h, Tensor3D reset_h_er, 
                                         Tensor3D g,       Tensor3D g_er, 
                                         Tensor3D x,       Tensor3D x_er) {
    Tensor2D cur_x, cur_g, cur_reset_h, cur_hi,  cur_h;
    Tensor2D cur_x_er, cur_g_er, cur_reset_h_er, cur_hi_er, cur_h_er; 
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    Tensor2D pre_h_l_er, pre_h_m_er, pre_h_t_er;
    for (int row_idx = x_len-1; row_idx >= 0; --row_idx) {
      for (int col_idx = y_len-1; col_idx >= 0; --col_idx) {
        if (row_idx == 0) {
          pre_h_t = begin_h;
          pre_h_t_er = begin_h_er;
        } else {
          pre_h_t = h[row_idx-1].Slice(col_idx, col_idx+1);
          pre_h_t_er = h_er[row_idx-1].Slice(col_idx, col_idx+1);
        }
        if (col_idx == 0) {
          pre_h_l = begin_h;
          pre_h_l_er = begin_h_er;
        } else {
          pre_h_l = h[row_idx].Slice(col_idx-1, col_idx);
          pre_h_l_er = h_er[row_idx].Slice(col_idx-1, col_idx);
        }
        if (row_idx == 0 || col_idx == 0) {
          pre_h_m = begin_h;
          pre_h_m_er = begin_h_er;
        } else {
          pre_h_m = h[row_idx-1].Slice(col_idx-1, col_idx);
          pre_h_m_er = h_er[row_idx-1].Slice(col_idx-1, col_idx);
        }
        cur_x          = x[row_idx].Slice(col_idx, col_idx+1);
        cur_g          = g[row_idx].Slice(col_idx, col_idx+1);
        cur_reset_h    = reset_h[row_idx].Slice(col_idx, col_idx+1);
        cur_hi         = hi[row_idx].Slice(col_idx, col_idx+1);
        cur_h          = h[row_idx].Slice(col_idx, col_idx+1);
        cur_x_er       = x_er[row_idx].Slice(col_idx, col_idx+1);
        cur_g_er       = g_er[row_idx].Slice(col_idx, col_idx+1);
        cur_reset_h_er = reset_h_er[row_idx].Slice(col_idx, col_idx+1);
        cur_hi_er      = hi_er[row_idx].Slice(col_idx, col_idx+1);
        cur_h_er       = h_er[row_idx].Slice(col_idx, col_idx+1);
        BpOneStep(cur_h_er,
                  pre_h_l,
                  pre_h_m,
                  pre_h_t,
                  cur_x,
                  cur_reset_h,
                  cur_g,
                  cur_hi,
                  cur_hi_er,
                  cur_g_er,
                  pre_h_l_er,
                  pre_h_m_er,
                  pre_h_t_er,
                  cur_x_er);
      }
    }
  }

  void BackpropForRightBottom2LeftTopGru(int x_len,        int y_len,
                                         Tensor3D h,       Tensor3D h_er, 
                                         Tensor3D hi,      Tensor3D hi_er, 
                                         Tensor3D reset_h, Tensor3D reset_h_er, 
                                         Tensor3D g,       Tensor3D g_er, 
                                         Tensor3D x,       Tensor3D x_er) {
    Tensor2D cur_x, cur_g, cur_reset_h, cur_hi,  cur_h;
    Tensor2D cur_x_er, cur_g_er, cur_reset_h_er, cur_hi_er, cur_h_er; 
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    Tensor2D pre_h_l_er, pre_h_m_er, pre_h_t_er;
    for (index_t row_idx = 0; row_idx < x_len; ++row_idx) {
      for (index_t col_idx = 0; col_idx < y_len; ++col_idx) {
        if (row_idx == x_len-1) {
          pre_h_t = begin_h;
          pre_h_t_er = begin_h_er;
        } else {
          pre_h_t = h[row_idx+1].Slice(col_idx, col_idx+1);
          pre_h_t_er = h_er[row_idx+1].Slice(col_idx, col_idx+1);
        }
        if (col_idx == y_len-1) {
          pre_h_l = begin_h;
          pre_h_l_er = begin_h_er;
        } else {
          pre_h_l = h[row_idx].Slice(col_idx+1, col_idx+2);
          pre_h_l_er = h_er[row_idx].Slice(col_idx+1, col_idx+2);
        }
        if (row_idx == x_len-1 || col_idx == y_len-1) {
          pre_h_m = begin_h;
          pre_h_m_er = begin_h_er;
        } else {
          pre_h_m = h[row_idx+1].Slice(col_idx+1, col_idx+2);
          pre_h_m_er = h_er[row_idx+1].Slice(col_idx+1, col_idx+2);
        }
        cur_x          = x[row_idx].Slice(col_idx, col_idx+1);
        cur_g          = g[row_idx].Slice(col_idx, col_idx+1);
        cur_reset_h    = reset_h[row_idx].Slice(col_idx, col_idx+1);
        cur_hi         = hi[row_idx].Slice(col_idx, col_idx+1);
        cur_h          = h[row_idx].Slice(col_idx, col_idx+1);
        cur_x_er       = x_er[row_idx].Slice(col_idx, col_idx+1);
        cur_g_er       = g_er[row_idx].Slice(col_idx, col_idx+1);
        cur_reset_h_er = reset_h_er[row_idx].Slice(col_idx, col_idx+1);
        cur_hi_er      = hi_er[row_idx].Slice(col_idx, col_idx+1);
        cur_h_er       = h_er[row_idx].Slice(col_idx, col_idx+1);
        BpOneStep(cur_h_er,
                  pre_h_l,
                  pre_h_m,
                  pre_h_t,
                  cur_x,
                  cur_reset_h,
                  cur_g,
                  cur_hi,
                  cur_hi_er,
                  cur_g_er,
                  pre_h_l_er,
                  pre_h_m_er,
                  pre_h_t_er,
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
    begin_h_er = 0.; g_er = 0.; reset_h_er = 0.; hi_er = 0.;
    for (index_t batch_idx = 0; batch_idx < x.size(0); ++batch_idx) {
      int x_len = len[batch_idx][0];
      int y_len = len[batch_idx][1];
      if (!reverse) {
        BackpropForLeftTop2RightBottomGru(x_len, y_len,
                                          h[batch_idx],
                                          h_er[batch_idx], 
                                          hi[batch_idx],
                                          hi_er[batch_idx],
                                          reset_h[batch_idx], 
                                          reset_h_er[batch_idx], 
                                          g[batch_idx],
                                          g_er[batch_idx],
                                          x[batch_idx],
                                          x_er[batch_idx]);
      } else {
        BackpropForRightBottom2LeftTopGru(x_len, y_len,
                                          h[batch_idx],
                                          h_er[batch_idx], 
                                          hi[batch_idx],
                                          hi_er[batch_idx],
                                          reset_h[batch_idx], 
                                          reset_h_er[batch_idx], 
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

 public:
// protected:
  // float max_norm2;
  int d_mem, d_input;
  bool no_bias, reverse, is_use_reset_gate, is_diag_connection; //, no_out_tanh; 
  // float grad_norm2;
  // float o_gate_bias_init;
  // float f_gate_bias_init;
  // float grad_cut_off;
  // string param_file;
  mshadow::TensorContainer<xpu, 4> hi, g, reset_h, hi_er, g_er, reset_h_er;
  mshadow::TensorContainer<xpu, 2> begin_h, begin_h_er;
  // clock_t time_1, time_2, time_3, time_4;
  duration<double> time_1, time_2, time_3, time_4;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LSTM_LAYER_INL_HPP_
