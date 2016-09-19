#ifndef TEXTNET_LAYER_BGRU_D2_LAYER_INL_HPP_
#define TEXTNET_LAYER_BGRU_D2_LAYER_INL_HPP_

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
class BGruD2Layer : public Layer<xpu> {
 public:
  BGruD2Layer(LayerType type) { this->layer_type = type; }
  virtual ~BGruD2Layer(void) { }
  
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "BGruD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "BGruD2Layer:top size problem.");

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

    this->params.resize(4);
    this->params[0].Resize(1, 1, 7*d_mem, d_input+3*d_mem, true); // w and u is in one matrix, gate 
    this->params[1].Resize(1, 1, 7*d_mem, 1, true); // b, gate
    this->params[2].Resize(1, 1, 1*d_mem, d_input+3*d_mem, true); // w and u is in one matrix, cc
    this->params[3].Resize(1, 1, 1*d_mem, 1, true); // b, cc
    
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
    utils::Check(bottom.size() == BottomNodeNum(), "BGruD2Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "BGruD2Layer:top size problem.");
    //high_resolution_clock::time_point b_time_2 = high_resolution_clock::now();
    
    nbatch = bottom[0]->data.size(0);
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;   // (batch size, x_len, y_len, d_mem)
    mshadow::Shape<4> shape_in1 = mshadow::Shape4(shape_in[1], shape_in[2], shape_in[0], shape_in[3]);
    mshadow::Shape<4> shape_in2 = mshadow::Shape4(shape_in[1], shape_in[2], shape_in[3], shape_in[0]);
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);
    mshadow::Shape<4> shape_out1 = mshadow::Shape4(shape_in[1], shape_in[2], shape_in[0], d_mem);
    mshadow::Shape<4> shape_out2 = mshadow::Shape4(shape_in[1], shape_in[2], d_mem, shape_in[0]);
    mshadow::Shape<4> shape_reset_h = mshadow::Shape4(shape_in[1], shape_in[2], d_mem*3, shape_in[0]);
    // input, forget * 3, output, candidate
    mshadow::Shape<4> shape_gate = mshadow::Shape4(shape_in[1], shape_in[2], d_mem*7, shape_in[0]); 

    top[0]->Resize(shape_out, mshadow::Shape2(shape_out[0],2), true);
    mask_data.Resize(shape_out2, 0.f); // mask_data : batch_size * max_length
    mask_diff.Resize(shape_in2, 0.f); // mask_diff: batch_size * max_length
    //mask_g.Resize(shape_gate, 0.f);
    //mask_reset_h.Resize(shape_reset_h, 0.f);
    rd_bottom_data.Resize(shape_in2, 0.f);
    rd_bottom_diff.Resize(shape_in2, 0.f);
    rd_top_data.Resize(shape_out2, 0.f);
    rd_top_diff.Resize(shape_out2, 0.f);
    sum.Resize(mshadow::Shape1(nbatch), 0.f);
    input.Resize(mshadow::Shape2(d_input + 3*d_mem, nbatch), 0.f);
    input_er.Resize(mshadow::Shape2(d_input + 3*d_mem, nbatch), 0.f);

    hi.Resize(shape_out2, 0.f); // h input
    hi_er.Resize(shape_out2, 0.f);
    g.Resize(shape_gate, 0.f);
    g_er.Resize(shape_gate, 0.f);
    reset_h.Resize(shape_reset_h, 0.f);
    //reset_h_er.Resize(shape_reset_h, 0.f);

    begin_h.Resize(mshadow::Shape2(d_mem, nbatch), 0.f);
    begin_h_er.Resize(mshadow::Shape2(d_mem, nbatch), 0.f);


    if(!no_bias){
      b_g_expand.Resize(mshadow::Shape2(nbatch, 7*d_mem), 0.f);
      b_c_expand.Resize(mshadow::Shape2(nbatch, d_mem), 0.f);
    }

    //high_resolution_clock::time_point e_time_2 = high_resolution_clock::now();
    //time_2 += duration_cast<duration<double>>(e_time_2 - b_time_2);
	  //utils::Printf("\tBGRU D2 Reshape Time:%fs\n", time_2.count()); 

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
    Tensor2D w_data = this->params[0].data[0][0]();
    Tensor2D w_diff = this->params[0].diff[0][0]();
    checkNan(w_data.dptr_, w_data.size());
    checkNan(w_diff.dptr_, w_diff.size());
  }
  void PrintTensor(const char * name, mshadow::Tensor<cpu, 4> x) {
    mshadow::Shape<4> s = x.shape_;
      cout << name << " shape " << s[0] << "x" << s[1] << "x" << s[2]<< "x" <<s[3]<<endl;
      for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
        for (unsigned int d2 = 0; d2 < s[1]; ++d2) {
          for (unsigned int d3 = 0; d3 < s[2]; ++d3) {
            for (unsigned int d4 = 0; d4 < s[3]; ++d4) {
              cout << x[d1][d2][d3][d4] << " ";
            }
            cout<<"|";
          }
          cout<<";";
        }
        cout << endl;
      }
      cout << endl;
  }
  void PrintTensor(const char * name, mshadow::Tensor<cpu, 2> x) {
    mshadow::Shape<2> s = x.shape_;
      cout << name << " shape " << s[0] << "x" << s[1] << endl;
      for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
        for (unsigned int d2 = 0; d2 < s[1]; ++d2) {
          cout << x[d1][d2] << " ";
        }
        cout << endl;
      }
      cout << endl;
  }
  void PrintTensor(const char * name, mshadow::Tensor<cpu, 1> x) {
    mshadow::Shape<1> s = x.shape_;
      cout << name << " shape " << s[0] << endl;
      for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
        cout << x[d1]<< " ";
      }
      cout << endl;
  }

  // this layer copy the input value to a continue memory
  void concat_input(Tensor2D x, Tensor2D h_l, Tensor2D h_m, Tensor2D h_t, Tensor2D input) {
    utils::Check(x.size(1) == h_l.size(1) && h_l.size(1) == h_m.size(1) &&
        h_m.size(1) == h_t.size(1) && h_t.size(1) == input.size(1), "BGruD2Layer:: concat_input wrong.");
    //mshadow::Tensor<xpu,1> input_d1(input.dptr_, mshadow::Shape1(input.size(0) * input.size(1)));

    int cnt = 0;
    input.Slice(cnt, cnt+x.size(0))   = mshadow::expr::F<op::identity>(x);
    cnt += x.size(0);
    input.Slice(cnt, cnt+h_l.size(0)) = mshadow::expr::F<op::identity>(h_l);
    cnt += h_l.size(0);
    input.Slice(cnt, cnt+h_m.size(0)) = mshadow::expr::F<op::identity>(h_m);
    cnt += h_m.size(0);
    input.Slice(cnt, cnt+h_t.size(0)) = mshadow::expr::F<op::identity>(h_t);
  }

  // this is different with concat_input(), this re-use the same memory
  // return several tensors with pointers to the inut tensor
  void split_input(Tensor2D t, Tensor2D &x, Tensor2D &h_l, Tensor2D &h_m, Tensor2D &h_t) {
    int msize = d_input * nbatch;
    x   = Tensor2D(t.dptr_,                   mshadow::Shape2(d_input, nbatch));
    h_l = Tensor2D(t.dptr_ + msize,         mshadow::Shape2(d_mem, nbatch));
    h_m = Tensor2D(t.dptr_ + msize + d_mem*nbatch, mshadow::Shape2(d_mem, nbatch));
    h_t = Tensor2D(t.dptr_ + msize + 2*d_mem*nbatch, mshadow::Shape2(d_mem, nbatch));
    utils::Check(t.size(1)==nbatch && t.size(0)==(d_input+3*d_mem) &&
        x.size(1) == nbatch && h_l.size(1) == nbatch && h_m.size(1)==nbatch && h_t.size(1) == nbatch, "BGruD2Layer: size error.");
    //printf("t: %d * %d\n",t.size(0),t.size(1));
    //printf("x: %d * %d\n",x.size(0),x.size(1));
    //printf("h_l: %d * %d\n",h_l.size(0),h_l.size(1));
    //printf("h_m: %d * %d\n",h_m.size(0),h_m.size(1));
    //printf("h_t: %d * %d\n",h_t.size(0),h_t.size(1));
  }

  void diff_softmax_z_no_diag(Tensor2D z_i,    Tensor2D z_l,    Tensor2D z_t, 
                              Tensor2D z_i_er, Tensor2D z_l_er, Tensor2D z_t_er) {
    int len = z_i_er.size(0);
    mshadow::TensorContainer<xpu, 2> z_i_er_tmp(mshadow::Shape2(len, nbatch));
    mshadow::TensorContainer<xpu, 2> z_l_er_tmp(mshadow::Shape2(len, nbatch));
    mshadow::TensorContainer<xpu, 2> z_t_er_tmp(mshadow::Shape2(len, nbatch));
    //mshadow::TensorContainer<xpu, 1> error_sum(mshadow::Shape1(nbatch));
    z_i_er_tmp = mshadow::expr::F<op::identity>(z_i_er);
    z_l_er_tmp = mshadow::expr::F<op::identity>(z_l_er);
    // z_m_er_tmp = mshadow::expr::F<op::identity>(z_m_er);
    z_t_er_tmp = mshadow::expr::F<op::identity>(z_t_er);
    for (int i = 0; i < len; ++i) {
      sum = 0.f;
      sum += z_i_er_tmp[i] * z_i[i];
      sum += z_l_er_tmp[i] * z_l[i];
      sum += z_t_er_tmp[i] * z_t[i];

      z_i_er[i] = (z_i_er_tmp[i] - sum) * z_i[i];
      z_l_er[i] = (z_l_er_tmp[i] - sum) * z_l[i];
      z_t_er[i] = (z_t_er_tmp[i] - sum) * z_t[i];
    }
  }
  
  void softmax_z_no_diag(Tensor2D z_i, Tensor2D z_l, Tensor2D z_t) {
    z_i = mshadow::expr::F<op::orc_exp>(z_i); //dmem * nbatch
    z_l = mshadow::expr::F<op::orc_exp>(z_l); //dmem * nbatch
    // z_m = mshadow::expr::F<op::orc_exp>(z_m);
    z_t = mshadow::expr::F<op::orc_exp>(z_t); //dmem * nbatch
    //mshadow::TensorContainer<xpu, 1> sum(mshadow::Shape1(nbatch));

    // normalize by col
    for (int i = 0; i < z_i.size(0); ++i) {
      sum = 0.f;
      sum += z_i[i];
      sum += z_l[i];
      sum += z_t[i];

      z_i[i] /= sum;
      z_l[i] /= sum;
      z_t[i] /= sum;
    }
  }


  void diff_softmax_z(Tensor2D z_i, Tensor2D z_l, Tensor2D z_m, Tensor2D z_t, 
                      Tensor2D z_i_er, Tensor2D z_l_er, Tensor2D z_m_er, Tensor2D z_t_er) {
    utils::Check(z_i.size(1) == nbatch && z_l.size(1) == nbatch && z_m.size(1) == nbatch && z_t.size(1) == nbatch &&
        z_i_er.size(1) == nbatch && z_l_er.size(1) == nbatch && z_m_er.size(1) == nbatch && z_t_er.size(1) == nbatch,
        " BGruD2Layer:: diff_softmax_z dimension 2 size wrong.");
    int len = z_i_er.size(0);
    mshadow::TensorContainer<xpu, 2> z_i_er_tmp(mshadow::Shape2(len, nbatch));
    mshadow::TensorContainer<xpu, 2> z_l_er_tmp(mshadow::Shape2(len, nbatch));
    mshadow::TensorContainer<xpu, 2> z_m_er_tmp(mshadow::Shape2(len, nbatch));
    mshadow::TensorContainer<xpu, 2> z_t_er_tmp(mshadow::Shape2(len, nbatch));
    // Tensor2D tmp_1 = z_i_er_tmp;
    // Tensor2D tmp_2 = z_l_er_tmp;
    // Tensor2D tmp_3 = z_m_er_tmp;
    // Tensor2D tmp_4 = z_t_er_tmp;
    // utils::Check(false, "tmp");
    //mshadow::TensorContainer<xpu, 1> error_sum(mshadow::Shape1(nbatch));
    z_i_er_tmp = mshadow::expr::F<op::identity>(z_i_er);
    z_l_er_tmp = mshadow::expr::F<op::identity>(z_l_er);
    z_m_er_tmp = mshadow::expr::F<op::identity>(z_m_er);
    z_t_er_tmp = mshadow::expr::F<op::identity>(z_t_er);
    for (int i = 0; i < len; ++i) {
      sum = 0.f;
      sum += z_i_er_tmp[i] * z_i[i];
      sum += z_l_er_tmp[i] * z_l[i];
      sum += z_m_er_tmp[i] * z_m[i];
      sum += z_t_er_tmp[i] * z_t[i];

      z_i_er[i] = (z_i_er_tmp[i] - sum) * z_i[i];
      z_l_er[i] = (z_l_er_tmp[i] - sum) * z_l[i];
      z_m_er[i] = (z_m_er_tmp[i] - sum) * z_m[i];
      z_t_er[i] = (z_t_er_tmp[i] - sum) * z_t[i];
    }
  }
  
  void softmax_z(Tensor2D z_i, Tensor2D z_l, Tensor2D z_m, Tensor2D z_t) {
    z_i = mshadow::expr::F<op::orc_exp>(z_i); // dmem * nbatch
    z_l = mshadow::expr::F<op::orc_exp>(z_l); // dmem * nbatch
    z_m = mshadow::expr::F<op::orc_exp>(z_m); // dmem * nbatch
    z_t = mshadow::expr::F<op::orc_exp>(z_t); // dmem * nbatch
    //mshadow::TensorContainer<xpu, 1> sum(mshadow::Shape1(nbatch));

    // normalize by col
    for (int i = 0; i < z_i.size(0); ++i) {
      sum = 0.f;
      sum += z_i[i];
      sum += z_l[i];
      sum += z_m[i];
      sum += z_t[i];

      z_i[i] /= sum;
      z_l[i] /= sum;
      z_m[i] /= sum;
      z_t[i] /= sum;
    }
  }

  virtual void ForwardOneStep(Tensor2D pre_h_l, // left   dmem * nbatch
                              Tensor2D pre_h_m, // middle dmem * nbatch
                              Tensor2D pre_h_t, // top    dmem * nbatch
                              Tensor2D cur_x,       //    dinput * nbatch
                              Tensor2D cur_g,      //     7dmem * nbatch
                              Tensor2D reset_h,    //     3dmem * nbatch
                              Tensor2D cur_hi,     //     dmem * nbatch
                              Tensor2D cur_h) {    //     dmem * nbatch
    //utils::Check(cur_x.size(0) == 1, "BGruD2Layer: ForwardOneStep(): input size error.");
    Tensor2D w_g_data = this->params[0].data[0][0]; // ( 7 * d_mem, dinput + 3dmem) 
    Tensor2D b_g_data = this->params[1].data[0][0]; // ( 7dem, 1)
    Tensor2D w_c_data = this->params[2].data[0][0]; // ( dmem, dinput + 3dmem)
    Tensor2D b_c_data = this->params[3].data[0][0]; // ( dmem, 1)

    //mshadow::TensorContainer<xpu, 2> input(mshadow::Shape2(d_input + 3*d_mem, nbatch));
    concat_input(cur_x, pre_h_l, pre_h_m, pre_h_t, input); // input: dinput + 3dmem * nbatch

    cur_g = dot(w_g_data, input); // cur_g : ( 7dmem ,nbatch ) , w_g_data:(7dmem, dinput+3dmem), input( dinput+3dmem, nbatch);
    if (!no_bias) {
      cur_g += b_g_expand.T();
    }

    Tensor2D r_l, r_m, r_t, z_i, z_l, z_m, z_t;
    SplitGate(cur_g, r_l, r_m, r_t, z_i, z_l, z_m, z_t);
    r_l = mshadow::expr::F<op::sigmoid_lookup>(r_l); // logi
    r_m = mshadow::expr::F<op::sigmoid_lookup>(r_m); // logi
    r_t = mshadow::expr::F<op::sigmoid_lookup>(r_t); // logi

    Tensor2D reset_h_l(reset_h.dptr_,         mshadow::Shape2(d_mem, nbatch));
    Tensor2D reset_h_m(reset_h.dptr_+d_mem * nbatch,   mshadow::Shape2(d_mem, nbatch));
    Tensor2D reset_h_t(reset_h.dptr_+2 * d_mem * nbatch, mshadow::Shape2(d_mem, nbatch));
    if (is_use_reset_gate) {
      reset_h_l = r_l * pre_h_l; // dmem * nbatch
      reset_h_m = r_m * pre_h_m; // dmem * nbatch
      reset_h_t = r_t * pre_h_t; // dmem * nbatch
    } else {
      reset_h_l = mshadow::expr::F<op::identity>(pre_h_l);
      reset_h_m = mshadow::expr::F<op::identity>(pre_h_m);
      reset_h_t = mshadow::expr::F<op::identity>(pre_h_t);
    }

    concat_input(cur_x, reset_h_l, reset_h_m, reset_h_t, input);
    cur_hi = dot(w_c_data, input); // cur_hi:(dmem, nbatch), w_c_data:(dmem, dinput+3dmem)
    if (!no_bias) {
      cur_hi += b_c_expand.T();
    }
    
    cur_hi = mshadow::expr::F<op::tanh_lookup>(cur_hi);

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

  void BpOneStep(Tensor2D cur_h_er, //  dmem * nbatch
                 Tensor2D pre_h_l,  // dmem * nbatch
                 Tensor2D pre_h_m,  // dmem * nbatch
                 Tensor2D pre_h_t,  // dmem * nbatch
                 Tensor2D cur_x,    // dinput * nbatch
                 Tensor2D cur_reset_h, // 3dmem * nbatch
                 Tensor2D cur_g,     // 7dmem * nbatch
                 Tensor2D cur_hi,    // dmem * nbatch
                 Tensor2D cur_hi_er,  //dmem * nbatch
                 Tensor2D cur_g_er,   // 7dmem * nbatch
                 Tensor2D pre_h_l_er, // dmem * nbatch
                 Tensor2D pre_h_m_er, // dmem * nbatch
                 Tensor2D pre_h_t_er, // dmem * nbatch
                 Tensor2D cur_x_er) {  // dinput * nbatch
    Tensor2D w_g_data = this->params[0].data[0][0];  //(7dmem, dinput+3dmem)
    Tensor2D w_g_er   = this->params[0].diff[0][0];
    Tensor2D b_g_er   = this->params[1].diff[0][0];  // (7dmem,1)
    Tensor2D w_c_data = this->params[2].data[0][0]; // ( dmem, dinput + 3dmem)
    Tensor2D w_c_er   = this->params[2].diff[0][0];
    Tensor2D b_c_er   = this->params[3].diff[0][0]; // (dmem, 1)
    Tensor2D r_l, r_m, r_t, z_i, z_l, z_m, z_t;  // dmem * nabtch
    Tensor2D r_l_er, r_m_er, r_t_er, z_i_er, z_l_er, z_m_er, z_t_er; // dmem * nbatch
    SplitGate(cur_g, r_l, r_m, r_t, z_i, z_l, z_m, z_t);
    SplitGate(cur_g_er, r_l_er, r_m_er, r_t_er, z_i_er, z_l_er, z_m_er, z_t_er);
    
    z_i_er = cur_h_er * cur_hi;  //dmem * nbatch
    z_l_er = cur_h_er * pre_h_l;  //dmem * nbatch
    if (is_diag_connection) {
      z_m_er = cur_h_er * pre_h_m;  // dmem * nbatch
    }
    z_t_er = cur_h_er * pre_h_t;  //dmem * nbatch

    cur_hi_er  = mshadow::expr::F<op::tanh_grad>(cur_hi) * (cur_h_er * z_i); //dmem *  nbatch
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

    //mshadow::TensorContainer<xpu, 2> input(mshadow::Shape2(d_input + 3*d_mem, nbatch));
    //mshadow::TensorContainer<xpu, 2> input_er(mshadow::Shape2(d_input + 3*d_mem, nbatch));
    Tensor2D reset_h_l(cur_reset_h.dptr_,         mshadow::Shape2(d_mem, nbatch));
    Tensor2D reset_h_m(cur_reset_h.dptr_+d_mem*nbatch,   mshadow::Shape2(d_mem, nbatch));
    Tensor2D reset_h_t(cur_reset_h.dptr_+2*d_mem*nbatch, mshadow::Shape2(d_mem, nbatch));
    input = 0.f;
    concat_input(cur_x, reset_h_l, reset_h_m, reset_h_t, input);

    // w_c_er: (dmem , input + 3dmem),  cur_hi_er: dmem * nbatch 
    // input;  ( dinput + 3dmem, nbatch)
    w_c_er += dot(cur_hi_er, input.T()); // curr_hi_er: dmem * nbatch 
    if (!no_bias) {
      this->params[3].diff_d1() += sum_rows(cur_hi_er.T());
    }
    // w_c_data:( dmem, input+3dmem);  
    input_er = dot(w_c_data.T(), cur_hi_er);
    Tensor2D cur_x_er_tmp, reset_h_l_er, reset_h_m_er, reset_h_t_er;
    split_input(input_er, cur_x_er_tmp, reset_h_l_er, reset_h_m_er, reset_h_t_er);
    cur_x_er += cur_x_er_tmp; // input * nbatch

    if (is_use_reset_gate) {
      r_l_er = mshadow::expr::F<op::sigmoid_grad>(r_l) * (reset_h_l_er * pre_h_l); //dmem * nbatch
      r_m_er = mshadow::expr::F<op::sigmoid_grad>(r_m) * (reset_h_m_er * pre_h_m); //dmem * nbatch
      r_t_er = mshadow::expr::F<op::sigmoid_grad>(r_t) * (reset_h_t_er * pre_h_t); //dmem * nbatch
      pre_h_l_er += reset_h_l_er * r_l; // dmem * nbatch
      pre_h_m_er += reset_h_m_er * r_m; // dmem * nbatch
      pre_h_t_er += reset_h_t_er * r_t; // dmem * nbatch
    } else {
      pre_h_l_er += reset_h_l_er; // dmem * nbatch
      pre_h_m_er += reset_h_m_er; // dmem * nbatch
      pre_h_t_er += reset_h_t_er; // dmem * nbatch
    }

    input = 0.f;
    concat_input(cur_x, pre_h_l, pre_h_m, pre_h_t, input);

    // cur_g_er: (7dmem, nbatch), input:(dinput+3dmem, nbatch)
    //w_g_er: (7dmem, dinput+3dmem);
    w_g_er += dot(cur_g_er, input.T());
    if (!no_bias) {
      this->params[1].diff_d1() += sum_rows(cur_g_er.T());
    }
    //cur_g_er:( 7dmem, nbatch), w_g_data:(7dmem, dinput+3dmem),
    input_er = dot(w_g_data.T(), cur_g_er); // input_er: ( dinput + 3dmem, nbatch)
    Tensor2D pre_h_l_er_tmp, pre_h_m_er_tmp, pre_h_t_er_tmp;
    split_input(input_er, cur_x_er_tmp, pre_h_l_er_tmp, pre_h_m_er_tmp, pre_h_t_er_tmp);

    cur_x_er   += cur_x_er_tmp;
    pre_h_l_er += pre_h_l_er_tmp;
    pre_h_m_er += pre_h_m_er_tmp;
    pre_h_t_er += pre_h_t_er_tmp;
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
// #if DEBUG
//     checkNanParams();
// #endif
    //high_resolution_clock::time_point b_time_1 = high_resolution_clock::now();
    using namespace mshadow::expr;
    Tensor4D bottom_data = bottom[0]->data;
    Tensor2D bottom_len  = bottom[0]->length;
    Tensor4D top_data    = top[0]->data;

    utils::Check(bottom_len.size(0) == bottom_data.size(0) && 
                 bottom_len.size(1) == 2, "BGruD2Layer: input length error.");
    top[0]->length = mshadow::expr::F<op::identity>(bottom[0]->length);

    mask_data = 0.f; //mask_g = 0.f; mask_reset_h = 0.f;
    rd_top_data = 0.f; rd_bottom_data=0.f;b_g_expand = 0.f, b_c_expand=0.f;
    top_data = 0.f; g = 0.f; reset_h = 0.f; hi = 0.f;
    Tensor2D pre_h_l, pre_h_m, pre_h_t;
    if(!no_bias){
      b_g_expand = repmat(this->params[1].data_d1(),nbatch); // expand bias to (nbatch * dmem)
      b_c_expand = repmat(this->params[3].data_d1(),nbatch); // expand bias to (nbatch * 7dmem)
    }
    int x_steps = rd_bottom_data.size(0);
    int y_steps = rd_bottom_data.size(1);

    //rd_bottom_data = swapaxis<3,2>(swapaxis<2,1>(swapaxis<1,0>(bottom_data))); //slow
    for(index_t bid = 0 ; bid < nbatch; ++ bid){
      for(index_t idx = 0 ; idx < bottom[0]->length[bid][0]; ++ idx){
        for(index_t idy = 0 ; idy < bottom[0]->length[bid][1]; ++ idy){
          for(index_t idim = 0 ; idim < d_input; ++ idim){
            rd_bottom_data[idx][idy][idim][bid] = bottom_data[bid][idx][idy][idim];
          }
          for(index_t idim = 0 ; idim < d_mem; ++ idim){
            mask_data[idx][idy][idim][bid] = 1.f;
          }
          /*
          for(index_t idim = 0 ; idim < 7 * d_mem; ++ idim){
            mask_g[idx][idy][idim][bid] = 1.f;
          }
          for(index_t idim = 0 ; idim < 3*d_mem; ++ idim){
            mask_reset_h[idx][idy][idim][bid] = 1.f;
          }
          */
        }
      }
    }

    if(!reverse){
      for(index_t idx = 0; idx < x_steps; ++ idx){
        for(index_t idy = 0 ; idy < y_steps; ++ idy){
          if( 0 == idx) pre_h_t = begin_h;
          else  pre_h_t = rd_top_data[idx-1][idy];
          if( 0 == idy) pre_h_l = begin_h;
          else  pre_h_l = rd_top_data[idx][idy-1];
          if( 0 == idx || 0 == idy )  pre_h_m = begin_h;
          else  pre_h_m = rd_top_data[idx-1][idy-1];
          ForwardOneStep(pre_h_l, pre_h_m, pre_h_t,
              rd_bottom_data[idx][idy], g[idx][idy], reset_h[idx][idy],
              hi[idx][idy], rd_top_data[idx][idy]);
        }
      }
    } else {
      for(int idx = x_steps - 1; idx >= 0; --idx){
        for(int idy = y_steps - 1; idy >= 0; --idy){
          if( x_steps - 1 == idx) pre_h_t = begin_h;
          else    pre_h_t = rd_top_data[idx+1][idy];
          if(y_steps - 1 == idy)  pre_h_l = begin_h;
          else    pre_h_l = rd_top_data[idx][idy+1];
          if(x_steps - 1 == idx || y_steps - 1 == idy)  pre_h_m = begin_h;
          else    pre_h_m = rd_top_data[idx+1][idy+1];
          ForwardOneStep(pre_h_l, pre_h_m, pre_h_t,
              rd_bottom_data[idx][idy], g[idx][idy], reset_h[idx][idy],
              hi[idx][idy], rd_top_data[idx][idy]);
        }
      }
    }

    rd_top_data = rd_top_data * mask_data;
    //hi = hi * mask_data;
    //g = g * mask_g;
    //reset_h = reset_h * mask_reset_h;

    //top_data = swapaxis<1,0>(swapaxis<2,1>(swapaxis<3,2>(rd_top_data))); // slow
    for(index_t bid = 0 ; bid < nbatch; ++ bid){
      for(index_t idx = 0 ; idx < bottom_data.size(1); ++ idx){
        for(index_t idy = 0 ; idy < bottom_data.size(2); ++ idy){
          for(index_t idim = 0 ; idim < d_mem; ++ idim){
            top_data[bid][idx][idy][idim] = rd_top_data[idx][idy][idim][bid];
          }
        }
      }
    }

    //high_resolution_clock::time_point e_time_1 = high_resolution_clock::now();
    //time_1 += duration_cast<duration<double>>(e_time_1 - b_time_1);
	 //utils::Printf("\tBGRU D2 Time:%fs,%fs,%f\n", time_1.count(), time_2.count(), time_3.count()); 
// #if DEBUG
//     checkNanParams();
// #endif
  }

  // too tricky, may bring errors
  void SplitGate(Tensor2D &g,  // 7dmem * nbatch
                 Tensor2D &r_l, 
                 Tensor2D &r_m, 
                 Tensor2D &r_t, 
                 Tensor2D &z_i, 
                 Tensor2D &z_l, 
                 Tensor2D &z_m, 
                 Tensor2D &z_t) {
    utils::Check(g.size(0) == 7 * d_mem && g.size(1) == nbatch, "BGruD2Layer:: SplitGate dimension wrong.");
    int msize = d_mem * nbatch;
    r_l = Tensor2D(g.dptr_,             mshadow::Shape2(d_mem, nbatch));
    r_m = Tensor2D(g.dptr_ + 1 * msize, mshadow::Shape2(d_mem, nbatch));
    r_t = Tensor2D(g.dptr_ + 2 * msize, mshadow::Shape2(d_mem, nbatch));
    z_i = Tensor2D(g.dptr_ + 3 * msize, mshadow::Shape2(d_mem, nbatch));
    z_l = Tensor2D(g.dptr_ + 4 * msize, mshadow::Shape2(d_mem, nbatch));
    z_m = Tensor2D(g.dptr_ + 5 * msize, mshadow::Shape2(d_mem, nbatch));
    z_t = Tensor2D(g.dptr_ + 6 * msize, mshadow::Shape2(d_mem, nbatch));
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
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    high_resolution_clock::time_point b_time_4 = high_resolution_clock::now();
// #if DEBUG
//     checkNanParams();
// #endif
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
        
    int x_steps = rd_bottom_diff.size(0);
    int y_steps = rd_bottom_diff.size(1);
    begin_h_er = 0.; g_er = 0.; hi_er = 0.;
    mask_diff = 0.f; rd_top_diff = 0.f; rd_bottom_diff = 0.f;
    Tensor2D pre_h_l, pre_h_m, pre_h_t, pre_h_l_er, pre_h_m_er, pre_h_t_er;

    //rd_top_diff = swapaxis<3,2>(swapaxis<2,1>(swapaxis<1,0>(top_diff)));
    for(index_t bid = 0 ; bid < nbatch; ++ bid){
      for(index_t idx = 0 ; idx < top_diff.size(1); ++ idx){
        for(index_t idy = 0 ; idy < top_diff.size(2); ++ idy){
          for(index_t idim = 0 ; idim < d_mem ; ++ idim){
            rd_top_diff[idx][idy][idim][bid] = top_diff[bid][idx][idy][idim];
          }
        }
      }
      for(index_t idx = 0 ; idx < bottom[0]->length[bid][0]; ++ idx){
        for(index_t idy = 0 ; idy < bottom[0]->length[bid][1]; ++ idy){
          for(index_t idim = 0 ; idim < d_input; ++ idim){
            mask_diff[idx][idy][idim][bid] = 1.f;
          }
        }
      }
    }
    if(!reverse){
      for(int idx = x_steps - 1; idx >= 0 ; -- idx){
        for(int idy = y_steps - 1; idy >= 0 ; -- idy){
          if( 0 == idx){
            pre_h_t = begin_h;
            pre_h_t_er = begin_h_er;
          } else {
            pre_h_t = rd_top_data[idx - 1][idy];
            pre_h_t_er = rd_top_diff[idx - 1][idy];
          }
          if( 0 == idy){
            pre_h_l = begin_h;
            pre_h_l_er = begin_h_er;
          } else {
            pre_h_l = rd_top_data[idx][idy - 1];
            pre_h_l_er = rd_top_diff[idx][idy - 1];
          }
          if( 0 == idx || 0 == idy){
            pre_h_m = begin_h;
            pre_h_m_er = begin_h_er;
          } else {
            pre_h_m = rd_top_data[idx - 1][idy - 1];
            pre_h_m_er = rd_top_diff[idx - 1][idy - 1];
          }
          BpOneStep(rd_top_diff[idx][idy], pre_h_l, pre_h_m, pre_h_t,
            rd_bottom_data[idx][idy], reset_h[idx][idy], g[idx][idy],
            hi[idx][idy], hi_er[idx][idy], g_er[idx][idy],
            pre_h_l_er, pre_h_m_er, pre_h_t_er,rd_bottom_diff[idx][idy]);
        }
      }
    } else {
      for(int idx = 0 ; idx < x_steps; ++ idx){
        for(int idy = 0 ; idy < y_steps; ++ idy){
          if( x_steps - 1 == idx){
            pre_h_t = begin_h;
            pre_h_t_er = begin_h_er;
          } else {
            pre_h_t = rd_top_data[idx + 1][idy];
            pre_h_t_er = rd_top_diff[idx + 1][idy];
          }
          if( y_steps - 1 == idy){
            pre_h_l = begin_h;
            pre_h_l_er = begin_h_er;
          } else {
            pre_h_l = rd_top_data[idx][idy + 1];
            pre_h_l_er = rd_top_diff[idx][idy + 1];
          }
          if( x_steps - 1 == idx || y_steps - 1 == idy){
            pre_h_m = begin_h;
            pre_h_m_er = begin_h_er;
          } else {
            pre_h_m = rd_top_data[idx + 1][idy + 1];
            pre_h_m_er = rd_top_diff[idx + 1][idy + 1];
          }
          BpOneStep(rd_top_diff[idx][idy], pre_h_l, pre_h_m, pre_h_t,
            rd_bottom_data[idx][idy], reset_h[idx][idy], g[idx][idy],
            hi[idx][idy], hi_er[idx][idy], g_er[idx][idy],
            pre_h_l_er, pre_h_m_er, pre_h_t_er,rd_bottom_diff[idx][idy]);
        }
      }
    }


    rd_bottom_diff = rd_bottom_diff * mask_diff;
    //high_resolution_clock::time_point s_time_00 = high_resolution_clock::now();
    for(index_t bid = 0 ; bid < nbatch; ++ bid){
      for(index_t idx = 0 ; idx < top_diff.size(1); ++ idx){
        for(index_t idy = 0 ; idy < top_diff.size(2); ++ idy){
          for(index_t idim = 0 ; idim < d_input ; ++ idim){
            bottom_diff[bid][idx][idy][idim] = rd_bottom_diff[idx][idy][idim][bid];
          }
        }
      }
    }
    //high_resolution_clock::time_point m_time_00 = high_resolution_clock::now();
    //bottom_diff = swapaxis<1,0>(swapaxis<2,1>(swapaxis<3,2>(rd_bottom_diff))); // slow
    //high_resolution_clock::time_point e_time_00 = high_resolution_clock::now();
    //duration<double> time_00 = duration_cast<duration<double>>(m_time_00 - s_time_00);
    //duration<double> time_01 = duration_cast<duration<double>>(e_time_00 - m_time_00);
	  //utils::Printf("\t BGRU BP Time for vs swap:%fs|%fs\n", time_00.count(),time_01.count()); 
    //bottom_diff *= mask_diff;
    //high_resolution_clock::time_point e_time_4 = high_resolution_clock::now();
    //time_4 += duration_cast<duration<double>>(e_time_4 - b_time_4);
	  //utils::Printf("\t BGRU D2 BP Time:%fs\n", time_4.count()); 
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
  int d_mem, d_input, nbatch;
  bool no_bias, reverse, is_use_reset_gate, is_diag_connection; //, no_out_tanh; 
  // float grad_norm2;
  // float o_gate_bias_init;
  // float f_gate_bias_init;
  // float grad_cut_off;
  // string param_file;
  mshadow::TensorContainer<xpu, 4> mask_data; // mask_g, mask_reset_h, mask_diff, rd_bottom_data, rd_top_data, rd_bottom_diff, rd_top_diff; 
  mshadow::TensorContainer<xpu, 4> mask_diff, rd_bottom_data, rd_top_data, rd_bottom_diff, rd_top_diff; 
  mshadow::TensorContainer<xpu, 4> hi, g, reset_h, hi_er, g_er; //reset_h_er;
  mshadow::TensorContainer<xpu, 2> b_g_expand, b_c_expand, begin_h, begin_h_er, input, input_er;
  mshadow::TensorContainer<xpu, 1> sum;
  // clock_t time_1, time_2, time_3, time_4;
  duration<double> time_1, time_2, time_3, time_4;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LSTM_LAYER_INL_HPP_
