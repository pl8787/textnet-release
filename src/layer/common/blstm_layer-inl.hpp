#ifndef TEXTNET_LAYER_BLSTM_LAYER_INL_HPP_
#define TEXTNET_LAYER_BLSTM_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
//#include "../../utils/utils.h"
//#include "../../io/json/json.h"
//#include <cassert>

namespace textnet {
namespace layer {

template<typename xpu>
class BLstmLayer : public Layer<xpu> {
 public:
  BLstmLayer(LayerType type) { this->layer_type = type; }
  virtual ~BLstmLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 3; }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);
    this->defaults["no_out_tanh"] = SettingV(false);
    this->defaults["param_file"] = SettingV("");
    this->defaults["o_gate_bias_init"] = SettingV(0.f);
    this->defaults["f_gate_bias_init"] = SettingV(0.f);
    // this->defaults["reverse"] = SettingV(false);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["max_norm2"] = SettingV();
    this->defaults["grad_norm2"] = SettingV();
    this->defaults["d_mem"] = SettingV();
    //this->defaults["d_input"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["u_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["u_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    this->defaults["reverse"] = SettingV();
    this->defaults["grad_cut_off"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(), "BLstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "BLstmLayer:top size problem.");
                  
    d_mem   = setting["d_mem"].iVal();
    //d_input   = setting["d_input"].iVal();
    d_input = bottom[0]->data.size(3);
    no_bias = setting["no_bias"].bVal();
    no_out_tanh = setting["no_out_tanh"].bVal();
    reverse = setting["reverse"].bVal();
    grad_norm2 = setting["grad_norm2"].fVal();
    param_file = setting["param_file"].sVal();
    o_gate_bias_init = setting["o_gate_bias_init"].fVal();
    f_gate_bias_init = setting["f_gate_bias_init"].fVal();
    grad_cut_off = setting["grad_cut_off"].fVal();
    max_norm2 = setting["max_norm2"].fVal();

    this->params.resize(3);
    this->params[0].Resize(1, 1, 4*d_mem, d_input, true); // w
    this->params[1].Resize(1, 1, 4*d_mem, d_mem, true); // u
    this->params[2].Resize(1, 1, 4*d_mem,    1, true); // b
    
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
    if (f_gate_bias_init != 0.f) {
        init_f_gate_bias(); // this must be after init()
    }
    if (o_gate_bias_init != 0.f) {
        init_o_gate_bias(); // this must be after init()
    }

    if (!param_file.empty()) {
      LoadParam();
    }
    
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

  // if want to capture long term dependency, should init as a positive value
  void init_f_gate_bias() {
    Tensor1D bias_data = this->params[2].data_d1();
    Tensor1D f_bias = Tensor1D(bias_data.dptr_ + 1*d_mem, mshadow::Shape1(d_mem));
    f_bias = f_gate_bias_init;
  }

  void init_o_gate_bias() {
    Tensor1D bias_data = this->params[2].data_d1();
    Tensor1D o_bias = Tensor1D(bias_data.dptr_ + 2*d_mem, mshadow::Shape1(d_mem));
    o_bias = o_gate_bias_init;
  }
  
  // bottom should be padded with only one zero on both sides
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "BLstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "BLstmLayer:top size problem.");
      //utils::ShowMemoryUse();
    
    nbatch = bottom[0]->data.size(0);
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_in1  = mshadow::Shape4(shape_in[2], shape_in[1], shape_in[0], shape_in[3]);
    mshadow::Shape<4> shape_in2  = mshadow::Shape4(shape_in[2], shape_in[1], shape_in[3], shape_in[0]);
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);
    mshadow::Shape<4> shape_out1 = mshadow::Shape4(shape_in[2], shape_in[1], shape_in[0], d_mem);
    mshadow::Shape<4> shape_out2 = mshadow::Shape4(shape_in[2], shape_in[1], d_mem, shape_in[0]);
    mshadow::Shape<4> shape_gate= mshadow::Shape4(shape_in[2], shape_in[1], 4*d_mem, shape_in[0]);

    top[0]->Resize(shape_out, true);
    mask_data.Resize(shape_out2, 0.f); // mask_data : batch_size * max_length
    mask_diff.Resize(shape_in2, 0.f); // mask_diff: batch_size * max_length
    rd_bottom_data_tmp.Resize(shape_in1);
    rd_bottom_data.Resize(shape_in2);
    rd_bottom_diff_tmp.Resize(shape_in1);
    rd_bottom_diff.Resize(shape_in2);
    rd_top_data_tmp.Resize(shape_out1);
    rd_top_data.Resize(shape_out2);
    rd_top_diff_tmp.Resize(shape_out1);
    rd_top_diff.Resize(shape_out2);
      //utils::ShowMemoryUse();
    c.Resize(shape_out2, 0.f);
      //utils::ShowMemoryUse();
    g.Resize(shape_gate, 0.f);
      //utils::ShowMemoryUse();
    c_er.Resize(shape_out2, 0.f);
      //utils::ShowMemoryUse();
    g_er.Resize(shape_gate, 0.f);
      //utils::ShowMemoryUse();

    begin_h.Resize(mshadow::Shape2(d_mem, nbatch), 0.f);
    begin_c.Resize(mshadow::Shape2(d_mem, nbatch), 0.f);
    begin_h_er.Resize(mshadow::Shape2(d_mem, nbatch), 0.f);
    begin_c_er.Resize(mshadow::Shape2(d_mem, nbatch), 0.f);

    b_expand.Resize(mshadow::Shape2(nbatch, 4*d_mem), 0.f);
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
      Tensor2D w_data = this->params[0].data[0][0];
      Tensor2D u_data = this->params[1].data[0][0];
      Tensor2D w_diff = this->params[0].diff[0][0];
      Tensor2D u_diff = this->params[1].diff[0][0];
      checkNan(w_data.dptr_, w_data.size(0) * w_data.size(1));
      checkNan(u_data.dptr_, u_data.size(0) * u_data.size(1));
      checkNan(w_diff.dptr_, w_diff.size(0) * w_diff.size(1));
      checkNan(u_diff.dptr_, u_diff.size(0) * u_diff.size(1));
  }

  virtual void ForwardOneStep(Tensor2D pre_c, 
                              Tensor2D pre_h,
                              Tensor2D x,
                              Tensor2D cur_g,
                              Tensor2D cur_c,
                              Tensor2D cur_h) {
      using namespace mshadow::expr;
      Tensor2D w_data = this->params[0].data[0][0];
      Tensor2D u_data = this->params[1].data[0][0];

      Tensor2D i, f, o, cc;
      cur_g = dot(w_data, x); // x: dinput * nbatch,   w_data: 4dmem * dinput
      
      cur_g += dot(u_data, pre_h); // u_data: 4dmem * 4dmem,  pre_h: 4dmem * nbatch
      //PrintTensor("cur_g",cur_g);
      if (!no_bias) {
        cur_g += b_expand.T(); // cur_g: 4dmem * nbatch,  b_expand: 4dmem * nbatch
      }
      SplitGate(cur_g, i, f, o, cc); // cur_g: 4dmem * nbatch
      i = mshadow::expr::F<op::sigmoid>(i); // logi   dmem * nbatch
      f = mshadow::expr::F<op::sigmoid>(f); // logi   dmem * nbatch
      o = mshadow::expr::F<op::sigmoid>(o); // logi   dmem * nbatch
      cc= mshadow::expr::F<op::tanh>(cc);   // tanh   dmem * nbatch

      cur_c = f * pre_c + i * cc;  //dmem * nbatch
      if (!no_out_tanh) {
        cur_h = o * mshadow::expr::F<op::tanh>(cur_c); // tanh
      } else {
        cur_h = o * cur_c;  // dmem * nbatch
      }
  }
  void PrintTensor(const char * name, mshadow::Tensor<cpu, 4> x) {
    mshadow::Shape<4> s = x.shape_;
      cout << name << " shape " << s[0] << "x" << s[1] << "x" << s[2] << "x" << s[3] << endl;
      for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
          for (unsigned int d2 = 0; d2 < s[1]; ++d2) {
              for (unsigned int d3 = 0; d3 < s[2]; ++d3) {
                  for (unsigned int d4 = 0; d4 < s[3]; ++d4) {
                      cout << x[d1][d2][d3][d4] << " ";
                  }
                  cout << "|";
              }
              cout << ";";
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

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
#if DEBUG
    checkNanParams();
#endif
    using namespace mshadow::expr;
    Tensor4D bottom_data = bottom[0]->data;
    Tensor4D top_data = top[0]->data;
    int n_steps = rd_bottom_data.size(0);
    mask_data = 0.f; rd_top_data = 0.f; rd_top_data_tmp=0.f; rd_bottom_data_tmp = 0.f; rd_bottom_data = 0.f; b_expand = 0.f;

    c = 0.f, g = 0.f; c_er = 0.f; g_er = 0.f;
    top[0]->length = F<op::identity>(bottom[0]->length);
    rd_bottom_data_tmp = swapaxis<2,0>(bottom_data);
    rd_bottom_data = swapaxis<3,2>(rd_bottom_data_tmp);
    b_expand = repmat(this->params[2].data_d1(),nbatch); // expand bias to (nbatch * 4dmem)

    for ( index_t batch_idx = 0 ; batch_idx < bottom_data.size(0); ++ batch_idx){
      for(index_t steps = 0 ; steps < bottom[0]->length[batch_idx][0]; ++ steps){
        for(index_t seq_idx = 0 ; seq_idx < bottom_data.size(1); ++ seq_idx){
          for(index_t dim_idx = 0 ; dim_idx < d_mem; ++ dim_idx){
            mask_data[steps][seq_idx][dim_idx][batch_idx] = 1.f;
          }
        }
      }
    }
    if(!reverse) {
      for(index_t step = 0 ;  step < n_steps; ++ step){
        for(index_t seq_idx = 0 ; seq_idx < rd_bottom_data.size(1); ++ seq_idx){
          Tensor2D pre_c, pre_h;
          if(step == 0){
            pre_c = begin_c;
            pre_h = begin_h;
          } else {
            pre_c = c[step-1][seq_idx];
            pre_h = rd_top_data[step-1][seq_idx];
          }
          ForwardOneStep(pre_c, pre_h, rd_bottom_data[step][seq_idx], g[step][seq_idx], c[step][seq_idx], rd_top_data[step][seq_idx]);
        }
      }
    } else {
      for(int step = n_steps - 1 ;  step >= 0; -- step){
        for(index_t seq_idx = 0 ; seq_idx < rd_bottom_data.size(1); ++ seq_idx){
          Tensor2D pre_c, pre_h;
          if(step == n_steps - 1){
            pre_c = begin_c;
            pre_h = begin_h;
          } else {
            pre_c = c[step+1][seq_idx];
            pre_h = rd_top_data[step+1][seq_idx];
          }
          ForwardOneStep(pre_c, pre_h, rd_bottom_data[step][seq_idx], g[step][seq_idx], c[step][seq_idx], rd_top_data[step][seq_idx]);
        }
      }
    }
    rd_top_data = mask_data * rd_top_data;
    //g = mask * g;
    c = mask_data * c;
    rd_top_data_tmp = swapaxis<3,2>(rd_top_data);
    top_data = swapaxis<2,0>(rd_top_data_tmp);
#if DEBUG
    checkNanParams();
#endif
  }

  // too tricky, may bring errors
  void SplitGate(Tensor2D g, Tensor2D &i, Tensor2D &f, Tensor2D &o, Tensor2D &cc) {
    //utils::Check(g.size(0) == nbatch, "BLstmLayer: gate problem."); 
    i = Tensor2D(g.dptr_, mshadow::Shape2(d_mem, nbatch));
    f = Tensor2D(g.dptr_ + nbatch * d_mem, mshadow::Shape2(d_mem, nbatch));
    o = Tensor2D(g.dptr_ + 2 * nbatch * d_mem, mshadow::Shape2(d_mem, nbatch));
    cc= Tensor2D(g.dptr_ + 3 * nbatch * d_mem, mshadow::Shape2(d_mem, nbatch));
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
                 Tensor2D pre_c,
                 Tensor2D pre_h,
                 Tensor2D x,
                 Tensor2D cur_g,
                 Tensor2D cur_c,
                 Tensor2D cur_h,
                 Tensor2D cur_c_er,
                 Tensor2D cur_g_er,
                 Tensor2D pre_c_er,
                 Tensor2D pre_h_er,
                 Tensor2D x_er) {

    using namespace mshadow::expr;
    Tensor2D w_data = this->params[0].data[0][0];
    Tensor2D u_data = this->params[1].data[0][0];
    Tensor2D w_er = this->params[0].diff[0][0];
    Tensor2D u_er = this->params[1].diff[0][0];
    Tensor2D b_er = this->params[2].diff[0][0];

    // gradient normalization by norm 2
    float n2 = norm2(cur_h_er);
    if (n2 > grad_norm2) {
      // utils::Printf("LSTM: grad norm, %f,%f\n", n2, grad_norm2);
      cur_h_er *= (grad_norm2/n2);
    }
    
    Tensor2D i, f, o, cc, i_er, f_er, o_er, cc_er;
    SplitGate(cur_g, i, f, o, cc);
    SplitGate(cur_g_er, i_er, f_er, o_er, cc_er);

    if (!no_out_tanh) {
      mshadow::TensorContainer<xpu, 2> tanhc(cur_c.shape_);
      tanhc = mshadow::expr::F<op::tanh>(cur_c);
      o_er = mshadow::expr::F<op::sigmoid_grad>(o) * (cur_h_er * tanhc); // logi
      cur_c_er += mshadow::expr::F<op::tanh_grad>(tanhc) * (cur_h_er * o);
    } else {
      o_er = mshadow::expr::F<op::sigmoid_grad>(o) * (cur_h_er * cur_c); // logi
      cur_c_er += cur_h_er * o;
    }

    i_er = mshadow::expr::F<op::sigmoid_grad>(i) * (cur_c_er * cc);    // logi
    cc_er = mshadow::expr::F<op::tanh_grad>(cc) * (cur_c_er * i);      // tanh
    pre_c_er = cur_c_er * f;
    f_er = mshadow::expr::F<op::sigmoid_grad>(f) * (cur_c_er * pre_c); // logi

    pre_h_er += dot(u_data.T(), cur_g_er); // cur_g_er: 4dmem * nbatch , u_data: 4dmem * dmem
    x_er += dot(w_data.T(), cur_g_er); // cur_g_er: 4dmem * nbatch, w_data: dinput * 4dmem

    // grad
    if (!no_bias) {  // need to speed
      this->params[2].diff_d1() += sum_rows(cur_g_er.T());
    }
    w_er += dot(cur_g_er, x.T());  // w_er: 4dmem * dinput, x.T(): d_input * nbatch , cur_g_er: 4dmem * nbatch
    u_er += dot(cur_g_er, pre_h.T());
  }

  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
#if DEBUG
    checkNanParams();
#endif
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;

    mask_diff = 0.f; rd_top_diff_tmp = 0.f; rd_top_diff = 0.f;
    rd_bottom_diff = 0.f; rd_bottom_diff_tmp = 0.f;
    begin_c_er = 0.; begin_h_er = 0.; g_er = 0.; c_er = 0.;
    int n_steps = rd_bottom_diff.size(0);
    Tensor2D pre_c, pre_h, pre_c_er, pre_h_er;

    for ( index_t batch_idx = 0 ; batch_idx < bottom_data.size(0); ++ batch_idx){
      for(index_t seq_idx = 0 ; seq_idx < bottom_data.size(1); ++ seq_idx){
        for(index_t dim_idx = 0 ; dim_idx < bottom_data.size(3); ++ dim_idx){
          for(index_t steps = 0 ; steps < bottom[0]->length[batch_idx][0]; ++ steps){
            mask_diff[steps][seq_idx][dim_idx][batch_idx] = 1.f;
          }
        }
      }
    }
    rd_top_diff_tmp = swapaxis<2,0>(top_diff);
    rd_top_diff = swapaxis<3,2>(rd_top_diff_tmp);
    if(!reverse) {
      for(int step = n_steps - 1; step >= 0; -- step){ //attention here step should be decleared as int
        for(index_t seq_idx = 0 ; seq_idx < rd_bottom_diff.size(1); ++ seq_idx){
          if(0 == step ){
            pre_c = begin_c;
            pre_h = begin_h;
            pre_c_er = begin_c_er;
            pre_h_er = begin_h_er;
          } else {
            pre_c = c[step - 1][seq_idx];
            pre_h = rd_top_data[step - 1][seq_idx];
            pre_c_er = c_er[step - 1][seq_idx];
            pre_h_er = rd_top_diff[step - 1][seq_idx];
          }
          BpOneStep(rd_top_diff[step][seq_idx],
                    pre_c,
                    pre_h,
                    rd_bottom_data[step][seq_idx], // d_input * nbatch
                    g[step][seq_idx],
                    c[step][seq_idx],
                    rd_top_data[step][seq_idx],
                    c_er[step][seq_idx],
                    g_er[step][seq_idx],
                    pre_c_er,
                    pre_h_er,
                    rd_bottom_diff[step][seq_idx]);
        }
      }
    } else {
      for(int step = 0 ; step < n_steps; ++ step){
        for(index_t seq_idx = 0 ; seq_idx < rd_bottom_diff.size(1); ++ seq_idx){
          if( n_steps-1 == step){
            pre_c = begin_c;
            pre_h = begin_h;
            pre_c_er = begin_c_er;
            pre_h_er = begin_h_er;
          } else {
            pre_c = c[step + 1][seq_idx];
            pre_h = rd_top_data[step + 1][seq_idx];
            pre_c_er = c_er[step + 1][seq_idx];
            pre_h_er = rd_top_diff[step + 1][seq_idx];
          }
          BpOneStep(rd_top_diff[step][seq_idx],
                    pre_c,
                    pre_h,
                    rd_bottom_data[step][seq_idx],
                    g[step][seq_idx],
                    c[step][seq_idx],
                    rd_top_data[step][seq_idx],
                    c_er[step][seq_idx],
                    g_er[step][seq_idx],
                    pre_c_er,
                    pre_h_er,
                    rd_bottom_diff[step][seq_idx]);
        }
      }
    }
    rd_bottom_diff = mask_diff * rd_bottom_diff;
    rd_bottom_diff_tmp = swapaxis<3,2>(rd_bottom_diff);
    bottom_diff = swapaxis<2,0>(rd_bottom_diff_tmp);

    this->params[0].CutOffGradient(grad_cut_off);
    this->params[1].CutOffGradient(grad_cut_off);
    this->params[2].CutOffGradient(grad_cut_off);

#if DEBUG
    this->params[0].PrintStatistic("BLSTM W");
    this->params[1].PrintStatistic("BLSTM U");
    this->params[2].PrintStatistic("BLSTM b");
    checkNanParams();
#endif
  }
  void LoadTensor(Json::Value &tensor_root, mshadow::TensorContainer<xpu, 4> &t) {
    Json::Value data_root = tensor_root["data"];
    int s0 = data_root["shape"][0].asInt();
    int s1 = data_root["shape"][1].asInt();
    int s2 = data_root["shape"][2].asInt();
    int s3 = data_root["shape"][3].asInt();
    utils::Check(t.size(0) == s0 && t.size(1) == s1 && t.size(2) == s2 && t.size(3) == s3, 
                 "BLstmLayer: load tensor error.");
    int size = s0*s1*s2*s3;
    for (int i = 0; i < size; ++i) {
      t.dptr_[i] = data_root["value"][i].asFloat();
    }
  }
  void LoadParam() {
    utils::Printf("BLstmLayer: load params...\n");
    Json::Value param_root;
    ifstream ifs(param_file.c_str());
    ifs >> param_root;
    ifs.close();
    LoadTensor(param_root[0], this->params[0].data);
    LoadTensor(param_root[1], this->params[1].data);
    LoadTensor(param_root[2], this->params[2].data);
  }

 public:
// protected:
  float max_norm2;
  int d_mem, d_input, nbatch;
  bool no_bias, reverse, no_out_tanh; 
  float grad_norm2;
  float o_gate_bias_init;
  float f_gate_bias_init;
  float grad_cut_off;
  string param_file;
  mshadow::TensorContainer<xpu, 4> mask_data, mask_diff, rd_bottom_data, rd_top_data, rd_bottom_diff, rd_top_diff; 
  mshadow::TensorContainer<xpu, 4> rd_bottom_data_tmp, rd_top_data_tmp, rd_bottom_diff_tmp, rd_top_diff_tmp; 
  mshadow::TensorContainer<xpu, 4> c, g, c_er, g_er; 
  mshadow::TensorContainer<xpu, 2> b_expand, begin_h, begin_c, begin_c_er, begin_h_er;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_BLSTM_LAYER_INL_HPP_
