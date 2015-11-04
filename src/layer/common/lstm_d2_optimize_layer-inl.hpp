#ifndef TEXTNET_LAYER_LSTM_D2_OPTIMIZE_LAYER_INL_HPP_
#define TEXTNET_LAYER_LSTM_D2_OPTIMIZE_LAYER_INL_HPP_

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
class LstmD2OptimizeLayer : public Layer<xpu> {
 public:
  LstmD2OptimizeLayer(LayerType type) { this->layer_type = type; }
  virtual ~LstmD2OptimizeLayer(void) { }
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  typedef mshadow::TensorContainer<xpu, 1> TensorC1D;
  typedef mshadow::TensorContainer<xpu, 2> TensorC2D;
  typedef mshadow::TensorContainer<xpu, 3> TensorC3D;
  typedef mshadow::TensorContainer<xpu, 4> TensorC4D;
  
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "LstmD2OptimizeLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LstmD2OptimizeLayer:top size problem.");

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

    // begin_h.Resize(mshadow::Shape2(1, d_mem), 0.f);
    // begin_c.Resize(mshadow::Shape2(1, d_mem), 0.f);
    // begin_h_er.Resize(mshadow::Shape2(1, d_mem), 0.f);
    // begin_c_er.Resize(mshadow::Shape2(1, d_mem), 0.f);

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
    utils::Check(bottom.size() == BottomNodeNum(), "LstmD2OptimizeLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LstmD2OptimizeLayer:top size problem.");
    
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    // (batch size, x_len, y_len, d_mem)
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);
    // input, forget * 3, output, candidate
    // mshadow::Shape<4> shape_gate= mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem*6); 

    top[0]->Resize(shape_out, mshadow::Shape2(shape_out[0],2), true);
    // c.Resize(shape_out, 0.f);
    // g.Resize(shape_gate, 0.f);
    // c_er.Resize(shape_out, 0.f);
    // g_er.Resize(shape_gate, 0.f);

    // 这个地方对中间变量进行初始化
    run_pos.Resize(mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], 1), true); 
    int batch_size = shape_out[0];
    int max_x = shape_out[1];
    int max_y = shape_out[2];
    int max_run = max_x + max_y - 1;
    run_max_len.clear();
    run_begin_idx.clear();
    int total_cnt = 0;
    for (int i = 0; i < max_run; ++i) {
      int min_len = max_x < max_y ? max_x : max_y;
      int cnt = i+1;
      if (cnt > (max_x+max_y)/2) {
        cnt = (max_x+max_y) - cnt;
      }
      if (cnt > min_len) {
        cnt = min_len;
      }
      cnt *= batch_size;
      run_max_len.push_back(cnt);
      run_begin_idx.push_back(total_cnt);
      total_cnt += cnt;
    }
    ReshapeRunTensors(max_run, total_cnt, d_input, d_mem);

	if (show_info) {
      bottom[0]->PrintShape("bottom0");
	  top[0]->PrintShape("top0");
	}
  }

  // 这个函数为中间结果重新分配内存
  void ReshapeRunTensors(int max_run, int total_cnt, int d_input, int d_mem) {
    utils::Assert(run_max_len.size() == max_run, "LstmD2OptimizeLayer: size error.");

    run_x.Resize(mshadow::Shape2(total_cnt, d_input), true);
    run_c.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_h.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_input.Resize(mshadow::Shape2(total_cnt, d_input+d_mem*3), true);
    run_g.Resize(mshadow::Shape2(total_cnt, d_mem*6), true);
    run_i.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_f_l.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_f_m.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_f_t.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_o.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_cc.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_pre_c_l.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_pre_c_m.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_pre_c_t.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_pre_h_l.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_pre_h_m.Resize(mshadow::Shape2(total_cnt, d_mem), true);
    run_pre_h_t.Resize(mshadow::Shape2(total_cnt, d_mem), true);

    /*
    for (int i = 0; i < run_x.size(); ++i) {
      delete run_x[i];
      delete run_c[i];
      delete run_h[i];
      delete run_input[i];
      delete run_g[i];
      delete run_i[i];
      delete run_f_l[i];
      delete run_f_m[i];
      delete run_f_t[i];
      delete run_o[i];
      delete run_cc[i];
      delete run_pre_c_l[i];
      delete run_pre_c_m[i];
      delete run_pre_c_t[i];
      delete run_pre_h_l[i];
      delete run_pre_h_m[i];
      delete run_pre_h_t[i];
    }
    run_x.clear();
    run_c.clear();
    run_h.clear();
    run_input.clear();
    run_g.clear();
    run_i.clear();
    run_f_l.clear();
    run_f_m.clear();
    run_f_t.clear();
    run_o.clear();
    run_cc.clear();
    run_pre_c_l.clear();
    run_pre_c_m.clear();
    run_pre_c_t.clear();
    run_pre_h_l.clear();
    run_pre_h_m.clear();
    run_pre_h_t.clear();

    for (int i = 0; i < max_run; ++i) {
      run_x.push_back(new TensorC4D);
      run_c.push_back(new TensorC4D);
      run_h.push_back(new TensorC4D);
      run_input.push_back(new TensorC4D);
      run_g.push_back(new TensorC4D);
      run_i.push_back(new TensorC4D);
      run_f_l.push_back(new TensorC4D);
      run_f_m.push_back(new TensorC4D);
      run_f_t.push_back(new TensorC4D);
      run_o.push_back(new TensorC4D);
      run_cc.push_back(new TensorC4D);
      run_pre_c_l.push_back(new TensorC4D);
      run_pre_c_m.push_back(new TensorC4D);
      run_pre_c_t.push_back(new TensorC4D);
      run_pre_h_l.push_back(new TensorC4D);
      run_pre_h_m.push_back(new TensorC4D);
      run_pre_h_t.push_back(new TensorC4D);

      run_x[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_input));
      run_c[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_h[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_input[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_input+d_mem*3));
      run_g[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem*6));
      run_i[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_f_l[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_f_m[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_f_t[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_o[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_cc[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_pre_c_l[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_pre_c_m[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_pre_c_t[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_pre_h_l[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_pre_h_m[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
      run_pre_h_t[i]->Resize(mshadow::Shape4(1,1,batch_size*run_max_len[i],d_mem));
    }
    */
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

  // 以前的这个函数不支持batch，现在改成batch的版本
  void concat_input_batch(Tensor2D &x, 
                          Tensor2D &h_l, 
                          Tensor2D &h_m, 
                          Tensor2D &h_t, 
                          Tensor2D &input) {
    utils::Check(x.size(1)+h_l.size(1)+h_m.size(1)+h_t.size(1) == input.size(1), "LstmD2OptimizeLayer: size error.");

    // 这个地方最好用memcpy直接优化，当然，这就要求内存必须连续存放了，一般使用没什么问题
    int batch_size = x.size(0);
    for (index_t row_idx = 0; row_idx < batch_size; ++row_idx) {
      float *p_dst = input.dptr_ + row_idx*input.size(1);
      float *p_src = x.dptr_ + row_idx*x.size(1);
      memcpy(p_dst, p_src, x.size(1)*sizeof(float));

      p_dst += x.size(1);
      p_src = h_l.dptr_ + row_idx*h_l.size(1);
      memcpy(p_dst, p_src, h_l.size(1)*sizeof(float));

      p_dst += h_l.size(1);
      p_src = h_m.dptr_ + row_idx*h_m.size(1);
      memcpy(p_dst, p_src, h_m.size(1)*sizeof(float));

      p_dst += h_m.size(1);
      p_src = h_t.dptr_ + row_idx*h_t.size(1);
      memcpy(p_dst, p_src, h_t.size(1)*sizeof(float));
    }
  }

  // 以前的这个函数不支持batch，现在改成batch的版本
  void split_gate_batch(Tensor2D &g, 
                        Tensor2D &i, 
                        Tensor2D &f_l, 
                        Tensor2D &f_m, 
                        Tensor2D &f_t, 
                        Tensor2D &o, 
                        Tensor2D &cc) {
    int batch_size = g.size(0);
    for (index_t row_idx = 0; row_idx < batch_size; ++row_idx) {
      float *p_src = g.dptr_ + row_idx*g.size(1);
      float *p_dst = i.dptr_ + row_idx*i.size(1);
      memcpy(p_dst, p_src, i.size(1)*sizeof(float));

      p_src += i.size(1);
      p_dst = f_l.dptr_ + row_idx*f_l.size(1);
      memcpy(p_dst, p_src, f_l.size(1)*sizeof(float));

      p_src += f_l.size(1);
      p_dst = f_m.dptr_ + row_idx*f_m.size(1);
      memcpy(p_dst, p_src, f_m.size(1)*sizeof(float));

      p_src += f_m.size(1);
      p_dst = f_t.dptr_ + row_idx*f_t.size(1);
      memcpy(p_dst, p_src, f_t.size(1)*sizeof(float));

      p_src += f_t.size(1);
      p_dst = o.dptr_ + row_idx*o.size(1);
      memcpy(p_dst, p_src, o.size(1)*sizeof(float));

      p_src += o.size(1);
      p_dst = cc.dptr_ + row_idx*cc.size(1);
      memcpy(p_dst, p_src, cc.size(1)*sizeof(float));
    }
  }

  // this is different with concat_input(), this re-use the same memory
  // void split_input(Tensor2D t, Tensor2D &x, Tensor2D &h_l, Tensor2D &h_m, Tensor2D &h_t) {
  //   utils::Check(t.size(0)==1 && t.size(1)==(d_input+3*d_mem), "LstmD2OptimizeLayer: size error.");

  //   x   = Tensor2D(t.dptr_,                   mshadow::Shape2(1, d_input));
  //   h_l = Tensor2D(t.dptr_ + d_input,         mshadow::Shape2(1, d_mem));
  //   h_m = Tensor2D(t.dptr_ + d_input+1*d_mem, mshadow::Shape2(1, d_mem));
  //   h_t = Tensor2D(t.dptr_ + d_input+2*d_mem, mshadow::Shape2(1, d_mem));
  // }

  // 第一步优化是把这个地方改成batch的，不要一个一个算
  // 这个地方用空间换时间，内存拷贝的中间结果一概保存
  // input里面的是在函数内部拼接
  // 这个函数目前不进行任何内存分配，所有在外部分配好，外部决定什么东西需要保存以加速计算，什么不需要
  virtual void ForwardOneStep(Tensor2D pre_c_l, // left
                              Tensor2D pre_c_m, // left top
                              Tensor2D pre_c_t, // top
                              Tensor2D pre_h_l,
                              Tensor2D pre_h_m,
                              Tensor2D pre_h_t,
                              Tensor2D cur_x, // 到此是输入，其余都是本函数填充的
                              Tensor2D input,
                              Tensor2D cur_g, // 这个是保存做非线性之前的值
                              Tensor2D cur_i,
                              Tensor2D cur_f_l,
                              Tensor2D cur_f_m,
                              Tensor2D cur_f_t,
                              Tensor2D cur_o,
                              Tensor2D cur_cc,
                              Tensor2D cur_c,
                              Tensor2D cur_h) {
      Tensor2D w_data = this->params[0].data_d2_reverse();
      Tensor1D b_data = this->params[1].data_d1_reverse();

      int batch_size = cur_x.size(0);

      high_resolution_clock::time_point b_time_3 = high_resolution_clock::now();
      concat_input_batch(cur_x, pre_h_l, pre_h_m, pre_h_t, input);
      cur_g = dot(input, w_data);
      if (!no_bias) {
        cur_g += repmat(b_data,batch_size);
      }
      split_gate_batch(cur_g, cur_i, cur_f_l, cur_f_m, cur_f_t, cur_o, cur_cc);


      cur_i   = mshadow::expr::F<op::sigmoid>(cur_i);   // logi
      cur_f_l = mshadow::expr::F<op::sigmoid>(cur_f_l); // logi
      cur_f_m = mshadow::expr::F<op::sigmoid>(cur_f_m); // logi
      cur_f_t = mshadow::expr::F<op::sigmoid>(cur_f_t); // logi
      cur_o   = mshadow::expr::F<op::sigmoid>(cur_o);   // logi
      cur_cc  = mshadow::expr::F<op::tanh>(cur_cc);     // tanh 

      cur_c = cur_f_l * pre_c_l + cur_f_m * pre_c_m + cur_f_t * pre_c_t + cur_i * cur_cc;
      // if (!no_out_tanh) {
        cur_h = cur_o * mshadow::expr::F<op::tanh>(cur_c); // tanh
      // } else {
      //   cur_h = o * cur_c; 
      // }
      high_resolution_clock::time_point e_time_3 = high_resolution_clock::now();
      time_3 += duration_cast<duration<double>>(e_time_3 - b_time_3);
  }

  // x: (x_max_len, y_max_len, d_input)
  void ForwardLeftTop2RightBottom(Tensor4D &x, vector<int> &x_lens, vector<int> &y_lens, Tensor4D &top) {
    // 这个外围函数也要好好设计一下，目前的思路是这样的，每一次不仅是不同的example之间的并行，
    // 也把同一个example里面没有数据依赖性的地方一起算了
    // utils::Check(x_len > 0 && y_len > 0 && x_len <= x.size(0) && y_len <= x.size(1), "LstmD2OptimizeLayer: input size error.");
    //
    int batch_size= x.size(0);
    int max_x_len = x.size(1);
    int max_y_len = x.size(2);
    int max_run   = max_x_len + max_y_len - 1;

    // forward前先进行清零
    // for (int run_idx = 0; run_idx < max_run; ++run_idx) {
    // 这个地方大概占据了0.05s
    // run_x = 0.f;
    // run_c = 0.f;
    // run_h = 0.f;
    // run_g = 0.f;
    // run_input = 0.f;
    // run_i   = 0.f;
    // run_f_l = 0.f;
    // run_f_m = 0.f;
    // run_f_t = 0.f;
    // run_o   = 0.f;
    // run_cc  = 0.f;
    // run_pre_c_l = 0.f;
    // run_pre_c_m = 0.f;
    // run_pre_c_t = 0.f;
    // run_pre_h_l = 0.f;
    // run_pre_h_m = 0.f;
    // run_pre_h_t = 0.f;
    // }

    for (int run_idx = 0; run_idx < max_run; ++run_idx) { // 这是一次forward的执行

      high_resolution_clock::time_point b_time_2 = high_resolution_clock::now();
      int begin_idx = run_begin_idx[run_idx];
      int end_idx   = begin_idx + run_max_len[run_idx];
      Tensor2D cur_x     = run_x.Slice(begin_idx, end_idx);
      Tensor2D cur_c     = run_c.Slice(begin_idx, end_idx);
      Tensor2D cur_h     = run_h.Slice(begin_idx, end_idx);
      Tensor2D cur_g     = run_g.Slice(begin_idx, end_idx);
      Tensor2D cur_input = run_input.Slice(begin_idx, end_idx);
      Tensor2D cur_i     = run_i.Slice(begin_idx, end_idx);
      Tensor2D cur_f_l   = run_f_l.Slice(begin_idx, end_idx);
      Tensor2D cur_f_m   = run_f_m.Slice(begin_idx, end_idx);
      Tensor2D cur_f_t   = run_f_t.Slice(begin_idx, end_idx);
      Tensor2D cur_o     = run_o.Slice(begin_idx, end_idx);
      Tensor2D cur_cc    = run_cc.Slice(begin_idx, end_idx);
      Tensor2D pre_c_l   = run_pre_c_l.Slice(begin_idx, end_idx);
      Tensor2D pre_c_m   = run_pre_c_m.Slice(begin_idx, end_idx);
      Tensor2D pre_c_t   = run_pre_c_t.Slice(begin_idx, end_idx);
      Tensor2D pre_h_l   = run_pre_h_l.Slice(begin_idx, end_idx);
      Tensor2D pre_h_m   = run_pre_h_m.Slice(begin_idx, end_idx);
      Tensor2D pre_h_t   = run_pre_h_t.Slice(begin_idx, end_idx);

      int cur_cnt = 0; // 这个是记录run内，每个batch中不同example中的不同(x,y)位置上的表达在这个run内所处的位置
      for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int x_len = x_lens[batch_idx];
        int y_len = y_lens[batch_idx];
        if (run_idx >= x_len+y_len-1)
            continue;
        // 这个是我推理得到的长度，因为程序比较复杂，比较难debug，所以相互印证一下
        int min_len = x_len < y_len ? x_len : y_len;
        int cnt = run_idx+1;
        if (cnt > (x_len+y_len)/2) {
          cnt = (x_len+y_len) - cnt;
        }
        if (cnt > min_len) {
          cnt = min_len;
        }

        utils::Assert(cnt > 0, "LstmD2OptimizeLayer: run position error.");
        // 寻找到每个run在当前这个矩阵中要处理的一个斜长条的开始和结束的位置
        // 注意，开始和结束的位置都是要处理的
        int begin_x = run_idx < x_len-1 ? run_idx : x_len-1;
        int end_y   = run_idx < y_len-1 ? run_idx : y_len-1;
        int begin_y = run_idx < x_len-1 ? 0 : (run_idx-(x_len-1));
        int end_x   = run_idx < y_len-1 ? 0 : (run_idx-(y_len-1));
        utils::Assert(begin_x >= end_x && begin_y <= end_y, "LstmD2OptimizeLayer: run position error.");
        utils::Assert((begin_x-end_x+1)==cnt && (end_y-end_y+1)==cnt, "LstmD2OptimizeLayer: run position error.");

        // 先搞定left
        if (begin_y == 0) { // 第一个没有left，设置为0
          pre_c_l[cur_cnt] = 0.f;
          pre_h_l[cur_cnt] = 0.f;
          if (cnt > 1) { // 这个设置剩余的left
            int pre_run_begin_idx = run_begin_idx[run_idx-1]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-1][begin_y][0]; // 这个是run内某个batch的开始位置索引
            begin_pos += pre_run_begin_idx; // 绝对位置
            pre_c_l.Slice(cur_cnt+1, cur_cnt+cnt) = 
              mshadow::expr::F<op::identity>(run_c.Slice(begin_pos, begin_pos+cnt-1));
            pre_h_l.Slice(cur_cnt+1, cur_cnt+cnt) = 
              mshadow::expr::F<op::identity>(run_h.Slice(begin_pos, begin_pos+cnt-1));
          }
        } else { // 所有的都有left
          int pre_run_begin_idx = run_begin_idx[run_idx-1];
          int begin_pos = run_pos[batch_idx][begin_x][begin_y-1][0]; // 这个是run内某个batch的开始位置索引
          begin_pos += pre_run_begin_idx;
          pre_c_l.Slice(cur_cnt, cur_cnt+cnt) = 
            mshadow::expr::F<op::identity>(run_c.Slice(begin_pos, begin_pos+cnt));
          pre_h_l.Slice(cur_cnt, cur_cnt+cnt) = 
            mshadow::expr::F<op::identity>(run_h.Slice(begin_pos, begin_pos+cnt));
        }
        // 再搞定top
        if (end_x == 0) { // 最后一个没有top，设置为0
          pre_c_t[cur_cnt+cnt-1] = 0.f;
          pre_h_t[cur_cnt+cnt-1] = 0.f;
          if (cnt > 1) { // 这个设置剩余的left
            int pre_run_begin_idx = run_begin_idx[run_idx-1]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-1][begin_y][0]; // 这个是run内某个batch的开始位置索引
            begin_pos = pre_run_begin_idx + begin_pos;
            pre_c_t.Slice(cur_cnt, cur_cnt+cnt-1) = 
              mshadow::expr::F<op::identity>(run_c.Slice(begin_pos, begin_pos+cnt-1));
            pre_h_t.Slice(cur_cnt, cur_cnt+cnt-1) = 
              mshadow::expr::F<op::identity>(run_h.Slice(begin_pos, begin_pos+cnt-1));
          }
        } else { // 所有的都有top
          int pre_run_begin_idx = run_begin_idx[run_idx-1];
          int begin_pos = run_pos[batch_idx][begin_x-1][begin_y][0]; // 这个是run内某个batch的开始位置索引
          begin_pos = pre_run_begin_idx + begin_pos;
          pre_c_t.Slice(cur_cnt, cur_cnt+cnt) = 
            mshadow::expr::F<op::identity>(run_c.Slice(begin_pos, begin_pos+cnt));
          pre_h_t.Slice(cur_cnt, cur_cnt+cnt) = 
            mshadow::expr::F<op::identity>(run_h.Slice(begin_pos, begin_pos+cnt));
        }
        // 最后搞定left top
        // 先处理好第一个与最后一个
        // 处理中间的
        if (begin_y == 0 && end_x == 0) { // 一头一尾都没有
          pre_c_m[cur_cnt]       = 0.f;
          pre_h_m[cur_cnt]       = 0.f;
          pre_c_m[cur_cnt+cnt-1] = 0.f;
          pre_h_m[cur_cnt+cnt-1] = 0.f;
          if (cnt > 2) {
            int pre_run_begin_idx = run_begin_idx[run_idx-2]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-2][begin_y][0]; // 这个是run内某个batch的开始位置索引
            begin_pos = pre_run_begin_idx + begin_pos;
            pre_c_m.Slice(cur_cnt+1, cur_cnt+cnt-1) = 
              mshadow::expr::F<op::identity>(run_c.Slice(begin_pos, begin_pos+cnt-2));
            pre_h_m.Slice(cur_cnt+1, cur_cnt+cnt-1) = 
              mshadow::expr::F<op::identity>(run_h.Slice(begin_pos, begin_pos+cnt-2));
          }
        } else if (begin_y == 0) { // 头没有
          pre_c_m[cur_cnt]       = 0.f;
          pre_h_m[cur_cnt]       = 0.f;
          if (cnt > 1) {
            int pre_run_begin_idx = run_begin_idx[run_idx-2]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-2][begin_y][0]; // 这个是run内某个batch的开始位置索引
            begin_pos = pre_run_begin_idx + begin_pos;
            pre_c_m.Slice(cur_cnt+1, cur_cnt+cnt) = 
              mshadow::expr::F<op::identity>(run_c.Slice(begin_pos, begin_pos+cnt-1));
            pre_h_m.Slice(cur_cnt+1, cur_cnt+cnt) = 
              mshadow::expr::F<op::identity>(run_h.Slice(begin_pos, begin_pos+cnt-1));
          }
        } else if (end_x == 0) { // 尾没有
          pre_c_m[cur_cnt+cnt-1] = 0.f;
          pre_h_m[cur_cnt+cnt-1] = 0.f;
          if (cnt > 1) {
            int pre_run_begin_idx = run_begin_idx[run_idx-2]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-1][begin_y-1][0]; // 这个是run内某个batch的开始位置索引
            begin_pos = pre_run_begin_idx + begin_pos;
            pre_c_m.Slice(cur_cnt, cur_cnt+cnt-1) = 
              mshadow::expr::F<op::identity>(run_c.Slice(begin_pos, begin_pos+cnt-1));
            pre_h_m.Slice(cur_cnt, cur_cnt+cnt-1) = 
              mshadow::expr::F<op::identity>(run_h.Slice(begin_pos, begin_pos+cnt-1));
          }
        } else {
          int pre_run_begin_idx = run_begin_idx[run_idx-2]; // 这个是整个run的开始的位置索引
          int begin_pos = run_pos[batch_idx][begin_x-1][begin_y-1][0]; // 这个是run内某个batch的开始位置索引
          begin_pos = pre_run_begin_idx + begin_pos;
          pre_c_m.Slice(cur_cnt, cur_cnt+cnt) = 
            mshadow::expr::F<op::identity>(run_c.Slice(begin_pos, begin_pos+cnt));
          pre_h_m.Slice(cur_cnt, cur_cnt+cnt) = 
            mshadow::expr::F<op::identity>(run_h.Slice(begin_pos, begin_pos+cnt));
        }

        for (int i = 0; i < cnt; ++i) {
          // 定位到当前要处理的位置，然后把计算当前位置表达所需要的东西全部放好
          int pos_x = begin_x - i;
          int pos_y = begin_y + i;

          // 注意，这个地方的内存拷贝是可以优化的，每一个run的前驱状态是连续存储的，
          // 因此其实是可以整块整块的拷贝的，而不是现在一个向量一个向量的拷贝
          run_pos[batch_idx][pos_x][pos_y][0] = cur_cnt; // 保存当前位置在整个run中的相对位置
          cur_x[cur_cnt] = mshadow::expr::F<op::identity>(x[batch_idx][pos_x][pos_y]);
          // if (pos_y > 0) {
          //   int pos = run_pos[batch_idx][pos_x][pos_y-1][0];
          //   int pre_run_begin_idx = run_begin_idx[run_idx-1];
          //   pre_c_l[cur_cnt] = mshadow::expr::F<op::identity>(run_c[pre_run_begin_idx+pos]);
          //   pre_h_l[cur_cnt] = mshadow::expr::F<op::identity>(run_h[pre_run_begin_idx+pos]);
          // }
          // if (pos_x > 0) {
          //   int pos = run_pos[batch_idx][pos_x-1][pos_y][0];
          //   int pre_run_begin_idx = run_begin_idx[run_idx-1];
          //   pre_c_t[cur_cnt] = mshadow::expr::F<op::identity>(run_c[pre_run_begin_idx+pos]);
          //   pre_h_t[cur_cnt] = mshadow::expr::F<op::identity>(run_h[pre_run_begin_idx+pos]);
          // }
          // if (pos_x > 0 && pos_y > 0) {
          //   int pos = run_pos[batch_idx][pos_x-1][pos_y-1][0];
          //   int pre_run_begin_idx = run_begin_idx[run_idx-2]; // Notice: this is pre pre run
          //   pre_c_m[cur_cnt] = mshadow::expr::F<op::identity>(run_c[pre_run_begin_idx+pos]);
          //   pre_h_m[cur_cnt] = mshadow::expr::F<op::identity>(run_h[pre_run_begin_idx+pos]);
          // }
          cur_cnt += 1;
        }
      }

      high_resolution_clock::time_point e_time_2 = high_resolution_clock::now();
      time_2 += duration_cast<duration<double>>(e_time_2 - b_time_2);
      ForwardOneStep(pre_c_l.Slice(0, cur_cnt),
                     pre_c_m.Slice(0, cur_cnt), 
                     pre_c_t.Slice(0, cur_cnt),
                     pre_h_l.Slice(0, cur_cnt),
                     pre_h_m.Slice(0, cur_cnt),
                     pre_h_t.Slice(0, cur_cnt),
                     cur_x.Slice(0, cur_cnt), // 到此是输入，其余都是本函数填充的
                     cur_input.Slice(0, cur_cnt),
                     cur_g.Slice(0, cur_cnt), // 这个是保存做非线性之前的值
                     cur_i.Slice(0, cur_cnt),
                     cur_f_l.Slice(0, cur_cnt),
                     cur_f_m.Slice(0, cur_cnt),
                     cur_f_t.Slice(0, cur_cnt),
                     cur_o.Slice(0, cur_cnt),
                     cur_cc.Slice(0, cur_cnt),
                     cur_c.Slice(0, cur_cnt),
                     cur_h.Slice(0, cur_cnt));
    }

    // 然后我们把结果拷贝到top_data中去
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int x_len = x_lens[batch_idx];
      int y_len = y_lens[batch_idx];
      for (int row_idx = 0; row_idx < x_len; ++row_idx) {
        for (int col_idx = 0; col_idx < y_len; ++col_idx) {
          int run_idx = row_idx + col_idx;
          int pos = run_pos[batch_idx][row_idx][col_idx][0];
          int cur_run_begin_idx = run_begin_idx[run_idx];
          top[batch_idx][row_idx][col_idx] = mshadow::expr::F<op::identity>(run_h[cur_run_begin_idx+pos]);
        }
      }
    }
    // high_resolution_clock::time_point e_time_3 = high_resolution_clock::now();
    // time_3 += duration_cast<duration<double>>(e_time_3 - e_time_2);
    //
    // high_resolution_clock::time_point e_time_1 = high_resolution_clock::now();
    // time_1 += duration_cast<duration<double>>(e_time_1 - b_time_1);
	// utils::Printf("\tTime:%fs,%fs,%f\n", time_1.count(), time_2.count(), time_3.count()); 
  }

  // x: (x_max_len, y_max_len, d_input)
  // void ForwardRightBottom2LeftTop(Tensor3D x, int x_len, int y_len, 
  //                                 Tensor3D g, Tensor3D c, Tensor3D h) {
  //   utils::Check(x_len > 0 && y_len > 0 && x_len <= x.size(0) && y_len <= x.size(1), "LstmD2OptimizeLayer: input size error.");
  //   Tensor2D pre_c_l, pre_c_m, pre_c_t;
  //   Tensor2D pre_h_l, pre_h_m, pre_h_t;
  //   // not need any padding, begin h and c are set to 0
  //   for (int row_idx = x_len-1; row_idx >= 0; --row_idx) {
  //     for (int col_idx = y_len-1; col_idx >= 0; --col_idx) {
  //       if (row_idx == x_len-1) {
  //         pre_c_t = begin_c;
  //         pre_h_t = begin_h;
  //       } else {
  //         pre_c_t = c[row_idx+1].Slice(col_idx, col_idx+1);
  //         pre_h_t = h[row_idx+1].Slice(col_idx, col_idx+1);
  //       }
  //       if (col_idx == y_len-1) {
  //         pre_c_l = begin_c;
  //         pre_h_l = begin_h;
  //       } else {
  //         pre_c_l = c[row_idx].Slice(col_idx+1, col_idx+2);
  //         pre_h_l = h[row_idx].Slice(col_idx+1, col_idx+2);
  //       }
  //       if (row_idx == x_len-1 || col_idx == y_len-1) {
  //         pre_c_m = begin_c;
  //         pre_h_m = begin_h;
  //       } else {
  //         pre_c_m = c[row_idx+1].Slice(col_idx+1, col_idx+2);
  //         pre_h_m = h[row_idx+1].Slice(col_idx+1, col_idx+2);
  //       }

  //       ForwardOneStep(pre_c_l,
  //                      pre_c_m,
  //                      pre_c_t,
  //                      pre_h_l,
  //                      pre_h_m,
  //                      pre_h_t,
  //                      x[row_idx].Slice(col_idx, col_idx+1),
  //                      g[row_idx].Slice(col_idx, col_idx+1), 
  //                      c[row_idx].Slice(col_idx, col_idx+1), 
  //                      h[row_idx].Slice(col_idx, col_idx+1));
  //     }
  //   }
  // }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
// #if DEBUG
//     checkNanParams();
// #endif
    Tensor4D bottom_data = bottom[0]->data;
    Tensor2D bottom_len  = bottom[0]->length;
    Tensor4D top_data    = top[0]->data;

    utils::Check(bottom_len.size(0) == bottom_data.size(0) && 
                 bottom_len.size(1) == 2, "LstmD2OptimizeLayer: input length error.");
    top[0]->length = mshadow::expr::F<op::identity>(bottom[0]->length);

    high_resolution_clock::time_point b_time_1 = high_resolution_clock::now();
    vector<int> x_len, y_len;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      x_len.push_back(bottom_len[batch_idx][0]);
      y_len.push_back(bottom_len[batch_idx][1]);
    }
    ForwardLeftTop2RightBottom(bottom_data, x_len, y_len, top_data);

    high_resolution_clock::time_point e_time_1 = high_resolution_clock::now();
    time_1 += duration_cast<duration<double>>(e_time_1 - b_time_1);
	utils::Printf("\tLSTM D2 OPTIMIZE Time:%fs,%fs,%f\n", time_1.count(), time_2.count(), time_3.count()); 

    // top_data = 0.f; c = 0.f, g = 0.f; c_er = 0.f; g_er = 0.f;
    // for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
    //   int x_len = bottom_len[batch_idx][0];
    //   int y_len = bottom_len[batch_idx][1];
    //   utils::Assert(x_len >= 0 && y_len >= 0, "LstmD2OptimizeLayer: sequence length error.");
    //   if (!reverse) {
    //     ForwardLeftTop2RightBottom(bottom_data,
    //                                x_len, y_len,
    //                                g[batch_idx], 
    //                                c[batch_idx],
    //                                top_data[batch_idx]);
    //   } else {
    //     ForwardRightBottom2LeftTop(bottom_data[batch_idx],
    //                                x_len, y_len,
    //                                g[batch_idx], 
    //                                c[batch_idx],
    //                                top_data[batch_idx]);
    //   }
    // }
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
    utils::Check(g.size(0) == 1, "LstmD2OptimizeLayer: gate problem."); 

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

  /*
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
  */

  /*
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
  */
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
// #if DEBUG
//     checkNanParams();
// #endif
    // mshadow::Tensor<xpu, 4> h    = top[0]->data;
    // mshadow::Tensor<xpu, 4> h_er = top[0]->diff;
    // mshadow::Tensor<xpu, 4> x    = bottom[0]->data;
    // mshadow::Tensor<xpu, 4> x_er = bottom[0]->diff;
    // mshadow::Tensor<xpu, 2> len  = bottom[0]->length;
    //     
    // begin_c_er = 0.; begin_h_er = 0.; g_er = 0.; c_er = 0.;
    // for (index_t batch_idx = 0; batch_idx < x.size(0); ++batch_idx) {
    //   int x_len = len[batch_idx][0];
    //   int y_len = len[batch_idx][1];
    //   if (!reverse) {
    //     BackpropForLeftTop2RightBottomLstm(x_len, y_len,
    //                                        h[batch_idx],
    //                                        h_er[batch_idx], 
    //                                        c[batch_idx],
    //                                        c_er[batch_idx],
    //                                        g[batch_idx],
    //                                        g_er[batch_idx],
    //                                        x[batch_idx],
    //                                        x_er[batch_idx]);
    //   } else {
    //     BackpropForRightBottom2LeftTopLstm(x_len, y_len,
    //                                        h[batch_idx],
    //                                        h_er[batch_idx], 
    //                                        c[batch_idx],
    //                                        c_er[batch_idx],
    //                                        g[batch_idx],
    //                                        g_er[batch_idx],
    //                                        x[batch_idx],
    //                                        x_er[batch_idx]);

    //   }
    // }
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
  //                "LstmD2OptimizeLayer: load tensor error.");
  //   int size = s0*s1*s2*s3;
  //   for (int i = 0; i < size; ++i) {
  //     t.dptr_[i] = data_root["value"][i].asFloat();
  //   }
  // }
  // void LoadParam() {
  //   utils::Printf("LstmD2OptimizeLayer: load params...\n");
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

  // TensorC2D run_x;
  TensorC2D run_x, run_c, run_h, run_input, run_g, run_i, \
            run_f_l, run_f_m, run_f_t, run_o, run_cc, \
            run_pre_c_l, run_pre_c_m, run_pre_c_t, \
            run_pre_h_l, run_pre_h_m, run_pre_h_t;
  TensorC4D run_pos; // 由于大家长短不一，所以每一个run中，特定表达存放的位置是不一样的，所以要保存一下
                     // 这个保存相对于run开始位置的相对位置

  // float grad_cut_off;
  // string param_file;
  duration<double> time_1, time_2, time_3, time_4;
  vector<int> run_max_len, run_begin_idx;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LSTM_LAYER_INL_HPP_
