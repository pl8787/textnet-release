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
    this->defaults["o_gate_bias_init"] = SettingV(0.f);
    this->defaults["f_gate_bias_init"] = SettingV(0.f);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["d_mem"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    this->defaults["reverse"] = SettingV();
    
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
    reverse = setting["reverse"].bVal();
    o_gate_bias_init = setting["o_gate_bias_init"].fVal();
    f_gate_bias_init = setting["f_gate_bias_init"].fVal();

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

    top[0]->Resize(shape_out, mshadow::Shape2(shape_out[0],2), true);

    // 这个地方对中间变量进行初始化
    run_pos.Resize(mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], 1), -1); 
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

    run_x.Resize(mshadow::Shape2(total_cnt, d_input), 0.f);
    run_c.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_h.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_input.Resize(mshadow::Shape2(total_cnt, d_input+d_mem*3), 0.f);
    run_g.Resize(mshadow::Shape2(total_cnt, d_mem*6), 0.f);
    run_i.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_f_l.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_f_m.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_f_t.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_o.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_cc.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_tanh_c.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_c_l.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_c_m.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_c_t.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_h_l.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_h_m.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_h_t.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);

    run_x_er.Resize(mshadow::Shape2(total_cnt, d_input), 0.f);
    run_c_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_h_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_input_er.Resize(mshadow::Shape2(total_cnt, d_input+d_mem*3), 0.f);
    run_g_er.Resize(mshadow::Shape2(total_cnt, d_mem*6), 0.f);
    run_i_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_f_l_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_f_m_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_f_t_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_o_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_cc_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_c_l_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_c_m_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_c_t_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_h_l_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_h_m_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
    run_pre_h_t_er.Resize(mshadow::Shape2(total_cnt, d_mem), 0.f);
  }

  bool is_nan(Tensor2D &t) {
    return is_nan(t.dptr_, t.size(0)*t.size(1));
  }

  bool is_nan(float *p, int l) {
    for (int i = 0; i < l; ++i) {
      if (isnan(p[i]))
        return true;
    }
    return false;
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
    utils::Assert(input.size(0) == h_l.size(0) && input.size(0) == x.size(0) && \
                  input.size(0) == h_m.size(0) && input.size(0) == h_t.size(0), "LstmD2OptimizeLayer: size error.");

    // 这个地方用memcpy直接优化，当然，这就要求内存必须连续存放了，一般使用没什么问题
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
  // 这个也是用于BP传error
  void split_input_batch(Tensor2D &x, 
                         Tensor2D &h_l, 
                         Tensor2D &h_m, 
                         Tensor2D &h_t, 
                         Tensor2D &input) {
    utils::Check(x.size(1)+h_l.size(1)+h_m.size(1)+h_t.size(1) == input.size(1), "LstmD2OptimizeLayer: size error.");
    utils::Assert(input.size(0) == h_l.size(0) && input.size(0) == x.size(0) && \
                  input.size(0) == h_m.size(0) && input.size(0) == h_t.size(0), "LstmD2OptimizeLayer: size error.");

    // 这个地方用memcpy直接优化，当然，这就要求内存必须连续存放了，一般使用没什么问题
    int batch_size = x.size(0);
    for (index_t row_idx = 0; row_idx < batch_size; ++row_idx) {
      float *p_src = input.dptr_ + row_idx*input.size(1);
      float *p_dst = x.dptr_ + row_idx*x.size(1);
      memcpy(p_dst, p_src, x.size(1)*sizeof(float));

      p_src += x.size(1);
      p_dst = h_l.dptr_ + row_idx*h_l.size(1);
      memcpy(p_dst, p_src, h_l.size(1)*sizeof(float));

      p_src += h_l.size(1);
      p_dst = h_m.dptr_ + row_idx*h_m.size(1);
      memcpy(p_dst, p_src, h_m.size(1)*sizeof(float));

      p_src += h_m.size(1);
      p_dst = h_t.dptr_ + row_idx*h_t.size(1);
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

  // 以前的这个函数不支持batch，现在改成batch的版本
  // 这个是bp的时候用的，是split的逆运算，传error的
  void concat_gate_batch(Tensor2D &g, 
                         Tensor2D &i, 
                         Tensor2D &f_l, 
                         Tensor2D &f_m, 
                         Tensor2D &f_t, 
                         Tensor2D &o, 
                         Tensor2D &cc) {
    int batch_size = g.size(0);
    for (index_t row_idx = 0; row_idx < batch_size; ++row_idx) {
      float *p_dst = g.dptr_ + row_idx*g.size(1);
      float *p_src = i.dptr_ + row_idx*i.size(1);
      memcpy(p_dst, p_src, i.size(1)*sizeof(float));

      p_dst += i.size(1);
      p_src = f_l.dptr_ + row_idx*f_l.size(1);
      memcpy(p_dst, p_src, f_l.size(1)*sizeof(float));

      p_dst += f_l.size(1);
      p_src = f_m.dptr_ + row_idx*f_m.size(1);
      memcpy(p_dst, p_src, f_m.size(1)*sizeof(float));

      p_dst += f_m.size(1);
      p_src = f_t.dptr_ + row_idx*f_t.size(1);
      memcpy(p_dst, p_src, f_t.size(1)*sizeof(float));

      p_dst += f_t.size(1);
      p_src = o.dptr_ + row_idx*o.size(1);
      memcpy(p_dst, p_src, o.size(1)*sizeof(float));

      p_dst += o.size(1);
      p_src = cc.dptr_ + row_idx*cc.size(1);
      memcpy(p_dst, p_src, cc.size(1)*sizeof(float));
    }
  }

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
                              Tensor2D cur_tanh_c,
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


    cur_i   = mshadow::expr::F<op::sigmoid_lookup>(cur_i);   // logi
    cur_f_l = mshadow::expr::F<op::sigmoid_lookup>(cur_f_l); // logi
    cur_f_m = mshadow::expr::F<op::sigmoid_lookup>(cur_f_m); // logi
    cur_f_t = mshadow::expr::F<op::sigmoid_lookup>(cur_f_t); // logi
    cur_o   = mshadow::expr::F<op::sigmoid_lookup>(cur_o);   // logi
    cur_cc  = mshadow::expr::F<op::tanh_lookup>(cur_cc);     // tanh 
    
    cur_c = cur_f_l * pre_c_l + cur_f_m * pre_c_m + cur_f_t * pre_c_t + cur_i * cur_cc;
    cur_tanh_c = mshadow::expr::F<op::tanh_lookup>(cur_c); // tanh
    cur_h = cur_o * cur_tanh_c;
    high_resolution_clock::time_point e_time_3 = high_resolution_clock::now();
    time_3 += duration_cast<duration<double>>(e_time_3 - b_time_3);
  }

  // 这个外围函数也要好好设计一下，目前的思路是这样的，每一次不仅是不同的example之间的并行，
  // 也把同一个example里面没有数据依赖性的地方一起算了
  // x: (x_max_len, y_max_len, d_input)
  void ForwardLeftTop2RightBottom(Tensor4D &x, vector<int> &x_lens, vector<int> &y_lens, Tensor4D &top) {
    int batch_size= x.size(0);
    int max_x_len = x.size(1);
    int max_y_len = x.size(2);
    int max_run   = max_x_len + max_y_len - 1;

    run_real_len.clear();
    // 注意：这个地方为了效率并没有进行清零操作，需要密切注意内存的值
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
      Tensor2D cur_tanh_c= run_tanh_c.Slice(begin_idx, end_idx);
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
          cur_cnt += 1;
        }
      }
      if (cur_cnt == 0)  // 这个说明已经不用再循环了
          break; 
      run_real_len.push_back(cur_cnt);

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
                     cur_g.Slice(0, cur_cnt), 
                     cur_i.Slice(0, cur_cnt),
                     cur_f_l.Slice(0, cur_cnt),
                     cur_f_m.Slice(0, cur_cnt),
                     cur_f_t.Slice(0, cur_cnt),
                     cur_o.Slice(0, cur_cnt),
                     cur_cc.Slice(0, cur_cnt),
                     cur_tanh_c.Slice(0, cur_cnt),
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
  }

  // x: (x_max_len, y_max_len, d_input)
  void BackpropForLeftTop2RightBottom(Tensor4D &top_er, Tensor4D &bottom_data, Tensor4D &bottom_er, 
                                      vector<int> &x_lens, vector<int> &y_lens) {
    int batch_size= bottom_data.size(0);
    int max_x_len = bottom_data.size(1);
    int max_y_len = bottom_data.size(2);
    int max_run   = max_x_len + max_y_len - 1;

    // 给c_er和h_er赋初始值
    run_c_er = 0.f;
    run_h_er = 0.f;

    high_resolution_clock::time_point b_time_4 = high_resolution_clock::now();
    // 先要把所有的top_er，拷贝到cur_h_er中去
    // 然后我们就在cur_h_er中不停更新
    // 这地方cur_h_er也不进行清零，直接用top_er覆盖，考虑好边界问题之后逻辑上没有问题
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int x_len = x_lens[batch_idx];
      int y_len = y_lens[batch_idx];
      for (int row_idx = 0; row_idx < x_len; ++row_idx) {
        for (int col_idx = 0; col_idx < y_len; ++col_idx) {
          int run_idx = row_idx + col_idx;
          int pos = run_pos[batch_idx][row_idx][col_idx][0];
          int cur_run_begin_idx = run_begin_idx[run_idx];
          run_h_er[cur_run_begin_idx+pos] = mshadow::expr::F<op::identity>(top_er[batch_idx][row_idx][col_idx]);
        }
      }
    }

    // 然后开始BP
    for (int run_idx = max_run-1; run_idx >= 0; --run_idx) { // 这是一次forward的执行
      if (run_idx >= run_real_len.size())  // 说明这个run不需要执行
          continue;
      int cur_run_real_len = run_real_len[run_idx];
      int begin_idx = run_begin_idx[run_idx];
      int end_idx   = begin_idx + cur_run_real_len;
      Tensor2D cur_x_er  = run_x_er.Slice(begin_idx, end_idx);
      Tensor2D cur_c_er  = run_c_er.Slice(begin_idx, end_idx);
      Tensor2D cur_h_er  = run_h_er.Slice(begin_idx, end_idx);
      Tensor2D cur_g_er  = run_g_er.Slice(begin_idx, end_idx);
      Tensor2D cur_input    = run_input.Slice(begin_idx, end_idx);
      Tensor2D cur_input_er = run_input_er.Slice(begin_idx, end_idx);
      Tensor2D cur_i     = run_i.Slice(begin_idx, end_idx);
      Tensor2D cur_i_er  = run_i_er.Slice(begin_idx, end_idx);
      Tensor2D cur_f_l   = run_f_l.Slice(begin_idx, end_idx);
      Tensor2D cur_f_l_er= run_f_l_er.Slice(begin_idx, end_idx);
      Tensor2D cur_f_m   = run_f_m.Slice(begin_idx, end_idx);
      Tensor2D cur_f_m_er= run_f_m_er.Slice(begin_idx, end_idx);
      Tensor2D cur_f_t   = run_f_t.Slice(begin_idx, end_idx);
      Tensor2D cur_f_t_er= run_f_t_er.Slice(begin_idx, end_idx);
      Tensor2D cur_o     = run_o.Slice(begin_idx, end_idx);
      Tensor2D cur_o_er  = run_o_er.Slice(begin_idx, end_idx);
      Tensor2D cur_cc    = run_cc.Slice(begin_idx, end_idx);
      Tensor2D cur_cc_er = run_cc_er.Slice(begin_idx, end_idx);
      Tensor2D cur_tanh_c= run_tanh_c.Slice(begin_idx, end_idx);
      Tensor2D pre_c_l   = run_pre_c_l.Slice(begin_idx, end_idx);
      Tensor2D pre_c_l_er= run_pre_c_l_er.Slice(begin_idx, end_idx);
      Tensor2D pre_c_m   = run_pre_c_m.Slice(begin_idx, end_idx);
      Tensor2D pre_c_m_er= run_pre_c_m_er.Slice(begin_idx, end_idx);
      Tensor2D pre_c_t   = run_pre_c_t.Slice(begin_idx, end_idx);
      Tensor2D pre_c_t_er= run_pre_c_t_er.Slice(begin_idx, end_idx);
      Tensor2D pre_h_l_er= run_pre_h_l_er.Slice(begin_idx, end_idx);
      Tensor2D pre_h_m_er= run_pre_h_m_er.Slice(begin_idx, end_idx);
      Tensor2D pre_h_t_er= run_pre_h_t_er.Slice(begin_idx, end_idx);
      BpOneStep(cur_h_er, // 输入的时候已经存储当前节点的所有error
                pre_c_l, 
                pre_c_m,
                pre_c_t,
                cur_g_er, // 这个实际上就是所有i,f,o,cc的er的拼接
                cur_i,
                cur_i_er,
                cur_f_l,
                cur_f_l_er,
                cur_f_m,
                cur_f_m_er,
                cur_f_t,
                cur_f_t_er,
                cur_o,
                cur_o_er,
                cur_cc,
                cur_cc_er,
                cur_c_er, // 这里面也必须要存储好之前已经传过来的error
                cur_tanh_c,
                cur_input,
                cur_input_er, // 注意，以下八项传入的时候不存储任何值，直接覆盖，在外层才考虑他们的依赖关系
                pre_c_l_er, 
                pre_c_m_er,
                pre_c_t_er,
                pre_h_l_er,
                pre_h_m_er,
                pre_h_t_er,
                cur_x_er);

      // Bp完之后，我们把计算得到的er，整合到一起，这地方注意er一方面是初始值的问题，一方面不要覆盖了
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
          if (cnt > 1) { // 这个设置剩余的left
            int pre_run_begin_idx = run_begin_idx[run_idx-1]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-1][begin_y][0]; // 这个是run内某个batch的开始位置索引
            begin_pos += pre_run_begin_idx; // 绝对位置
            run_c_er.Slice(begin_pos, begin_pos+cnt-1) += pre_c_l_er.Slice(cur_cnt+1, cur_cnt+cnt);
            run_h_er.Slice(begin_pos, begin_pos+cnt-1) += pre_h_l_er.Slice(cur_cnt+1, cur_cnt+cnt);
          }
        } else { // 所有的都有left
          int pre_run_begin_idx = run_begin_idx[run_idx-1];
          int begin_pos = run_pos[batch_idx][begin_x][begin_y-1][0]; // 这个是run内某个batch的开始位置索引
          begin_pos += pre_run_begin_idx;
          run_c_er.Slice(begin_pos, begin_pos+cnt) += pre_c_l_er.Slice(cur_cnt, cur_cnt+cnt);
          run_h_er.Slice(begin_pos, begin_pos+cnt) += pre_h_l_er.Slice(cur_cnt, cur_cnt+cnt);
        }

        // 再搞定top
        if (end_x == 0) { // 最后一个没有top，设置为0
          if (cnt > 1) { // 这个设置剩余的left
            int pre_run_begin_idx = run_begin_idx[run_idx-1]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-1][begin_y][0]; // 这个是run内某个batch的开始位置索引
            begin_pos = pre_run_begin_idx + begin_pos;
            run_c_er.Slice(begin_pos, begin_pos+cnt-1) += pre_c_t_er.Slice(cur_cnt, cur_cnt+cnt-1);
            run_h_er.Slice(begin_pos, begin_pos+cnt-1) += pre_h_t_er.Slice(cur_cnt, cur_cnt+cnt-1);
          }
        } else { // 所有的都有top
          int pre_run_begin_idx = run_begin_idx[run_idx-1];
          int begin_pos = run_pos[batch_idx][begin_x-1][begin_y][0]; // 这个是run内某个batch的开始位置索引
          begin_pos = pre_run_begin_idx + begin_pos;
          run_c_er.Slice(begin_pos, begin_pos+cnt) += pre_c_t_er.Slice(cur_cnt, cur_cnt+cnt);
          run_h_er.Slice(begin_pos, begin_pos+cnt) += pre_h_t_er.Slice(cur_cnt, cur_cnt+cnt);
        }
        // 最后搞定left top
        // 先处理好第一个与最后一个
        // 处理中间的
        if (begin_y == 0 && end_x == 0) { // 一头一尾都没有
          if (cnt > 2) {
            int pre_run_begin_idx = run_begin_idx[run_idx-2]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-2][begin_y][0]; // 这个是run内某个batch的开始位置索引
            begin_pos = pre_run_begin_idx + begin_pos;
            run_c_er.Slice(begin_pos, begin_pos+cnt-2) += pre_c_m_er.Slice(cur_cnt+1, cur_cnt+cnt-1);
            run_h_er.Slice(begin_pos, begin_pos+cnt-2) += pre_h_m_er.Slice(cur_cnt+1, cur_cnt+cnt-1);
          }
        } else if (begin_y == 0) { // 头没有
          if (cnt > 1) {
            int pre_run_begin_idx = run_begin_idx[run_idx-2]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-2][begin_y][0]; // 这个是run内某个batch的开始位置索引
            begin_pos = pre_run_begin_idx + begin_pos;
            run_c_er.Slice(begin_pos, begin_pos+cnt-1) += pre_c_m_er.Slice(cur_cnt+1, cur_cnt+cnt);
            run_h_er.Slice(begin_pos, begin_pos+cnt-1) += pre_h_m_er.Slice(cur_cnt+1, cur_cnt+cnt);
          }
        } else if (end_x == 0) { // 尾没有
          if (cnt > 1) {
            int pre_run_begin_idx = run_begin_idx[run_idx-2]; // 这个是整个run的开始的位置索引
            int begin_pos = run_pos[batch_idx][begin_x-1][begin_y-1][0]; // 这个是run内某个batch的开始位置索引
            begin_pos = pre_run_begin_idx + begin_pos;
            run_c_er.Slice(begin_pos, begin_pos+cnt-1) += pre_c_m_er.Slice(cur_cnt, cur_cnt+cnt-1);
            run_h_er.Slice(begin_pos, begin_pos+cnt-1) += pre_h_m_er.Slice(cur_cnt, cur_cnt+cnt-1);
          }
        } else {
          int pre_run_begin_idx = run_begin_idx[run_idx-2]; // 这个是整个run的开始的位置索引
          int begin_pos = run_pos[batch_idx][begin_x-1][begin_y-1][0]; // 这个是run内某个batch的开始位置索引
          begin_pos = pre_run_begin_idx + begin_pos;
          run_c_er.Slice(begin_pos, begin_pos+cnt) += pre_c_m_er.Slice(cur_cnt, cur_cnt+cnt);
          run_h_er.Slice(begin_pos, begin_pos+cnt) += pre_h_m_er.Slice(cur_cnt, cur_cnt+cnt);
        }

        cur_cnt += cnt;
      }
    }

    // 把run_x_er写入到bottom_diff中去
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      int x_len = x_lens[batch_idx];
      int y_len = y_lens[batch_idx];
      for (int row_idx = 0; row_idx < x_len; ++row_idx) {
        for (int col_idx = 0; col_idx < y_len; ++col_idx) {
          int run_idx = row_idx + col_idx;
          int pos = run_pos[batch_idx][row_idx][col_idx][0];
          int cur_run_begin_idx = run_begin_idx[run_idx];
          // 注意，这个地方一定是+=，因为bottom node中可能已经存放梯度了
          bottom_er[batch_idx][row_idx][col_idx] += mshadow::expr::F<op::identity>(run_x_er[cur_run_begin_idx+pos]); 
        }
      }
    }

    high_resolution_clock::time_point e_time_4 = high_resolution_clock::now();
    time_4 += duration_cast<duration<double>>(e_time_4 - b_time_4);
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
// #if DEBUG
//     checkNanParams();
// #endif
    Tensor4D bottom_data = bottom[0]->data;
    Tensor2D bottom_len  = bottom[0]->length;
    Tensor4D top_data    = top[0]->data;
    top_data = 0.f;

    utils::Check(bottom_len.size(0) == bottom_data.size(0) && 
                 bottom_len.size(1) == 2, "LstmD2OptimizeLayer: input length error.");
    top[0]->length = mshadow::expr::F<op::identity>(bottom[0]->length);
    run_pos = -1.f;

    high_resolution_clock::time_point b_time_1 = high_resolution_clock::now();
    vector<int> x_len, y_len;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      x_len.push_back(bottom_len[batch_idx][0]);
      y_len.push_back(bottom_len[batch_idx][1]);
    }
    ForwardLeftTop2RightBottom(bottom_data, x_len, y_len, top_data);

    high_resolution_clock::time_point e_time_1 = high_resolution_clock::now();
    time_1 += duration_cast<duration<double>>(e_time_1 - b_time_1);
	// utils::Printf("\tLSTM D2 OPTIMIZE Time:%fs,%fs,%f\n", time_1.count(), time_2.count(), time_3.count()); 

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
   
  // 这个地方的方式还是和以前的算法一样，认为当前层的c_er什么的已经计算好了，然后往前传
  // 这个地方一定要注意梯度不要传漏了，或者拷贝过程中覆盖或者丢失了
  void BpOneStep(Tensor2D cur_h_er, // 只读，输入的时候要求已经存储当前节点的所有er，需要初始化，所以调用此函数前保证er已到位
                 Tensor2D pre_c_l, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D pre_c_m, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D pre_c_t, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D cur_g_er, // 写覆盖，这个实际上就是所有i,f,o,cc的er的拼接，为了进一步往前求input和w的er的
                 Tensor2D cur_i, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D cur_i_er, // 写覆盖，不存在梯度累积问题
                 Tensor2D cur_f_l, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D cur_f_l_er, // 写覆盖，不存在梯度累积问题
                 Tensor2D cur_f_m, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D cur_f_m_er, // 写覆盖，不存在梯度累积问题
                 Tensor2D cur_f_t, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D cur_f_t_er, // 写覆盖，不存在梯度累积问题
                 Tensor2D cur_o, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D cur_o_er, // 写覆盖，不存在梯度累积问题
                 Tensor2D cur_cc, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D cur_cc_er, // 写覆盖，不存在梯度累积问题
                 Tensor2D cur_c_er, // 注意，梯度累积，这里面也必须要存储好之前已经传过来的error
                 Tensor2D cur_tanh_c, // 只读，在forward的时候已经保存了，直接传进来
                 Tensor2D input, // 只读，在forward的时候已经保存了，直接传进来，用于计算w的er
                 Tensor2D input_er, // 注意，以下八项为写覆盖，传入的时候不存储任何值，在外层才考虑他们的依赖关系
                 Tensor2D pre_c_l_er, // 这些是为上一层准备的，所以计算完之后写入到上一层的h_er和c_er中，这样就可以继续BP了
                 Tensor2D pre_c_m_er,
                 Tensor2D pre_c_t_er,
                 Tensor2D pre_h_l_er,
                 Tensor2D pre_h_m_er,
                 Tensor2D pre_h_t_er,
                 Tensor2D cur_x_er) { // cur_x_er也要注意，这个也是写覆盖的，然后在外层要和bottom_diff中可能包含的梯度进行累加

    Tensor2D w_data = this->params[0].data_d2_reverse();
    Tensor2D w_er   = this->params[0].diff_d2_reverse();
    Tensor1D b_er   = this->params[1].diff_d1_reverse();

    // 第一步，先根据cur_h_er来BP得到cur_c和cur_o的error
    cur_o_er = mshadow::expr::F<op::sigmoid_grad>(cur_o) * (cur_h_er * cur_tanh_c); // logi
    cur_c_er += mshadow::expr::F<op::tanh_grad>(cur_tanh_c) * (cur_h_er * cur_o); // NOTICE: +=，至此cur_c_er计算完成

    cur_i_er = mshadow::expr::F<op::sigmoid_grad>(cur_i) * (cur_c_er * cur_cc);    // logi
    cur_cc_er = mshadow::expr::F<op::tanh_grad>(cur_cc) * (cur_c_er * cur_i);      // tanh
    pre_c_l_er = cur_c_er * cur_f_l; 
    pre_c_m_er = cur_c_er * cur_f_m;
    pre_c_t_er = cur_c_er * cur_f_t;
    cur_f_l_er = mshadow::expr::F<op::sigmoid_grad>(cur_f_l) * (cur_c_er * pre_c_l); // logi
    cur_f_m_er = mshadow::expr::F<op::sigmoid_grad>(cur_f_m) * (cur_c_er * pre_c_m); // logi
    cur_f_t_er = mshadow::expr::F<op::sigmoid_grad>(cur_f_t) * (cur_c_er * pre_c_t); // logi
    concat_gate_batch(cur_g_er, cur_i_er, cur_f_l_er, cur_f_m_er, cur_f_t_er, cur_o_er, cur_cc_er);
   
    input_er = dot(cur_g_er, w_data.T());
    split_input_batch(cur_x_er, pre_h_l_er, pre_h_m_er, pre_h_t_er, input_er);

    // grad
    if (!no_bias) {
      // b_er += cur_g_er;
      b_er += sum_rows(cur_g_er);
    }
    w_er += dot(input.T(), cur_g_er); 
  }

  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

    Tensor2D bottom_len  = bottom[0]->length;
    vector<int> x_lens, y_lens;
    for (index_t batch_idx = 0; batch_idx < bottom[0]->data.size(0); ++batch_idx) {
      x_lens.push_back(bottom_len[batch_idx][0]);
      y_lens.push_back(bottom_len[batch_idx][1]);
    }

    BackpropForLeftTop2RightBottom(top[0]->diff, bottom[0]->data, bottom[0]->diff, x_lens, y_lens);
	// utils::Printf("\tLSTM D2 OPTIMIZE BP Time:%fs,%fs,%f\n", time_4.count(), time_5.count(), time_6.count()); 
  }

 protected:
  int d_mem, d_input;
  bool no_bias, reverse;
  float o_gate_bias_init;
  float f_gate_bias_init;

  // 这些变量记录的都是计算当前位置的表达需要的东西，需要前一个的c和l，所以pre_c和pre_h中存放的都是拼接好的context
  TensorC2D run_x, run_x_er,\
            run_c, run_c_er,\
            run_h, run_h_er,\
            run_input, run_input_er,\
            run_g, run_g_er,\
            run_i, run_i_er,\
            run_f_l, run_f_l_er,\
            run_f_m, run_f_m_er,\
            run_f_t, run_f_t_er,\
            run_o, run_o_er,\
            run_cc, run_cc_er,\
            run_tanh_c,\
            run_pre_c_l, run_pre_c_l_er,\
            run_pre_c_m, run_pre_c_m_er,\
            run_pre_c_t, run_pre_c_t_er,\
            run_pre_h_l, run_pre_h_l_er,\
            run_pre_h_m, run_pre_h_m_er,\
            run_pre_h_t, run_pre_h_t_er;

  TensorC4D run_pos; // 由于大家长短不一，所以每一个run中，特定表达存放的位置是不一样的，所以要保存一下
                     // 这个保存相对于run开始位置的相对位置

  // float grad_cut_off;
  // string param_file;
  duration<double> time_1, time_2, time_3, time_4, time_5, time_6;
  vector<int> run_max_len, run_begin_idx, run_real_len; // run_real_len是因为每个run都是变长的，这个记录他的真实长度
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LSTM_D2_OPTIMIZE_LAYER_INL_HPP_
