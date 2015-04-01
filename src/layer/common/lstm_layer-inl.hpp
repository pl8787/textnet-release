#ifndef TEXTNET_LAYER_LSTM_LAYER_INL_HPP_
#define TEXTNET_LAYER_LSTM_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"
#include <cassert>

namespace textnet {
namespace layer {

template<typename xpu>
class LstmLayer : public Layer<xpu> {
 public:
  LstmLayer(LayerType type) { this->layer_type = type; }
  virtual ~LstmLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 3; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(),
                 "LstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                 "LstmLayer:top size problem.");
                  
    d_mem   = setting["d_mem"].i_val;
    d_input = setting["d_input"].i_val;
    no_bias = setting["no_bias"].b_val;

    begin_h.Resize(mshadow::Shape2(1, d_mem));
    begin_c.Resize(mshadow::Shape2(1, d_mem));
    begin_h_er.Resize(mshadow::Shape2(1, d_mem));
    begin_c_er.Resize(mshadow::Shape2(1, d_mem));

    this->params.resize(3);
    this->params[0].Resize(1, 1, d_input, 4*d_mem, true); // w
    this->params[1].Resize(1, 1, d_mem,   4*d_mem, true); // u
    this->params[2].Resize(1, 1, 1,       4*d_mem, true); // b
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].m_val;
    std::map<std::string, SettingV> &u_setting = *setting["u_filler"].m_val;
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].m_val;
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].i_val, w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(u_setting["init_type"].i_val, u_setting, this->prnd_);
    this->params[2].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].i_val, b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    this->params[2].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].m_val;
    std::map<std::string, SettingV> &u_updater = *setting["u_updater"].m_val;
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].m_val;

    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].i_val, w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(u_updater["updater_type"].i_val, u_updater, this->prnd_);
    this->params[2].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].i_val, b_updater, this->prnd_);
  }
  
  // bottom should be padded with only one zero on both sides
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "LstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LstmLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);
    mshadow::Shape<4> shape_gate = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem*4);

    std::cout << shape_in[0] << "x" << shape_in[1] << "x" << shape_in[2] << "x" << shape_in[3] << std::endl;
    std::cout << shape_out[0] << "x" << shape_out[1] << "x" << shape_out[2] << "x" << shape_out[3] << std::endl;

    top[0]->Resize(shape_out);
    c.Resize(shape_out);
    g.Resize(shape_gate);
    c_er.Resize(shape_out);
    g_er.Resize(shape_gate);
    
    // temp_col_.Resize(mshadow::Shape2(channel_in*kernel_x*kernel_y, shape_out[2]*shape_out[3]));
    // Share the memory
    // temp_dif_ = temp_col_;
    // temp_data_.Resize(mshadow::Shape2(channel_out, shape_out[2]*shape_out[3]));
  }

  void LocateBeginEnd(mshadow::Tensor<xpu, 2> seq, 
                      int &begin, int &end) { // input a 2D tensor, out put a sub 2d tensor, with 0 padding
    for (int i = 0; i < seq.size(0); ++i) {
      if (!isnan(seq[i][0])) {
          begin = i;
          break;
      }
    }
    for (int i = seq.size(0)-1; i >= 0; --i) {
      if (!isnan(seq[i][0])) {
          end = i + 1;
          break;
      }
    }
    // for (index_t i = 0; i < seq.shape_[0]*seq.shape_[1]; ++i) {
    //   if (seq.dptr_[i] != 0.) {
    //     begin = i / seq.shape_[1];
    //     break;
    //   }
    // }

    // for (int i = (seq.shape_[0]*seq.shape_[1]-1); i >= 0; --i) {
    //   if (seq.dptr_[i] != 0.) {
    //     end = (i / seq.shape_[1]) + 1;
    //     break;
    //   }
    // }
    utils::Check(begin < end && begin >= 0, "LstmLayer: input error."); 
  }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;

  virtual void ForwardOneStep(Tensor2D pre_c, 
                              Tensor2D pre_h,
                              Tensor2D x,
                              Tensor2D cur_g,
                              Tensor2D cur_c,
                              Tensor2D cur_h) {
      // assert(false); // check again
      Tensor2D w_data = this->params[0].data[0][0];
      Tensor2D u_data = this->params[1].data[0][0];
      Tensor2D b_data = this->params[2].data[0][0];

      Tensor2D i, f, o, cc;
      cur_g = dot(x, w_data);
      cur_g += dot(pre_h, u_data);
      cur_g += b_data;
      SplitGate(cur_g, i, f, o, cc);
      i = mshadow::expr::F<op::sigmoid>(i); // logi
      f = mshadow::expr::F<op::sigmoid>(f); // logi
      o = mshadow::expr::F<op::sigmoid>(o); // logi
      cc= mshadow::expr::F<op::tanh>(cc);   // tanh 

      cur_c = f * pre_c + i * cc;
      cur_h = o * mshadow::expr::F<op::tanh>(cur_c); // tanh
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D bottom_data = bottom[0]->data;
    Tensor4D top_data = top[0]->data;
    top_data = NAN; c = NAN, g = NAN; c_er = NAN; g_er = NAN;
    const index_t nbatch = bottom_data.size(0); 
    Tensor2D pre_c, pre_h;
    for (index_t i = 0; i < nbatch; ++i) {
      int begin = 0, end = 0;
      LocateBeginEnd(bottom_data[i][0], begin, end);
      assert(begin >= 0 && begin < end);
      // not need any padding, begin h and c are set to 0
      for (index_t row_idx = begin; row_idx < end; ++row_idx) {
        if (row_idx == begin) {
          pre_c = begin_c; 
          pre_h = begin_h;
        } else {
          pre_c = c[i][0].Slice(row_idx-1, row_idx);
          pre_h = top_data[i][0].Slice(row_idx-1, row_idx);
        }
        ForwardOneStep(pre_c,
                       pre_h,
                       bottom_data[i][0].Slice(row_idx, row_idx+1),
                       g[i][0].Slice(row_idx, row_idx+1), 
                       c[i][0].Slice(row_idx, row_idx+1), 
                       top_data[i][0].Slice(row_idx, row_idx+1));
      }
      
    }
    //if (no_bias == 0) {
      // add bias, broadcast bias to dim 1: channel
      // top_data += broadcast<1>(bias_data, top_data.shape_);
    // }
  }

  void SplitGate(Tensor2D g, Tensor2D &i, Tensor2D &f, Tensor2D &o, Tensor2D &cc) {
    utils::Check(g.shape_[0] == 1, "LstmLayer: gate problem."); 
    // Tensor1D g_1D = reshape(g, mshadow::Shape1(g.shape_[1]));
    i.shape_ = mshadow::Shape2(1, d_mem);
    f.shape_ = mshadow::Shape2(1, d_mem);
    o.shape_ = mshadow::Shape2(1, d_mem);
    cc.shape_= mshadow::Shape2(1, d_mem);

    i.dptr_ = g.dptr_;
    f.dptr_ = g.dptr_ + 1 * d_mem;
    o.dptr_ = g.dptr_ + 2 * d_mem;
    cc.dptr_= g.dptr_ + 3 * d_mem;
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
                 Tensor2D x_er,
                 Tensor2D w_er,
                 Tensor2D u_er,
                 Tensor2D b_er) {

    Tensor2D w_data = this->params[0].data[0][0];
    Tensor2D u_data = this->params[1].data[0][0];
    
    Tensor2D i, f, o, cc, i_er, f_er, o_er, cc_er;
    SplitGate(cur_g, i, f, o, cc);
    SplitGate(cur_g_er, i_er, f_er, o_er, cc_er);

    mshadow::TensorContainer<xpu, 2> tanhc(cur_c.shape_);
    tanhc = mshadow::expr::F<op::tanh>(cur_c);
    o_er = mshadow::expr::F<op::sigmoid_grad>(o) * (cur_h_er * tanhc); // logi
    cur_c_er += mshadow::expr::F<op::tanh_grad>(tanhc) * (cur_h_er * o);

    i_er = mshadow::expr::F<op::sigmoid_grad>(i) * (cur_c_er * cc);    // logi
    cc_er = mshadow::expr::F<op::tanh_grad>(cc) * (cur_c_er * i);      // tanh
    pre_c_er = cur_c_er * f;
    f_er = mshadow::expr::F<op::sigmoid_grad>(f) * (cur_c_er * pre_c); // logi

    pre_h_er += dot(cur_g_er, u_data.T());
    x_er = dot(cur_g_er, w_data.T());

    // grad
    b_er += cur_g_er;
    w_er += dot(x.T(), cur_g_er); 
    u_er += dot(pre_h.T(), cur_g_er);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> w_diff = this->params[0].diff[0][0];
    mshadow::Tensor<xpu, 2> u_diff = this->params[1].diff[0][0];
    mshadow::Tensor<xpu, 2> b_diff = this->params[2].diff[0][0];
    const index_t nbatch = bottom_data.size(0);
        
    // if (!no_bias && this->prop_grad[1]) {
    //   bias_diff = sumall_except_dim<1>(top_diff);
    // }
    
	w_diff = 0.; u_diff = 0.; b_diff = 0.; bottom_diff = 0.;
    begin_c_er = 0.; begin_h_er = 0.; g_er = 0.; c_er = 0.;

    Tensor2D pre_c, pre_h, pre_c_er, pre_h_er;
    for (index_t i = 0; i < nbatch; ++i) {
      int begin = 0, end = 0;
      LocateBeginEnd(bottom_data[i][0], begin, end);
      assert(begin >= 0 && begin < end); 
      for (int row_idx = end-1; row_idx >= begin; --row_idx) {
          if (row_idx == begin) {
              pre_c = begin_c;
              pre_h = begin_h;
              pre_c_er = begin_c_er;
              pre_h_er = begin_h_er;
          } else {
              pre_c = c[i][0].Slice(row_idx-1, row_idx);
              pre_h = top_data[i][0].Slice(row_idx-1, row_idx);
              pre_c_er = c_er[i][0].Slice(row_idx-1, row_idx);
              pre_h_er = top_diff[i][0].Slice(row_idx-1, row_idx);
          }
          BpOneStep(top_diff[i][0].Slice(row_idx, row_idx+1), 
                    pre_c,
                    pre_h,
                    bottom_data[i][0].Slice(row_idx, row_idx+1), 
                    g[i][0].Slice(row_idx, row_idx+1), 
                    c[i][0].Slice(row_idx, row_idx+1), 
                    top_data[i][0].Slice(row_idx, row_idx+1),
                    c_er[i][0].Slice(row_idx, row_idx+1),
                    g_er[i][0].Slice(row_idx, row_idx+1),
                    pre_c_er,
                    pre_h_er,
                    bottom_diff[i][0].Slice(row_idx, row_idx+1), 
                    w_diff, u_diff, b_diff);
      }
    }
  }

 protected:
  int d_mem, d_input;
  bool no_bias; // to do 
  mshadow::TensorContainer<xpu, 4> c, g, c_er, g_er;
  mshadow::TensorContainer<xpu, 2> begin_h, begin_c, begin_c_er, begin_h_er;

  // mshadow::TensorContainer<xpu, 2> temp_col_;
  // mshadow::TensorContainer<xpu, 2> temp_dif_;
  // mshadow::TensorContainer<xpu, 2> temp_data_;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LSTM_LAYER_INL_HPP_
