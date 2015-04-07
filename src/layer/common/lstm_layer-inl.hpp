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

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);
    this->defaults["reverse"] = SettingV(false);
    
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "LstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LstmLayer:top size problem.");
    utils::Check(setting.count("d_mem"), "LstmLayer:setting problem.");
                  
    d_mem   = setting["d_mem"].iVal();
    d_input = bottom[0]->data.size(3);
    no_bias = setting["no_bias"].bVal();
    reverse = setting["reverse"].bVal();

    begin_h.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_c.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_h_er.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_c_er.Resize(mshadow::Shape2(1, d_mem), 0.f);

    this->params.resize(3);
    this->params[0].Resize(1, 1, d_input, 4*d_mem, true); // w
    this->params[1].Resize(1, 1, d_mem,   4*d_mem, true); // u
    this->params[2].Resize(1, 1, 1,       4*d_mem, true); // b
    
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
    utils::Check(bottom.size() == BottomNodeNum(), "LstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LstmLayer:top size problem.");
    
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);
    mshadow::Shape<4> shape_gate= mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem*4);

    std::cout << "Lstm io shape:" << std::endl;
    std::cout << shape_in[0] << "x" << shape_in[1] << "x" << shape_in[2] << "x" << shape_in[3] << std::endl;
    std::cout << shape_out[0] << "x" << shape_out[1] << "x" << shape_out[2] << "x" << shape_out[3] << std::endl;

    top[0]->Resize(shape_out);
    c.Resize(shape_out, 0.f);
    g.Resize(shape_gate, 0.f);
    c_er.Resize(shape_out, 0.f);
    g_er.Resize(shape_gate, 0.f);
  }

  void LocateBeginEnd(mshadow::Tensor<xpu, 2> seq, 
                      int &begin, int &end) { // input a 2D tensor, out put a sub 2d tensor, with 0 padding
    begin = seq.size(0);
    for (int i = 0; i < seq.size(0); ++i) {
      if (!isnan(seq[i][0])) { // the first number
          begin = i;
          break;
      }
    }
    end = seq.size(0);
    for (int i = begin; i < seq.size(0); ++i) {
      if (isnan(seq[i][0])) { // the first NAN
          end = i;
          break;
      }
    }
    utils::Check(begin < end && begin >= 0, "LstmLayer: input error."); 
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

  virtual void ForwardOneStep(Tensor2D pre_c, 
                              Tensor2D pre_h,
                              Tensor2D x,
                              Tensor2D cur_g,
                              Tensor2D cur_c,
                              Tensor2D cur_h) {
      Tensor2D w_data = this->params[0].data[0][0];
      Tensor2D u_data = this->params[1].data[0][0];
      Tensor2D b_data = this->params[2].data[0][0];

      Tensor2D i, f, o, cc;
      cur_g = dot(x, w_data);
      cur_g += dot(pre_h, u_data);
      if (!no_bias) {
        cur_g += b_data;
      }
      SplitGate(cur_g, i, f, o, cc);
      i = mshadow::expr::F<op::sigmoid>(i); // logi
      f = mshadow::expr::F<op::sigmoid>(f); // logi
      o = mshadow::expr::F<op::sigmoid>(o); // logi
      cc= mshadow::expr::F<op::tanh>(cc);   // tanh 

      cur_c = f * pre_c + i * cc;
      cur_h = o * mshadow::expr::F<op::tanh>(cur_c); // tanh
  }

  void ForwardLeft2Right(Tensor2D in, Tensor2D g, Tensor2D c, Tensor2D out) {
      int begin = 0, end = 0;
      LocateBeginEnd(in, begin, end);
      Tensor2D pre_c, pre_h;
      // not need any padding, begin h and c are set to 0
      for (index_t row_idx = begin; row_idx < end; ++row_idx) {
        if (row_idx == begin) {
          pre_c = begin_c; 
          pre_h = begin_h;
        } else {
          pre_c = c.Slice(row_idx-1, row_idx);
          pre_h = out.Slice(row_idx-1, row_idx);
        }
        ForwardOneStep(pre_c,
                       pre_h,
                       in.Slice(row_idx, row_idx+1),
                       g.Slice(row_idx, row_idx+1), 
                       c.Slice(row_idx, row_idx+1), 
                       out.Slice(row_idx, row_idx+1));
      }
  }
  void ForwardRight2Left(Tensor2D in, Tensor2D g, Tensor2D c, Tensor2D out) {
      int begin = 0, end = 0;
      LocateBeginEnd(in, begin, end);
      assert(begin >= 0 && begin < end);
      Tensor2D pre_c, pre_h;
      // not need any padding, begin h and c are set to 0
      for (int row_idx = end-1; row_idx >= begin; --row_idx) {
        if (row_idx == end-1) {
          pre_c = begin_c; 
          pre_h = begin_h;
        } else {
          pre_c = c.Slice(row_idx+1, row_idx+2);
          pre_h = out.Slice(row_idx+1, row_idx+2);
        }
        ForwardOneStep(pre_c,
                       pre_h,
                       in.Slice(row_idx, row_idx+1),
                       g.Slice(row_idx, row_idx+1), 
                       c.Slice(row_idx, row_idx+1), 
                       out.Slice(row_idx, row_idx+1));
      }
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
#if DEBUG
    checkNanParams();
#endif
    Tensor4D bottom_data = bottom[0]->data;
    Tensor4D top_data = top[0]->data;
    top_data = NAN; c = NAN, g = NAN; c_er = NAN; g_er = NAN;
    const index_t nbatch = bottom_data.size(0); 
    for (index_t i = 0; i < nbatch; ++i) {
        if (reverse) {
          ForwardLeft2Right(bottom_data[i][0], g[i][0], c[i][0], top_data[i][0]);
        } else {
          ForwardRight2Left(bottom_data[i][0], g[i][0], c[i][0], top_data[i][0]);
        }
    }
#if DEBUG
    checkNanParams();
#endif
  }

  // too tricky, may bring errors
  void SplitGate(Tensor2D g, Tensor2D &i, Tensor2D &f, Tensor2D &o, Tensor2D &cc) {
    utils::Check(g.shape_[0] == 1, "LstmLayer: gate problem."); 
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
                 Tensor2D x_er) {

    Tensor2D w_data = this->params[0].data[0][0];
    Tensor2D u_data = this->params[1].data[0][0];
    Tensor2D w_er = this->params[0].diff[0][0];
    Tensor2D u_er = this->params[1].diff[0][0];
    Tensor2D b_er = this->params[2].diff[0][0];
    
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
    x_er += dot(cur_g_er, w_data.T());

    // grad
    if (!no_bias) {
        b_er += cur_g_er;
    }
    w_er += dot(x.T(), cur_g_er); 
    u_er += dot(pre_h.T(), cur_g_er);
  }

  void BackpropForLeft2RightLstm(Tensor2D top_data, Tensor2D top_diff, 
                                 Tensor2D c, Tensor2D c_er, 
                                 Tensor2D g, Tensor2D g_er, 
                                 Tensor2D bottom_data, Tensor2D bottom_diff) {
      int begin = 0, end = 0;
      LocateBeginEnd(bottom_data, begin, end);

      Tensor2D pre_c, pre_h, pre_c_er, pre_h_er;
      for (int row_idx = end-1; row_idx >= begin; --row_idx) {
        if (row_idx == begin) {
            pre_c = begin_c;
            pre_h = begin_h;
            pre_c_er = begin_c_er;
            pre_h_er = begin_h_er;
        } else {
            pre_c = c.Slice(row_idx-1, row_idx);
            pre_h = top_data.Slice(row_idx-1, row_idx);
            pre_c_er = c_er.Slice(row_idx-1, row_idx);
            pre_h_er = top_diff.Slice(row_idx-1, row_idx);
        }
        BpOneStep(top_diff.Slice(row_idx, row_idx+1), 
                  pre_c,
                  pre_h,
                  bottom_data.Slice(row_idx, row_idx+1), 
                  g.Slice(row_idx, row_idx+1), 
                  c.Slice(row_idx, row_idx+1), 
                  top_data.Slice(row_idx, row_idx+1),
                  c_er.Slice(row_idx, row_idx+1),
                  g_er.Slice(row_idx, row_idx+1),
                  pre_c_er,
                  pre_h_er,
                  bottom_diff.Slice(row_idx, row_idx+1));
      }
  }
  void BackpropForRight2LeftLstm(Tensor2D top_data, Tensor2D top_diff, 
                                 Tensor2D c, Tensor2D c_er, 
                                 Tensor2D g, Tensor2D g_er, 
                                 Tensor2D bottom_data, Tensor2D bottom_diff) {
      int begin = 0, end = 0;
      LocateBeginEnd(bottom_data, begin, end);

      Tensor2D pre_c, pre_h, pre_c_er, pre_h_er;
      for (int row_idx = begin; row_idx < end; ++row_idx) {
        if (row_idx == end-1) {
            pre_c = begin_c;
            pre_h = begin_h;
            pre_c_er = begin_c_er;
            pre_h_er = begin_h_er;
        } else {
            pre_c = c.Slice(row_idx+1, row_idx+2);
            pre_h = top_data.Slice(row_idx+1, row_idx+2);
            pre_c_er = c_er.Slice(row_idx+1, row_idx+2);
            pre_h_er = top_diff.Slice(row_idx+1, row_idx+2);
        }
        BpOneStep(top_diff.Slice(row_idx, row_idx+1), 
                  pre_c,
                  pre_h,
                  bottom_data.Slice(row_idx, row_idx+1), 
                  g.Slice(row_idx, row_idx+1), 
                  c.Slice(row_idx, row_idx+1), 
                  top_data.Slice(row_idx, row_idx+1),
                  c_er.Slice(row_idx, row_idx+1),
                  g_er.Slice(row_idx, row_idx+1),
                  pre_c_er,
                  pre_h_er,
                  bottom_diff.Slice(row_idx, row_idx+1));
      }
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
    const index_t nbatch = bottom_data.size(0);
        
    begin_c_er = 0.; begin_h_er = 0.; g_er = 0.; c_er = 0.;

    for (index_t i = 0; i < nbatch; ++i) {
        if (!reverse) {
            BackpropForLeft2RightLstm(top_data[i][0], top_diff[i][0], c[i][0], c_er[i][0],
                                      g[i][0], g_er[i][0], bottom_data[i][0], bottom_diff[i][0]);
        } else {
            BackpropForRight2LeftLstm(top_data[i][0], top_diff[i][0], c[i][0], c_er[i][0],
                                      g[i][0], g_er[i][0], bottom_data[i][0], bottom_diff[i][0]);
        }
    }
#if DEBUG
    checkNanParams();
#endif
  }

 protected:
  int d_mem, d_input;
  bool no_bias, reverse; 
  mshadow::TensorContainer<xpu, 4> c, g, c_er, g_er;
  mshadow::TensorContainer<xpu, 2> begin_h, begin_c, begin_c_er, begin_h_er;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LSTM_LAYER_INL_HPP_
