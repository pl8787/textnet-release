#ifndef TEXTNET_LAYER_MAX_RECURRENT_LAYER_INL_HPP_
#define TEXTNET_LAYER_MAX_RECURRENT_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"
#include <cassert>

namespace textnet {
namespace layer {

template<typename xpu>
class MaxRecurrentLayer : public Layer<xpu> {
 public:
  MaxRecurrentLayer(LayerType type) { this->layer_type = type; }
  virtual ~MaxRecurrentLayer(void) {}
  
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
    this->defaults["reverse"] = SettingV(false);
    // this->defaults["input_transform"] = SettingV(true);
    this->defaults["nonlinear_type"] = SettingV("tanh"); // other options "sigmoid", "rectifier"
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["d_mem"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["u_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["t_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["u_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    this->defaults["t_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(), "MaxRecurrentLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MaxRecurrentLayer:top size problem.");
                  
    d_mem   = setting["d_mem"].iVal();
    d_input = bottom[0]->data.size(3);
    no_bias = setting["no_bias"].bVal();
    reverse = setting["reverse"].bVal();
    // input_transform = setting["input_transform"].bVal();
    nonlinear_type = setting["nonlinear_type"].sVal();
    // if (!input_transform) {
    // utils::Check(d_input == d_mem, "MaxRecurrentLayer:input does not match with memory, need transform.");
    // }

    begin_h.Resize(mshadow::Shape2(1, d_mem), 0.f);
    begin_h_er.Resize(mshadow::Shape2(1, d_mem), 0.f);

    this->params.resize(4);
    this->params[0].Resize(d_input, d_mem, 1, 1, true); // w
    this->params[1].Resize(d_mem,   d_mem, 1, 1, true); // u
    this->params[2].Resize(d_mem,       1, 1, 1, true); // b
    this->params[3].Resize(d_input, d_mem, 1, 1, true); // w_trans
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &u_setting = *setting["u_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    std::map<std::string, SettingV> &t_setting = *setting["t_filler"].mVal(); // trans
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(), w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(u_setting["init_type"].iVal(), u_setting, this->prnd_);
    this->params[2].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), b_setting, this->prnd_);
    this->params[3].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(t_setting["init_type"].iVal(), t_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    this->params[2].Init();
    this->params[3].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &u_updater = *setting["u_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    std::map<std::string, SettingV> &t_updater = *setting["t_updater"].mVal();

    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(), w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(u_updater["updater_type"].iVal(), u_updater, this->prnd_);
    this->params[2].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(), b_updater, this->prnd_);
    this->params[3].updater_ = 
        updater::CreateUpdater<xpu, 4>(t_updater["updater_type"].iVal(), t_updater, this->prnd_);
  }
  
  // bottom should be padded with only one zero on both sides
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "MaxRecurrentLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MaxRecurrentLayer:top size problem.");
    
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], d_mem);

    std::cout << "Rnn io shape:" << std::endl;
    std::cout << shape_in[0] << "x" << shape_in[1] << "x" << shape_in[2] << "x" << shape_in[3] << std::endl;
    std::cout << shape_out[0] << "x" << shape_out[1] << "x" << shape_out[2] << "x" << shape_out[3] << std::endl;

    top[0]->Resize(shape_out, true);
    pos.Resize(shape_out, true);
    cc.Resize(shape_out, true);
    cc_er.Resize(shape_out, true);
    x_t.Resize(shape_out, true);
    x_t_er.Resize(shape_out, true);
  }

  void checkNan(float *p, int l) {
    for (int i = 0; i < l; ++i) {
      assert(!std::isnan(p[i]));
    }
  }

  void checkNanParams() {
      Tensor2D w_data = this->params[0].data[0][0];
      Tensor2D u_data = this->params[1].data[0][0];
      Tensor2D t_data = this->params[3].data[0][0];
      Tensor2D w_diff = this->params[0].diff[0][0];
      Tensor2D u_diff = this->params[1].diff[0][0];
      Tensor2D t_diff = this->params[3].diff[0][0];
      checkNan(w_data.dptr_, w_data.size(0) * w_data.size(1));
      checkNan(u_data.dptr_, u_data.size(0) * u_data.size(1));
      checkNan(t_data.dptr_, t_data.size(0) * t_data.size(1));
      checkNan(w_diff.dptr_, w_diff.size(0) * w_diff.size(1));
      checkNan(u_diff.dptr_, u_diff.size(0) * u_diff.size(1));
      checkNan(t_diff.dptr_, t_diff.size(0) * t_diff.size(1));
  }

  virtual void ForwardOneStep(Tensor2D pre_h,
                              Tensor2D x,
                              Tensor2D cc,
                              Tensor2D x_t,
                              Tensor2D pos,
                              Tensor2D cur_h) {
      Tensor2D w_data = this->params[0].data_d2();
      Tensor2D u_data = this->params[1].data_d2();
      Tensor1D b_data = this->params[2].data_d1();
      Tensor2D t_data = this->params[3].data_d2();

      cc = dot(pre_h, u_data);
      // if (input_transform) {
      cc += dot(x, w_data);
      // } else {
      //   cc += x;
      // }
      if (!no_bias) {
        cc += repmat(b_data, 1);
      }

      x_t = dot(x, t_data);
      if (nonlinear_type == "sigmoid") {
        cc = mshadow::expr::F<op::sigmoid>(cc);   // sigmoid_grad
        x_t = mshadow::expr::F<op::sigmoid>(x_t); // sigmoid_grad
      } else if (nonlinear_type == "tanh") {         
        cc = mshadow::expr::F<op::tanh>(cc);      // tanh_grad
        x_t = mshadow::expr::F<op::tanh>(x_t);    // tanh_grad
      } else if (nonlinear_type == "rectifier") {
        cc = mshadow::expr::F<op::relu>(cc);      // relu_grad
        x_t = mshadow::expr::F<op::relu>(x_t);    // relu_grad
      } else {
        utils::Check(false, "MaxRecurrentLayer:nonlinear type error.");
      }

      // max_pooling rnn
      maxPooling(cc, x_t, pre_h, pos, cur_h);
  }
  void maxPooling(Tensor2D p, Tensor2D c1, Tensor2D c2, Tensor2D pos, Tensor2D h) {
    utils::Check(p.size(0) == 1 && c1.size(0) == 1 && c2.size(0) == 1 && \
                 pos.size(0) == 1 && h.size(0) == 1, "MaxRecurrent: pooling io size error");
    utils::Check(p.size(1) == c1.size(1) && p.size(1) == c2.size(1) && \
                 p.size(1) == pos.size(1) && p.size(1) == h.size(1),
                 "MaxRecurrent: pooling io size error");

    pos = 0;
    h = mshadow::expr::F<op::identity>(p);
    for (index_t col = 0; col < c1.size(1); ++col) {
        if (c1[0][col] > h[0][col]) {
          pos[0][col] = 1;
          h[0][col] = c1[0][col];
        }
    }
    for (index_t col = 0; col < c2.size(1); ++col) {
        if (c2[0][col] > h[0][col]) {
          pos[0][col] = 2;
          h[0][col] = c2[0][col];
        }
    }
  }
  void unMaxPooling(Tensor2D p, Tensor2D c1, Tensor2D c2, Tensor2D pos, Tensor2D h) {
    utils::Check(p.size(0) == 1 && c1.size(0) == 1 && c2.size(0) == 1 && \
                 pos.size(0) == 1 && h.size(0) == 1, "MaxRecurrent: pooling io size error");
    utils::Check(p.size(1) == c1.size(1) && p.size(1) == c2.size(1) && \
                 p.size(1) == pos.size(1) && p.size(1) == h.size(1),
                 "MaxRecurrent: pooling io size error");

    for (index_t col = 0; col < pos.size(1); ++col) {
      int row = pos[0][col];
      if (row == 0) {
        p[0][col] += h[0][col];
      } else if (row == 1) {
        c1[0][col] += h[0][col];
      } else if (row == 2) {
        c2[0][col] += h[0][col];
      } else {
        utils::Assert(false, "xxx");
      }
    }
  }

  void ForwardLeft2Right(Tensor2D in, Tensor2D cc, Tensor2D x_t, Tensor2D pos, Tensor2D out) {
      int begin = 0, end = in.size(0);
      Tensor2D pre_h;
      // not need any padding, begin h and c are set to 0
      for (index_t row_idx = begin; row_idx < end; ++row_idx) {
        if (row_idx == begin) {
          pre_h = begin_h;
        } else {
          pre_h = out.Slice(row_idx-1, row_idx);
        }
        ForwardOneStep(pre_h,
                       in.Slice(row_idx, row_idx+1),
                       cc.Slice(row_idx, row_idx+1),
                       x_t.Slice(row_idx, row_idx+1),
                       pos.Slice(row_idx, row_idx+1),
                       out.Slice(row_idx, row_idx+1));
      }
  }
  void ForwardRight2Left(Tensor2D in, Tensor2D cc, Tensor2D x_t, Tensor2D pos, Tensor2D out) {
      int begin = 0, end = in.size(0);
      Tensor2D pre_h;
      // not need any padding, begin h and c are set to 0
      for (int row_idx = end-1; row_idx >= begin; --row_idx) {
        if (row_idx == end-1) {
          pre_h = begin_h;
        } else {
          pre_h = out.Slice(row_idx+1, row_idx+2);
        }
        ForwardOneStep(pre_h,
                       in.Slice(row_idx, row_idx+1),
                       cc.Slice(row_idx, row_idx+1),
                       x_t.Slice(row_idx, row_idx+1),
                       pos.Slice(row_idx, row_idx+1),
                       out.Slice(row_idx, row_idx+1));
      }
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    // checkNanParams();
    Tensor4D bottom_data = bottom[0]->data;
    Tensor4D top_data = top[0]->data;
    top[0]->length = mshadow::expr::F<op::identity>(bottom[0]->length);

    pos = 0; cc = 0.f; x_t = 0.f; top_data = 0.f;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < bottom_data.size(1); ++seq_idx) {
        int len = bottom[0]->length[batch_idx][seq_idx];
        utils::Assert(len >= 0, "MaxRecurrentLayer: sequence length error.");
        if (!reverse) {
          ForwardLeft2Right(bottom_data[batch_idx][seq_idx].Slice(0,len),  
                            cc[batch_idx][seq_idx].Slice(0,len),
                            x_t[batch_idx][seq_idx].Slice(0,len),
                            pos[batch_idx][seq_idx].Slice(0,len),
                            top_data[batch_idx][seq_idx].Slice(0,len));
        } else {
          ForwardRight2Left(bottom_data[batch_idx][seq_idx].Slice(0,len), 
                            cc[batch_idx][seq_idx].Slice(0,len),
                            x_t[batch_idx][seq_idx].Slice(0,len),
                            pos[batch_idx][seq_idx].Slice(0,len),
                            top_data[batch_idx][seq_idx].Slice(0,len));
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
                 Tensor2D x_er,
                 Tensor2D cc,
                 Tensor2D cc_er,
                 Tensor2D x_t,
                 Tensor2D x_t_er,
                 Tensor2D pos) {
    Tensor2D w_er = this->params[0].diff_d2();
    Tensor2D u_er = this->params[1].diff_d2();
    Tensor1D b_er = this->params[2].diff_d1();
    Tensor2D t_er = this->params[3].diff_d2();

    Tensor2D w_data = this->params[0].data_d2();
    Tensor2D u_data = this->params[1].data_d2();
    Tensor2D t_data = this->params[3].data_d2();

    unMaxPooling(cc_er, x_t_er, pre_h_er, pos, cur_h_er);

    if (nonlinear_type == "sigmoid") {
      cc_er *= mshadow::expr::F<op::sigmoid_grad>(cc); // sigmoid_grad
      x_t_er *= mshadow::expr::F<op::sigmoid_grad>(x_t); // sigmoid_grad
    } else if (nonlinear_type == "tanh") {         
      cc_er *= mshadow::expr::F<op::tanh_grad>(cc);    // tanh_grad
      x_t_er *= mshadow::expr::F<op::tanh_grad>(x_t);    // tanh_grad
    } else if (nonlinear_type == "rectifier") {
      cc_er *= mshadow::expr::F<op::relu_grad>(cc);    // relu_grad
      x_t_er *= mshadow::expr::F<op::relu_grad>(x_t);    // relu_grad
    } else {
      utils::Check(false, "MaxRecurrentLayer:nonlinear type error.");
    }

    x_er += dot(x_t_er, t_data.T());
    t_er += dot(x.T(), x_t_er);

    pre_h_er += dot(cc_er, u_data.T());
    u_er += dot(pre_h.T(), cc_er);

    // if (input_transform) {
    x_er += dot(cc_er, w_data.T());
    w_er += dot(x.T(), cc_er); 
    // } else {
    //   x_er += cur_h_er;
    // }
    if (!no_bias) {
      b_er += sum_rows(cur_h_er);
    }
  }

  void BackpropForLeft2RightRnn(Tensor2D top_data, Tensor2D top_diff, 
                                Tensor2D bottom_data, Tensor2D bottom_diff,
                                Tensor2D cc, Tensor2D cc_er,
                                Tensor2D x_t, Tensor2D x_t_er,
                                Tensor2D pos) {
      int begin = 0, end = top_data.size(0);

      Tensor2D pre_h, pre_h_er;
      for (int row_idx = end-1; row_idx >= begin; --row_idx) {
        if (row_idx == begin) {
            pre_h = begin_h;
            pre_h_er = begin_h_er;
        } else {
            pre_h = top_data.Slice(row_idx-1, row_idx);
            pre_h_er = top_diff.Slice(row_idx-1, row_idx);
        }
        BpOneStep(top_diff.Slice(row_idx, row_idx+1), 
                  top_data.Slice(row_idx, row_idx+1),
                  pre_h,
                  bottom_data.Slice(row_idx, row_idx+1), 
                  pre_h_er,
                  bottom_diff.Slice(row_idx, row_idx+1),
                  cc.Slice(row_idx, row_idx+1),
                  cc_er.Slice(row_idx, row_idx+1),
                  x_t.Slice(row_idx, row_idx+1),
                  x_t_er.Slice(row_idx, row_idx+1),
                  pos.Slice(row_idx, row_idx+1));
      }
  }
  void BackpropForRight2LeftRnn(Tensor2D top_data, Tensor2D top_diff, 
                                Tensor2D bottom_data, Tensor2D bottom_diff,
                                Tensor2D cc, Tensor2D cc_er,
                                Tensor2D x_t, Tensor2D x_t_er,
                                Tensor2D pos) {
      int begin = 0, end = top_data.size(0);

      Tensor2D pre_h, pre_h_er;
      for (int row_idx = begin; row_idx < end; ++row_idx) {
        if (row_idx == end-1) {
            pre_h = begin_h;
            pre_h_er = begin_h_er;
        } else {
            pre_h = top_data.Slice(row_idx+1, row_idx+2);
            pre_h_er = top_diff.Slice(row_idx+1, row_idx+2);
        }
        BpOneStep(top_diff.Slice(row_idx, row_idx+1), 
                  top_data.Slice(row_idx, row_idx+1),
                  pre_h,
                  bottom_data.Slice(row_idx, row_idx+1), 
                  pre_h_er,
                  bottom_diff.Slice(row_idx, row_idx+1),
                  cc.Slice(row_idx, row_idx+1),
                  cc_er.Slice(row_idx, row_idx+1),
                  x_t.Slice(row_idx, row_idx+1),
                  x_t_er.Slice(row_idx, row_idx+1),
                  pos.Slice(row_idx, row_idx+1));
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
        
    begin_h_er = 0.f; cc_er = 0.f; x_t_er = 0.f;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < bottom_data.size(1); ++seq_idx) {
        int len = bottom[0]->length[batch_idx][seq_idx];
        utils::Assert(len >= 0, "MaxRecurrentLayer: sequence length error.");
        if (!reverse) {
            BackpropForLeft2RightRnn(top_data[batch_idx][seq_idx].Slice(0,len), 
                                     top_diff[batch_idx][seq_idx].Slice(0,len), 
                                     bottom_data[batch_idx][seq_idx].Slice(0,len), 
                                     bottom_diff[batch_idx][seq_idx].Slice(0,len),
                                     cc[batch_idx][seq_idx].Slice(0,len),
                                     cc_er[batch_idx][seq_idx].Slice(0,len),
                                     x_t[batch_idx][seq_idx].Slice(0,len),
                                     x_t_er[batch_idx][seq_idx].Slice(0,len),
                                     pos[batch_idx][seq_idx].Slice(0,len));
        } else {
            BackpropForRight2LeftRnn(top_data[batch_idx][seq_idx].Slice(0,len), 
                                     top_diff[batch_idx][seq_idx].Slice(0,len), 
                                     bottom_data[batch_idx][seq_idx].Slice(0,len), 
                                     bottom_diff[batch_idx][seq_idx].Slice(0,len),
                                     cc[batch_idx][seq_idx].Slice(0,len),
                                     cc_er[batch_idx][seq_idx].Slice(0,len),
                                     x_t[batch_idx][seq_idx].Slice(0,len),
                                     x_t_er[batch_idx][seq_idx].Slice(0,len),
                                     pos[batch_idx][seq_idx].Slice(0,len));
        }
      }
    }
  }

 protected:
  int d_mem, d_input;
  bool no_bias, reverse;
  mshadow::TensorContainer<xpu, 2> begin_h, begin_h_er;
  mshadow::TensorContainer<xpu, 4> pos, cc, x_t, cc_er, x_t_er;
  std::string nonlinear_type;
};
}  // namespace layer
}  // namespace textnet
#endif
