#ifndef TEXTNET_LAYER_LSTM_LAYER_INL_HPP_
#define TEXTNET_LAYER_LSTM_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class LstmLayer : public Layer<xpu> {
 public:
  LstmLayer(LayerType type) { this->layer_type = type; }
  virtual ~LstmLayer(void) {}
  
  // to do
  // virtual int BottomNodeNum() { return 1; }
  // virtual int TopNodeNum() { return 1; }
  // virtual int ParamNodeNum() { return 2; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    // utils::Check(bottom.size() == BottomNodeNum(),
    //               "LstmLayer:bottom size problem."); 
    // utils::Check(top.size() == TopNodeNum(),
    //               "LstmLayer:top size problem.");
                  
    d_mem = setting["d_mem"].i_val;
    d_input = setting["d_input"].i_val;
    no_bias = setting["no_bias"].b_val;
    
    this->params.resize(3);
    this->params[0].Resize(1, 1, d_input, 4*d_mem, true); // w
    this->params[1].Resize(1, 1, d_mem,   4*d_mem, true); // u
    this->params[2].Resize(1, 1, 1,       4*d_mem, true); // b
    
    std::map<std::string, SettingV> &w_setting = *setting["w"].m_val;
    std::map<std::string, SettingV> &u_setting = *setting["u"].m_val;
    std::map<std::string, SettingV> &b_setting = *setting["b"].m_val;
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

    std::cout << shape_in[0] << "x" << shape_in[1] << "x" << shape_in[2] << "x" << shape_in[3] << std::endl;
    std::cout << shape_out[0] << "x" << shape_out[1] << "x" << shape_out[2] << "x" << shape_out[3] << std::endl;

    top[0]->Resize(shape_out);
    h.Resize(shape_out); // h is top[0], is the hidden rep, is the output
    c.Resize(shape_out);
    g.Resize(shape_out);
    h_er.Resize(shape_out);
    c_er.Resize(shape_out);
    g_er.Resize(shape_out);
     
    // temp_col_.Resize(mshadow::Shape2(channel_in*kernel_x*kernel_y, shape_out[2]*shape_out[3]));
    // Share the memory
    // temp_dif_ = temp_col_;
    // temp_data_.Resize(mshadow::Shape2(channel_out, shape_out[2]*shape_out[3]));
  }

  void locateBeginEnd(mshadow::Tensor<xpu, 2> seq, 
                      index_t &begin, index_t &end) { // input a 2D tensor, out put a sub 2d tensor, with 0 padding
      assert(false); // to do
      return;
  }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;

  virtual void ForwardOneStep(Tensor1D pre_c, 
                              Tensor1D pre_h,
                              Tensor1D x,
                              Tensor1D cur_g,
                              Tensor1D cur_c,
                              Tensor1D cur_h) {
      assert(false); // check again
      Tensor2D w_data = this->params[0].data_d2();
      Tensor2D u_data = this->params[1].data_d2();
      Tensor1D b_data = this->params[2].data_d1();

      Tensor1D i, f, o, cc;
      cur_g = dot(x, w_data) + dot(pre_h, u_data) + b;
      splitGate(i, f, o, cc);
      i = F<ForwardOp>(i);  // logi
      f = F<ForwardOp>(f);  // logi
      o = F<ForwardOp>(o);  // logi
      cc= F<ForwardOp>(cc); // tanh 

      cur_c = f * cur_c + i * cc;
      cur_h = o * F<ForwardOp>(cur_c); // tanh
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D bottom_data = bottom[0]->data;
    Tensor4D top_data = top[0]->data;
    const index_t nbatch = bottom_data.size(0); 
    for (index_t i = 0; i < nbatch; ++i) {
        mshadow::Tensor<xpu, 2> seq = bottom_data[i].data_d2();
        index_t begin = 0, end = 0;
        locateBeginEnd(seq, begin, end);
        assert(begin > 0 && begin < end); 
        for (index_t row_idx = begin; row_idx < end; ++row_idx) {
            // shoude use slice？
            ForwardOneStep(c[i][0][row_idx-1], h[i][0][row_idx-1], seq[row_idx]
                           g[i][0][row_idx], c[i][0][row_idx], h[i][0][row_idx]);
        }
        
    }
    top[0]->data = h;
    //if (no_bias == 0) {
      // add bias, broadcast bias to dim 1: channel
      // top_data += broadcast<1>(bias_data, top_data.shape_);
    // }
  }

  void SplitGate(Tensor1D gate, Tensor1D i, Tensor1D f, Tensor1D o) {
      assert(false);
  }

  void BpOneStep(Tensor1D cur_h_er,
                 Tensor1D pre_c,
                 Tensor1D pre_h,
                 Tensor1D x,
                 Tensor1D cur_g,
                 Tensor1D cur_c,
                 Tensor1D cur_h,
                 Tensor1D cur_c_er,
                 Tensor1D cur_g_er,
                 Tensor1D x_er,
                 Tensor1D pre_h_er,
                 Tensor1D pre_c_er,
                 Tensor1D w_er,
                 Tensor1D u_er,
                 Tensor1D b_er) {
    Tensor1D i, f, o, cc, tanhc, i_er, f_er, o_er, cc_er, tanhc_er,;
    splitGate(cur_g, i, f, o, cc);
    splitGate(cur_g_er, i_er, f_er, o_er, cc_er);

    tanhc = F<tanh>(cur_c);
    o_er = F<BackOp>(o) * (cur_h_er * tanhc); // logi
    cur_c_er += F<BackOp>(tanh_c) * (cur_h_er * o);

    i_er = F<BackOp>(i) * (cur_c_er * cc);    // logi
    cc_er = F<BackOp>(cc) * (cur_c_er * i);   // tanh
    pre_c_er = cur_c_er * f;
    f_er = F<BackOp>(f) * (cur_c_er * pre_c); // logi

    pre_h_er += cur_g_er * u;
    x_er = cur_g_er * w;

    // grad
    b_er += cur_g_er;
    w_er += dot(x.T(), cur_g_er); 
    u_er += dot(pre_h.T(), cur_g_er);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> top_diff = top[0]->diff_d3();
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> weight_data = this->params[0].data_d2();
    mshadow::Tensor<xpu, 2> weight_diff = this->params[0].diff_d2();
    mshadow::Tensor<xpu, 1> bias_diff = this->params[1].diff_d1();
    const index_t nbatch = bottom_data.size(0);
        
    if (!no_bias && this->prop_grad[1]) {
      bias_diff = sumall_except_dim<1>(top_diff);
    }
    
	weight_diff = 0.0f;

    for (index_t i = 0; i < nbatch; ++i) {
      Tensor2D seq = bottom_data[i].data_d2();
      index_t begin = 0, end = 0;
      locateBeginEnd(seq, begin, end);
      assert(begin > 0 && begin < end); 
      for (index_t row_idx = begin; row_idx < end; ++row_idx) {
          // shoude use slice？
          BpOneStep(h_er[i][0][row_idx], 
                    c[i][0][row_idx-1], 
                    h[i][0][row_idx-1],
                    seq[row_idx], 
                    g[i][0][row_idx], 
                    c[i][0][row_idx], 
                    h[i][0][row_idx],
                    c_er[i][0][row_idx],
                    g_er[i][0][row_idx], 
                    x_er[i][0][row_idx], 
                    h_er[i][0][row_idx-1], 
                    c_er[i][0][row_idx-1],
                    w_er, u_er, b_er);
      }
    }
    bottom[0]->data = x_er;
  }

 protected:
  int d_mem, d_input;
  bool no_bias; // to do 
  mshadow::TensorContainer<xpu, 4> h, c, g, h_er, c_er, g_er, x_er;

  // mshadow::TensorContainer<xpu, 2> temp_col_;
  // mshadow::TensorContainer<xpu, 2> temp_dif_;
  // mshadow::TensorContainer<xpu, 2> temp_data_;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LSTM_LAYER_INL_HPP_
