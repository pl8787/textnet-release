#ifndef TEXTNET_LAYER_CONVOLUTION_LAYER_INL_HPP_
#define TEXTNET_LAYER_CONVOLUTION_LAYER_INL_HPP_

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

  virtual void ForwardOneStep(mshadow::Tensor<xpu, 2> 
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> w_data = this->params[0].data_d2();
    mshadow::Tensor<xpu, 2> u_data = this->params[1].data_d2();
    mshadow::Tensor<xpu, 1> b_data = this->params[2].data_d1();
    const index_t nbatch = bottom_data.size(0); 
    for (index_t i = 0; i < nbatch; ++i) {
        mshadow::Tensor<xpu, 2> seq = bottom_data[i].data_d2();
        index_t begin = 0, end = 0;
        locateBeginEnd(seq, begin, end);
        assert(begin > 0 && begin < end); 
        for (index_t row_idx = begin; row_idx < end; ++row_idx) {
            
        } 
        
        // const index_t  = bottom_data.size(0); 

      if (pad_x == 0 && pad_y == 0) {
        temp_col_ = unpack_patch2col(bottom_data[i], kernel_y, kernel_x, stride);
      } else {
        temp_col_ = unpack_patch2col(pad(bottom_data[i], pad_y, pad_x),
                                     kernel_y, kernel_x, stride);
      }
      temp_data_ = dot(weight_data, temp_col_);
      top_data.Slice(i,i+1) = reshape(temp_data_, top_data.Slice(i,i+1).shape_);
    }
    if (no_bias == 0) {
      // add bias, broadcast bias to dim 1: channel
      top_data += broadcast<1>(bias_data, top_data.shape_);
    }
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

    for (int i = 0; i < nbatch; ++i) {
      if (pad_x == 0 && pad_y == 0) {
        temp_col_ = unpack_patch2col(bottom_data[i], kernel_y, kernel_x, stride);
      }else{
        temp_col_ = unpack_patch2col(pad(bottom_data[i], pad_y, pad_x),
                                     kernel_y, kernel_x, stride);
      }
      
      if (this->prop_grad[0]) {
        weight_diff += dot(top_diff[i], temp_col_.T());
      }

      if (this->prop_error[0]) {
        temp_dif_ = dot(weight_data.T(), top_diff[i]);
        mshadow::Tensor<xpu, 3> one_diff = bottom_diff[i];
        if (pad_x == 0 && pad_y == 0) {
          one_diff = pack_col2patch(temp_dif_, one_diff.shape_, 
              kernel_y, kernel_x, stride);
        } else {
          mshadow::Shape<3> pshape = one_diff.shape_;
          pshape[1] += 2*pad_y; 
          pshape[2] += 2*pad_x;
          one_diff = crop(pack_col2patch(temp_dif_, pshape, 
              kernel_y, kernel_x, stride), one_diff[0].shape_);
        }
      }
      
    }
      
    
  }

 protected:
  int d_mem, d_input;
  // mshadow::TensorContainer<xpu, 2> temp_col_;
  mshadow::TensorContainer<xpu, 4> w, u, b;
  mshadow::TensorContainer<xpu, 4> h, c, g, h_er, c_er, g_er;

  int kernel_x;
  int kernel_y;
  int pad_x;
  int pad_y;
  int stride;
  int channel_in;
  int channel_out;
  bool no_bias;
  // mshadow::TensorContainer<xpu, 2> temp_col_;
  // mshadow::TensorContainer<xpu, 2> temp_dif_;
  // mshadow::TensorContainer<xpu, 2> temp_data_;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CONVOLUTION_LAYER_INL_HPP_
