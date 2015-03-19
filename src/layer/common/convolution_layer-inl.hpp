#ifndef TEXTNET_LAYER_CONVOLUTION_LAYER_INL_HPP_
#define TEXTNET_LAYER_CONVOLUTION_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class ConvolutionLayer : public Layer<xpu> {
 public:
  ConvolutionLayer(LayerType type) { this->layer_type = type; }
  virtual ~ConvolutionLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ConvolutionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ConvolutionLayer:top size problem.");
                  
    kernel_x = setting["kernel_x"].i_val;
    kernel_y = setting["kernel_y"].i_val;
    pad_x = setting["pad_x"].i_val;
    pad_y = setting["pad_y"].i_val;
    stride = setting["stride"].i_val;
    channel_in = bottom[0]->data.size(1);
    channel_out = setting["channel_out"].i_val;
    no_bias = setting["no_bias"].b_val;
    
    this->params.resize(2);
    this->params[0].Resize(channel_out, channel_in * kernel_x * kernel_y, 1, 1);
    this->params[1].Resize(channel_out, 1, 1, 1);
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].m_val;
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].m_val;
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].i_val,
          w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].i_val, 
          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].m_val;
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].m_val;
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].i_val,
          w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].i_val,
          b_updater, this->prnd_);
          
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ConvolutionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ConvolutionLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], channel_out, 
                   (shape_in[2] + pad_y * 2 - kernel_y) / stride + 1,
                   (shape_in[3] + pad_x * 2 - kernel_x) / stride + 1);

    std::cout << shape_in[0] << "x" << shape_in[1] << "x" << shape_in[2] << "x" << shape_in[3] << std::endl;
    std::cout << shape_out[0] << "x" << shape_out[1] << "x" << shape_out[2] << "x" << shape_out[3] << std::endl;

    top[0]->Resize(shape_out);
    temp_col_.Resize(mshadow::Shape2(channel_in*kernel_x*kernel_y, shape_out[2]*shape_out[3]));
    // Share the memory
    temp_dif_ = temp_col_;

    temp_data_.Resize(mshadow::Shape2(channel_out, shape_out[2]*shape_out[3]));
    
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> weight_data = this->params[0].data_d2();
    mshadow::Tensor<xpu, 1> bias_data = this->params[1].data_d1();
    const index_t nbatch = bottom_data.size(0);
    for (index_t i = 0; i < nbatch; ++i) {
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
      bias_diff += sumall_except_dim<1>(top_diff);
    }
    
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
  
  virtual void SaveModel(utils::IStream &fo) const {
    
  }
  virtual void LoadModel(utils::IStream &fi) {
    
  }

 protected:
  int kernel_x;
  int kernel_y;
  int pad_x;
  int pad_y;
  int stride;
  int channel_in;
  int channel_out;
  bool no_bias;
  mshadow::TensorContainer<xpu, 2> temp_col_;
  mshadow::TensorContainer<xpu, 2> temp_dif_;
  mshadow::TensorContainer<xpu, 2> temp_data_;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CONVOLUTION_LAYER_INL_HPP_
