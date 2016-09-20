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
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["pad_x"] = SettingV(0);
    this->defaults["pad_y"] = SettingV(0);
    this->defaults["stride"] = SettingV(1);
    this->defaults["no_bias"] = SettingV(false);
    this->defaults["d1_var_len"] = SettingV(false);
	this->defaults["ignore_len"] = SettingV(false);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["kernel_x"] = SettingV();
    this->defaults["kernel_y"] = SettingV();
    this->defaults["channel_out"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ConvolutionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ConvolutionLayer:top size problem.");
                  
    kernel_x = setting["kernel_x"].iVal();
    kernel_y = setting["kernel_y"].iVal();
    pad_x = setting["pad_x"].iVal();
    pad_y = setting["pad_y"].iVal();
    stride = setting["stride"].iVal();
    channel_in = bottom[0]->data.size(1);
    channel_out = setting["channel_out"].iVal();
    no_bias = setting["no_bias"].bVal();
    d1_var_len = setting["d1_var_len"].bVal();
	ignore_len = setting["ignore_len"].bVal();
    
    this->params.resize(2);
    this->params[0].Resize(channel_out, channel_in * kernel_x * kernel_y, 1, 1, true);
    this->params[1].Resize(channel_out, 1, 1, 1, true);
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
          w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), 
          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
          w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
          b_updater, this->prnd_);
          
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ConvolutionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ConvolutionLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], channel_out, 
                   (shape_in[2] + pad_y * 2 - kernel_y) / stride + 1,
                   (shape_in[3] + pad_x * 2 - kernel_x) / stride + 1);

    // std::cout << shape_in[0] << "x" << shape_in[1] << "x" << shape_in[2] << "x" << shape_in[3] << std::endl;
    // std::cout << shape_out[0] << "x" << shape_out[1] << "x" << shape_out[2] << "x" << shape_out[3] << std::endl;

	mshadow::Shape<2> shape_len;
	if (d1_var_len) {
		shape_len[0] = shape_in[0];
		shape_len[1] = 1;
	} else {
		shape_len = bottom[0]->length.shape_;
	}
	top[0]->Resize(shape_out, shape_len);

    temp_col_.Resize(mshadow::Shape2(channel_in*kernel_x*kernel_y, shape_out[2]*shape_out[3]));
    // Share the memory
    temp_dif_ = temp_col_;

    temp_data_.Resize(mshadow::Shape2(channel_out, shape_out[2]*shape_out[3]));
    
    if (show_info) {
      bottom[0]->PrintShape("bottom0");
      top[0]->PrintShape("top0");
    }
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], channel_out, 
                   (shape_in[2] + pad_y * 2 - kernel_y) / stride + 1,
                   (shape_in[3] + pad_x * 2 - kernel_x) / stride + 1);
    if (! (shape_out == top[0]->data.shape_)) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> weight_data = this->params[0].data_d2();
    mshadow::Tensor<xpu, 1> bias_data = this->params[1].data_d1();
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;
    const index_t nbatch = bottom_data.size(0);
    if (d1_var_len) { // variable length, all input channels in one example should have the same length
      utils::Check(bottom_len.dptr_[0] >= 0, 
                   "ConvolutionLayer: length error.");
      utils::Check(pad_x == 0 && kernel_x == bottom_data.size(3),
                   "ConvolutionLayer: varibale length convolution only supports 1D, kernel_x size error.");
      utils::Check(pad_y < kernel_y,
                   "ConvolutionLayer: pad_y is too much, will hurt the computation of length.");
    }
    for (index_t i = 0; i < nbatch; ++i) {
      if (d1_var_len) {
          top_len[i][0] = (bottom_len[i][0] + pad_y * 2 - kernel_y)/stride + 1; // all input channels shoud have the same length
		  
		  if (ignore_len && top_len[i][0] <= 0) {
		     top_len[i][0] = 1;
		  }

		  utils::Check(top_len[i][0] > 0, "top_len must positive.");
      } else {
		  top_len[i][0] = (bottom_len[i][0] + pad_x * 2 - kernel_x) / stride + 1;
		  top_len[i][1] = (bottom_len[i][1] + pad_y * 2 - kernel_y) / stride + 1;

		  if (ignore_len && top_len[i][0] <= 0) {
		       top_len[i][0] = 1;
		  }
		  if (ignore_len && top_len[i][1] <= 0) {
		       top_len[i][1] = 1;
		  }

		  utils::Check(top_len[i][0] > 0 && top_len[i][1] > 0, "top_len must positive.");
	  }
      if (pad_x == 0 && pad_y == 0) {
        temp_col_ = unpack_patch2col(bottom_data[i], kernel_y, kernel_x, stride);
      } else {
        temp_col_ = unpack_patch2col(pad(bottom_data[i], pad_y, pad_x),
                                     kernel_y, kernel_x, stride);
      }
      temp_data_ = dot(weight_data, temp_col_);
      top_data.Slice(i,i+1) = reshape(temp_data_, top_data.Slice(i,i+1).shape_);
    }
    if (!no_bias) {
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
          one_diff += pack_col2patch(temp_dif_, one_diff.shape_, 
              kernel_y, kernel_x, stride);
        } else {
          mshadow::Shape<3> pshape = one_diff.shape_;
          pshape[1] += 2*pad_y; 
          pshape[2] += 2*pad_x;
          one_diff += crop(pack_col2patch(temp_dif_, pshape, 
              kernel_y, kernel_x, stride), one_diff[0].shape_);
        }
      }
      
    }
  }

 protected:
  int kernel_x;
  int kernel_y;
  int pad_x;
  int pad_y;
  int stride;
  int channel_in;
  int channel_out;
  bool no_bias, d1_var_len, ignore_len;
  mshadow::TensorContainer<xpu, 2> temp_col_;
  mshadow::TensorContainer<xpu, 2> temp_dif_;
  mshadow::TensorContainer<xpu, 2> temp_data_;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CONVOLUTION_LAYER_INL_HPP_
