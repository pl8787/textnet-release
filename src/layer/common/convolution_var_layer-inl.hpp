#ifndef TEXTNET_LAYER_CONVOLUTION_VAR_LAYER_INL_HPP_
#define TEXTNET_LAYER_CONVOLUTION_VAR_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class ConvolutionVarLayer : public Layer<xpu> {
 public:
  ConvolutionVarLayer(LayerType type) { this->layer_type = type; }
  virtual ~ConvolutionVarLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["pad_x"] = SettingV(0);
    this->defaults["pad_y"] = SettingV(0);
    this->defaults["stride_x"] = SettingV(1);
    this->defaults["stride_y"] = SettingV(1);
    this->defaults["no_bias"] = SettingV(false);
	  this->defaults["dim"] = SettingV(2);
    
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
                  "ConvolutionVarLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ConvolutionVarLayer: top size problem.");
                  
    this->param_file = setting["param_file"].sVal();
    kernel_x = setting["kernel_x"].iVal();
    kernel_y = setting["kernel_y"].iVal();
    pad_x = setting["pad_x"].iVal();
    pad_y = setting["pad_y"].iVal();
    stride_x = setting["stride_x"].iVal();
    stride_y = setting["stride_y"].iVal();
    channel_in = bottom[0]->data.size(1);
    channel_out = setting["channel_out"].iVal();
    no_bias = setting["no_bias"].bVal();
	dim = setting["dim"].iVal();
    
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
    if (!this->param_file.empty()) {
      this->LoadParams();
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ConvolutionVarLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ConvolutionVarLayer: top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out;
	if (dim == 1) {
		shape_out = mshadow::Shape4(shape_in[0], channel_out,
				(shape_in[2] + pad_y * 2 - kernel_y) / stride_y + 1,
				1);
	} else {
		shape_out = mshadow::Shape4(shape_in[0], channel_out, 
                (shape_in[2] + pad_y * 2 - kernel_y) / stride_y + 1,
                (shape_in[3] + pad_x * 2 - kernel_x) / stride_x + 1);
	}
	mshadow::Shape<2> shape_len;
	shape_len = bottom[0]->length.shape_;
	top[0]->Resize(shape_out, shape_len);

    temp_col_.Resize(mshadow::Shape2(shape_out[2]*shape_out[3], channel_in*kernel_x*kernel_y));
    // Share the memory
    temp_dif_ = temp_col_;

    temp_data_.Resize(mshadow::Shape2(shape_out[2]*shape_out[3], channel_out));
    
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
                   (shape_in[2] + pad_y * 2 - kernel_y) / stride_y + 1,
                   (shape_in[3] + pad_x * 2 - kernel_x) / stride_x + 1);
    if (! (shape_out == top[0]->data.shape_)) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }

void PrintTensor(const char * name, mshadow::Tensor<xpu, 1> x) {
	mshadow::Shape<1> s = x.shape_;
    cout << name << " shape " << s[0] << endl;
    for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
      cout << x[d1] << " ";
    }
    cout << endl;
}

void PrintTensor(const char * name, mshadow::Tensor<xpu, 2> x) {
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

void PrintTensor(const char * name, mshadow::Tensor<xpu, 3> x) {
    mshadow::Shape<3> s = x.shape_;
    cout << name << " shape " << s[0] << "x" << s[1] << "x" << s[2] << endl;
    for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
        for (unsigned int d2 = 0; d2 < s[1]; ++d2) {
            for (unsigned int d3 = 0; d3 < s[2]; ++d3) {
                    cout << x[d1][d2][d3] << " ";
            }
            cout << ";";
        }
        cout << endl;
    }
}

void PrintTensor(const char * name, mshadow::Tensor<xpu, 4> x) {
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
}

  void unpack_patch2col_var(mshadow::Tensor<xpu, 2> data_col, mshadow::Tensor<xpu, 3> data_im, 
		  int len_y, int len_x, int kernel_y, int kernel_x, 
		  int stride_y, int stride_x, int pad_y, int pad_x) {
	const int channels = data_im.size(0);
    const int y_col = (len_y + 2 * pad_y - kernel_y) / stride_y + 1;
    const int x_col = (len_x + 2 * pad_x - kernel_x) / stride_x + 1;
    const int channels_col = channels * kernel_y * kernel_x;
    for (int c_col = 0; c_col < channels_col; ++c_col) {
      int x_offset = c_col % kernel_x;
      int y_offset = (c_col / kernel_x) % kernel_y;
      int c_im = c_col / kernel_y / kernel_x;
      for (int yy_col = 0; yy_col < y_col; ++yy_col) {
        for (int xx_col = 0; xx_col < x_col; ++xx_col) {
          int y_im = yy_col * stride_y - pad_y + y_offset;
          int x_im = xx_col * stride_x - pad_x + x_offset;
          data_col[yy_col * x_col + xx_col][c_col] =
              (y_im >= 0 && x_im >= 0 && y_im < len_y && x_im < len_x) ?
              data_im[c_im][y_im][x_im] : 0;
        }
      }
    }
  }

  void pack_col2patch_var(mshadow::Tensor<xpu, 3> data_im, mshadow::Tensor<xpu, 2> data_col, 
		  int len_y, int len_x, int kernel_y, int kernel_x, 
		  int stride_y, int stride_x, int pad_y, int pad_x) {
	const int channels = data_im.size(0);
    const int y_col = (len_y + 2 * pad_y - kernel_y) / stride_y + 1;
    const int x_col = (len_x + 2 * pad_x - kernel_x) / stride_x + 1;
    const int channels_col = channels * kernel_y * kernel_x;
    for (int c_col = 0; c_col < channels_col; ++c_col) {
      int x_offset = c_col % kernel_x;
      int y_offset = (c_col / kernel_x) % kernel_y;
      int c_im = c_col / kernel_y / kernel_x;
      for (int yy_col = 0; yy_col < y_col; ++yy_col) {
        for (int xx_col = 0; xx_col < x_col; ++xx_col) {
          int y_im = yy_col * stride_y - pad_y + y_offset;
          int x_im = xx_col * stride_x - pad_x + x_offset;
          if (y_im >= 0 && y_im < len_y && x_im >= 0 && x_im < len_x)
            data_im[c_im][y_im][x_im] +=
                data_col[yy_col * x_col + xx_col][c_col];
        }
      }
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
	top_data = 0;
	int top_len_x = 0, top_len_y = 0, bottom_len_x = 0, bottom_len_y = 0;
    for (index_t i = 0; i < nbatch; ++i) {
      if (dim == 1) {
          top_len[i][0] = (bottom_len[i][0] + pad_y * 2 - kernel_y) / stride_y + 1; // all input channels shoud have the same length
		  utils::Check(top_len[i][0] > 0, "ConvolutionVarLayer: top_len must positive. i=%d, bottom_len=%f, top_len=%f", i, bottom_len[i][0], top_len[i][0]);
		  utils::Check(pad_x == 0, "ConvolutionVarLayer: dim=1 pad_x!=0.");
		  top_len_x = 1;
		  top_len_y = top_len[i][0];
		  bottom_len_x = kernel_x;
		  bottom_len_y = bottom_len[i][0];
	  } else {
		  top_len[i][0] = (bottom_len[i][0] + pad_y * 2 - kernel_y) / stride_y + 1;
		  top_len[i][1] = (bottom_len[i][1] + pad_x * 2 - kernel_x) / stride_x + 1;
		  utils::Check(top_len[i][0] > 0 && top_len[i][1] > 0, "ConvolutionVarLayer: top_len must positive. i=%d, bottom_len=(%f,%f), top_len=(%f,%f)",
				  i, bottom_len[i][0], bottom_len[i][1], top_len[i][0], top_len[i][1]);
		  top_len_y = top_len[i][0];
		  top_len_x = top_len[i][1];
		  bottom_len_y = bottom_len[i][0];
		  bottom_len_x = bottom_len[i][1];
	  }
      unpack_patch2col_var(temp_col_, bottom_data[i], bottom_len_y, bottom_len_x,
					   kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x);
	  //PrintTensor("temp_col_", temp_col_);
	  temp_data_.Resize(mshadow::Shape2(top_len_y * top_len_x, channel_out));
      temp_data_ = dot(temp_col_.Slice(0, top_len_y * top_len_x), weight_data.T());
	  //PrintTensor("temp_data_", temp_data_);
	  for (index_t idx = 0; idx < temp_data_.size(0); ++idx) {
		for (index_t ch = 0; ch < temp_data_.size(1); ++ch) {
		  int row = idx / top_len_x;
		  int col = idx % top_len_x;
		  if (no_bias) {
			top_data[i][ch][row][col] = temp_data_[idx][ch];
		  } else {
			top_data[i][ch][row][col] = temp_data_[idx][ch] + bias_data[ch];
		  }
		}
	  }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> weight_data = this->params[0].data_d2();
    mshadow::Tensor<xpu, 2> weight_diff = this->params[0].diff_d2();
    mshadow::Tensor<xpu, 1> bias_diff = this->params[1].diff_d1();
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;
    const index_t nbatch = bottom_data.size(0);
	int top_len_x = 0, top_len_y = 0, bottom_len_x = 0, bottom_len_y = 0;
        
    for (int i = 0; i < nbatch; ++i) {
      if (dim == 1) {
		  top_len_x = 1;
		  top_len_y = top_len[i][0];
		  bottom_len_x = kernel_x;
		  bottom_len_y = bottom_len[i][0];
	  } else {
		  top_len_y = top_len[i][0];
		  top_len_x = top_len[i][1];
		  bottom_len_y = bottom_len[i][0];
		  bottom_len_x = bottom_len[i][1];
	  }

	  temp_data_.Resize(mshadow::Shape2(top_len_y * top_len_x, channel_out));
	  
	  for (index_t idx = 0; idx < temp_data_.size(0); ++idx) {
		for (index_t ch = 0; ch < temp_data_.size(1); ++ch) {
		  int row = idx / top_len_x;
		  int col = idx % top_len_x;
		  temp_data_[idx][ch] = top_diff[i][ch][row][col];
		}
	  }

	  //PrintTensor("temp_data_", temp_data_);

      unpack_patch2col_var(temp_col_, bottom_data[i], bottom_len_y, bottom_len_x, 
					   kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x);
      
	  //PrintTensor("temp_col_", temp_col_);

      if (this->prop_grad[0]) {
        weight_diff += dot(temp_data_.T(), temp_col_.Slice(0, top_len_y * top_len_x));
      }

	  //PrintTensor("weight_diff", weight_diff);

	  if (!no_bias && this->prop_grad[1]) {
		bias_diff += sumall_except_dim<1>(temp_data_);
	  }

      if (this->prop_error[0]) {
		temp_dif_.Resize(mshadow::Shape2(top_len_y * top_len_x, channel_in * kernel_x * kernel_y));
        temp_dif_ = dot(temp_data_, weight_data);
        pack_col2patch_var(bottom_diff[i], temp_dif_, bottom_len_y, bottom_len_x,
              kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x);
      }
      
    }
  }

 protected:
  int kernel_x;
  int kernel_y;
  int pad_x;
  int pad_y;
  int stride_x;
  int stride_y;
  int channel_in;
  int channel_out;
  int dim;
  bool no_bias;
  mshadow::TensorContainer<xpu, 2> temp_col_;
  mshadow::TensorContainer<xpu, 2> temp_dif_;
  mshadow::TensorContainer<xpu, 2> temp_data_;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CONVOLUTION_VAR_LAYER_INL_HPP_
