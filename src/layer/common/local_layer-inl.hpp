#ifndef TEXTNET_LAYER_LOCAL_LAYER_INL_HPP_
#define TEXTNET_LAYER_LOCAL_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class LocalLayer : public Layer<xpu> {
 public:
  LocalLayer(LayerType type) { this->layer_type = type; }
  virtual ~LocalLayer(void) {}
  
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
                  "LocalLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LocalLayer:top size problem.");
                  
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
    
    shape_in = bottom[0]->data.shape_;
	if (dim == 1) {
		shape_out = mshadow::Shape4(shape_in[0], channel_out,
				(shape_in[2] + pad_y * 2 - kernel_y) / stride_y + 1,
				1);
	} else {
		shape_out = mshadow::Shape4(shape_in[0], channel_out, 
                (shape_in[2] + pad_y * 2 - kernel_y) / stride_y + 1,
                (shape_in[3] + pad_x * 2 - kernel_x) / stride_x + 1);
	}

	// Set Length
	top_len_x = 0, top_len_y = 0, bottom_len_x = 0, bottom_len_y = 0;
    if (dim == 1) {
	    bottom_len_x = kernel_x;
	    bottom_len_y = shape_in[2];
	    top_len_x = 1;
	    top_len_y = shape_out[2];
	    utils::Check(top_len_y > 0, "top_len must positive.");
	    utils::Check(pad_x == 0, "dim=1 pad_x!=0.");
	} else {
	    bottom_len_x = shape_in[2];
	    bottom_len_y = shape_in[3];
	    top_len_x = shape_out[2];
	    top_len_y = shape_out[3];
	    utils::Check(top_len_x > 0 && top_len_y > 0, "top_len must positive.");
	}

    this->params.resize(2);
    this->params[0].Resize(channel_out, shape_out[2] * shape_out[3], channel_in * kernel_x * kernel_y, 1, true);
    this->params[1].Resize(channel_out, shape_out[2] * shape_out[3], 1, 1, true);
    
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
                  "LocalLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LocalLayer:top size problem.");
    
	mshadow::Shape<2> shape_len;
	shape_len = bottom[0]->length.shape_;
	top[0]->Resize(shape_out, shape_len);

    temp_col_.Resize(mshadow::Shape2(shape_out[2] * shape_out[3], channel_in * kernel_x * kernel_y));
    // Share the memory
    temp_dif_ = temp_col_;

    temp_data_.Resize(mshadow::Shape2(shape_out[2] * shape_out[3], channel_in * kernel_x * kernel_y));
	temp_diff_.Resize(mshadow::Shape2(channel_out, shape_out[2] * shape_out[3]));
	temp_sumall_.Resize(mshadow::Shape1(shape_out[2] * shape_out[3]));
    
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
    if (! (shape_in[0] == top[0]->data.shape_[0])) {
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
    mshadow::Tensor<xpu, 3> weight_data = this->params[0].data_d3();
    mshadow::Tensor<xpu, 2> bias_data = this->params[1].data_d2();
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;
    const index_t nbatch = bottom_data.size(0);
	top_data = 0;
	
	if (dim == 1) {
	  top_len = top_len_y;
	}

    for (index_t i = 0; i < nbatch; ++i) {
	  if (dim != 1) {
		top_len[i][0] = top_len_x;
		top_len[i][1] = top_len_y;
	  }
      unpack_patch2col_var(temp_col_, bottom_data[i], bottom_len_y, bottom_len_x,
					   kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x);
	  // PrintTensor("temp_col_", temp_col_);
	  // PrintTensor("weight_data", weight_data);
	  for (int ch = 0; ch < channel_out; ++ch) {
        temp_data_ = temp_col_ * weight_data[ch];
		temp_sumall_ = sumall_except_dim<0>(temp_data_);
		if (!no_bias) {
			temp_sumall_ += bias_data[ch];
		}
		top_data[i][ch] = reshape(temp_sumall_, mshadow::Shape2(shape_out[2], shape_out[3]));
	  }
	  // PrintTensor("temp_data_", temp_data_);

    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 3> weight_data = this->params[0].data_d3();
    mshadow::Tensor<xpu, 3> weight_diff = this->params[0].diff_d3();
    mshadow::Tensor<xpu, 2> bias_diff = this->params[1].diff_d2();
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;
    const index_t nbatch = bottom_data.size(0);
        
    for (int i = 0; i < nbatch; ++i) {
	  for (index_t ch = 0; ch < temp_diff_.size(0); ++ch) {
	    for (index_t idx = 0; idx < temp_diff_.size(1); ++idx) {
		  int row = idx / top_len_x;
		  int col = idx % top_len_x;
		  temp_diff_[ch][idx] = top_diff[i][ch][row][col];
		}
	  }
	  // PrintTensor("temp_diff_", temp_diff_);

      unpack_patch2col_var(temp_col_, bottom_data[i], bottom_len_y, bottom_len_x, 
					   kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x);
	  // PrintTensor("temp_col_", temp_col_);

      if (this->prop_grad[0]) {
	    for (index_t ch = 0; ch < channel_out; ++ch) {
		  for (index_t m = 0; m < shape_out[2]*shape_out[3]; ++m) {
			float temp_data_value = temp_diff_[ch][m];
		    for (index_t k = 0; k < channel_in * kernel_x * kernel_y; ++k) {
		      weight_diff[ch][m][k] += temp_data_value * temp_col_[m][k];
		    }
		  }
        }
	    // PrintTensor("weight_diff", weight_diff);
	  }

	  if (!no_bias && this->prop_grad[1]) {
		bias_diff += temp_diff_;
	  }

      if (this->prop_error[0]) {
	    for (index_t ch = 0; ch < channel_out; ++ch) {
		  for (index_t m = 0; m < shape_out[2]*shape_out[3]; ++m) {
			float temp_data_value = temp_diff_[ch][m];
		    for (index_t k = 0; k < channel_in * kernel_x * kernel_y; ++k) {
		      temp_dif_[m][k] += temp_data_value * weight_data[ch][m][k];
		    }
		  }
        }
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
  int top_len_x, top_len_y, bottom_len_x, bottom_len_y;

  mshadow::Shape<4> shape_in;
  mshadow::Shape<4> shape_out;

  mshadow::TensorContainer<xpu, 2> temp_col_;
  mshadow::TensorContainer<xpu, 2> temp_dif_;
  mshadow::TensorContainer<xpu, 2> temp_data_;
  mshadow::TensorContainer<xpu, 2> temp_diff_;
  mshadow::TensorContainer<xpu, 1> temp_sumall_;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_CONVOLUTION_VAR_LAYER_INL_HPP_
