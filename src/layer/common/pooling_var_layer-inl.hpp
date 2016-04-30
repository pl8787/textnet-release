#ifndef TEXTNET_LAYER_POOLING_VAR_LAYER_INL_HPP_
#define TEXTNET_LAYER_POOLING_VAR_LAYER_INL_HPP_

#include <iostream>
#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {
  
template<typename xpu>
class PoolingVarLayer : public Layer<xpu> {
 public:
  PoolingVarLayer(LayerType type) { this->layer_type = type; }
  virtual ~PoolingVarLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["pad_x"] = SettingV(0);
    this->defaults["pad_y"] = SettingV(0);
    this->defaults["stride_x"] = SettingV(1);
    this->defaults["stride_y"] = SettingV(1);
	this->defaults["dim"] = SettingV(2);
    this->defaults["pooling_mode"] = SettingV("max"); // max or avg 

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["kernel_x"] = SettingV();
    this->defaults["kernel_y"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);

    utils::Check(bottom.size() == BottomNodeNum(),
                  "PoolingVarLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PoolingVarLayer:top size problem.");    
                           
    kernel_x = setting["kernel_x"].iVal();
    kernel_y = setting["kernel_y"].iVal();
    pad_x = setting["pad_x"].iVal();
    pad_y = setting["pad_y"].iVal();
    stride_x = setting["stride_x"].iVal();
    stride_y = setting["stride_y"].iVal();
	dim = setting["dim"].iVal();
    pooling_mode = setting["pooling_mode"].sVal();
    
    utils::Check(pooling_mode == "max" || pooling_mode == "avg",
                  "PoolingVarLayer: pooling mode error.");    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "PoolingVarLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "PoolingVarLayer:top size problem.");
                  
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out;

    nbatch = shape_in[0];
    channel = shape_in[1];

	if (dim == 1) {
		shape_out = mshadow::Shape4(nbatch, channel,
				(shape_in[2] + pad_y * 2 - kernel_y) / stride_y + 1,
				1);
	} else {
		shape_out = mshadow::Shape4(nbatch, channel, 
                (shape_in[2] + pad_y * 2 - kernel_y) / stride_y + 1,
                (shape_in[3] + pad_x * 2 - kernel_x) / stride_x + 1);
	}
	mshadow::Shape<2> shape_len;
	shape_len = bottom[0]->length.shape_;
	top[0]->Resize(shape_out, shape_len);

    mask.Resize(shape_out);

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
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], channel, 
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
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
	mshadow::Tensor<xpu, 2> top_len = top[0]->length;
	mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    
    if (pooling_mode == "max") {
	  top_data = 0.0;
    } else if (pooling_mode == "avg") {
      top_data = 0.0;
    }

	int top_len_x = 0, top_len_y = 0, bottom_len_x = 0, bottom_len_y = 0;
    for (index_t i = 0; i < nbatch; ++i) {
      if (dim == 1) {
          top_len[i][0] = int(bottom_len[i][0] + pad_y * 2 - kernel_y) / stride_y + 1; // all input channels shoud have the same length
		  utils::Check(top_len[i][0] > 0, "PoolingVarLayer: top_len must positive. i=%d, bottom_len=%f, top_len=%f", i, bottom_len[i][0], top_len[i][0]);
		  utils::Check(pad_x == 0, "PoolingVarLayer: dim=1 pad_x!=0.");
		  top_len_x = 1;
		  top_len_y = top_len[i][0];
		  bottom_len_x = kernel_x;
		  bottom_len_y = bottom_len[i][0];
	  } else {
		  top_len[i][0] = int(bottom_len[i][0] + pad_y * 2 - kernel_y) / stride_y + 1;
		  top_len[i][1] = int(bottom_len[i][1] + pad_x * 2 - kernel_x) / stride_x + 1;
		  utils::Check(top_len[i][0] > 0 && top_len[i][1] > 0, "PoolingVarLayer: top_len must positive. i=%d, bottom_len=(%f,%f), top_len=(%f,%f)",
				  i, bottom_len[i][0], bottom_len[i][1], top_len[i][0], top_len[i][1]);
		  top_len_y = top_len[i][0];
		  top_len_x = top_len[i][1];
		  bottom_len_y = bottom_len[i][0];
		  bottom_len_x = bottom_len[i][1];
	  }

      for (int c = 0; c < channel; ++c) {
        for (int py = 0; py < top_len_y; ++py) {
          for (int px = 0; px < top_len_x; ++px) {
            int xstart = px * stride_x - pad_x;
            int ystart = py * stride_y - pad_y;
            int xend = min(xstart + kernel_x, bottom_len_x);
            int yend = min(ystart + kernel_y, bottom_len_y);
            xstart = max(xstart, 0);
            ystart = max(ystart, 0);
            int pooling_size = (xend - xstart) * (yend - ystart);

            if (pooling_mode == "max") {
              top_data[i][c][py][px] = -FLT_MAX;
              for (int y = ystart; y < yend; ++y) {
                for (int x = xstart; x < xend; ++x) {
                  if (bottom_data[i][c][y][x] > top_data[i][c][py][px]) {
                    top_data[i][c][py][px] = bottom_data[i][c][y][x];
                    mask[i][c][py][px] = y * bottom_len_x + x;
                  }
                }
              }
            } else if (pooling_mode == "avg") {
              for (int y = ystart; y < yend; ++y) {
                for (int x = xstart; x < xend; ++x) {
                  top_data[i][c][py][px] += bottom_data[i][c][y][x];
                }
              }
              top_data[i][c][py][px] /= pooling_size;
            }
          }
        }
      }

    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;
    
    if (!this->prop_error[0]) {
      return;
    }

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

      for (int c = 0; c < channel; ++c) {
        for (int py = 0; py < top_len_y; ++py) {
          for (int px = 0; px < top_len_x; ++px) {
            int xstart = px * stride_x - pad_x;
            int ystart = py * stride_y - pad_y;
            int xend = min(xstart + kernel_x, bottom_len_x);
            int yend = min(ystart + kernel_y, bottom_len_y);
            xstart = max(xstart, 0);
            ystart = max(ystart, 0);
            int pooling_size = (xend - xstart) * (yend - ystart);

            if (pooling_mode == "max") {
              int y = int(mask[i][c][py][px]) / bottom_len_x;
              int x = int(mask[i][c][py][px]) % bottom_len_x;
              bottom_diff[i][c][y][x] += top_diff[i][c][py][px];
            } else if (pooling_mode == "avg") {
              for (int y = ystart; y < yend; ++y) {
                for (int x = xstart; x < xend; ++x) {
                  if (bottom_data[i][c][y][x] > top_data[i][c][py][px]) {
                    bottom_diff[i][c][y][x] += top_diff[i][c][py][px] / pooling_size;
                  }
                }
              }
            }
          }
        }
      }
    }

  }
 protected:
  int kernel_x;
  int kernel_y;
  int stride_x;
  int stride_y;
  int pad_x;
  int pad_y;
  int channel;
  int nbatch;
  int dim;
  string pooling_mode;
  mshadow::TensorContainer<xpu, 4> mask;
  
};   // class PoolingVarLayer
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_POOLING_VAR_LAYER_INL_HPP_

