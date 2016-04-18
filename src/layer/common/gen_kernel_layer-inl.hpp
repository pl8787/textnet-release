#ifndef TEXTNET_LAYER_GEN_KERNEL_LAYER_INL_HPP_
#define TEXTNET_LAYER_GEN_KERNEL_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include <algorithm> 
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class GenKernelLayer : public Layer<xpu> {
 public:
  GenKernelLayer(LayerType type) { this->layer_type = type; }
  virtual ~GenKernelLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["kernel_mode"] = SettingV("single");
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
                  "GenKernelLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "GenKernelLayer:top size problem.");
                            
    kernel_mode = setting["kernel_mode"].sVal();

    utils::Check(kernel_mode == "single" || kernel_mode == "diag" || kernel_mode == "permutation", 
                 "GenKernelLayer: kernel mode unknown.");

  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "GenKernelLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "GenKernelLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    nbatch = shape_in[0];
    kernel_count = shape_in[1];
    max_kernel_size = shape_in[2];
    
	// utils::Check(shape_in[1] == 1, "GenKernelLayer: kernel out must be 1.");
	utils::Check(shape_in[3] == 1, "GenKernelLayer: last dimension must be 1.");
    
	if (kernel_mode == "single") {
      channel_out = max_kernel_size * kernel_count;
      top[0]->Resize(nbatch, channel_out, max_kernel_size, 1, nbatch, 3, true);
    } else if (kernel_mode == "diag") {
      channel_out = kernel_count;
      top[0]->Resize(nbatch, channel_out, max_kernel_size * max_kernel_size, 1, nbatch, 3, true);
    } else if (kernel_mode == "permutation") {
      channel_out = kernel_count;
      for (int i = 2; i <= max_kernel_size; ++i) {
        channel_out *= i;
      }
      top[0]->Resize(nbatch, channel_out, max_kernel_size * max_kernel_size, 1, nbatch, 3, true);
    }

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (nbatch != bottom[0]->data.size(0)) {
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
    mshadow::Tensor<xpu, 3> bottom_data = bottom[0]->data_d3();
    mshadow::Tensor<xpu, 3> top_data = top[0]->data_d3();
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;
	
	top_data = 0.0;
    // top_len = F<op::identity>(bottom_len);

    int length_idx = 1;
    int length_mode = bottom_len.size(1);
    if (length_mode == 1) {
      length_idx = 0;
    } else if (length_mode == 2) {
      length_idx = 0;
    } else if (length_mode == 3) {
      length_idx = 1;
    }

	if (kernel_mode == "single") {
      for (int i = 0; i < nbatch; ++i) {
        for (int k = 0; k < kernel_count; ++k) {
          for (int j = 0; j < max_kernel_size; ++j) {
            top_data[i][k*max_kernel_size+j][j] = bottom_data[i][k][j];
          }
        }
        top_len[i][0] = 1;
        top_len[i][1] = bottom_len[i][length_idx];
        top_len[i][2] = 1;
      }
	} else if (kernel_mode == "diag") {
      for (int i = 0; i < nbatch; ++i) {
        for (int k = 0; k < kernel_count; ++k) {
          for (int j = 0; j < max_kernel_size; ++j) {
            top_data[i][k][j*max_kernel_size+j] = bottom_data[i][k][j];
          }
        }
        top_len[i][0] = 1;
        top_len[i][1] = bottom_len[i][length_idx];
        top_len[i][2] = bottom_len[i][length_idx];
      }
    } else if (kernel_mode == "permutation") {
      vector<int> cur_idx(max_kernel_size);
      for (int i = 0; i < nbatch; ++i) {
        for (int j = 0; j < max_kernel_size; ++j) {
          cur_idx[j] = max_kernel_size - j - 1;
        }
        int cur_channel = 0;
        do {
          for (int k = 0; k < kernel_count; ++k) {
            for (int j = 0; j < max_kernel_size; ++j) {
              int s = cur_idx[max_kernel_size - j - 1];
              top_data[i][cur_channel][s*max_kernel_size+j] = bottom_data[i][k][j];
            }
            cur_channel += 1;
          }
        } while ( std::prev_permutation(cur_idx.begin(), cur_idx.end()) );
        top_len[i][0] = 1;
        top_len[i][1] = bottom_len[i][length_idx];
        top_len[i][2] = bottom_len[i][length_idx];
      }
    }

  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> bottom_data = bottom[0]->data_d3();
    mshadow::Tensor<xpu, 3> top_data = top[0]->data_d3();
    mshadow::Tensor<xpu, 3> bottom_diff = bottom[0]->diff_d3();
    mshadow::Tensor<xpu, 3> top_diff = top[0]->diff_d3();

	if (kernel_mode == "single") {
      for (int i = 0; i < nbatch; ++i) {
        for (int k = 0; k < kernel_count; ++k) {
          for (int j = 0; j < max_kernel_size; ++j) {
            bottom_diff[i][k][j] += top_diff[i][k*max_kernel_size+j][j];
          }
        }
      }
	} else if (kernel_mode == "diag") {
      for (int i = 0; i < nbatch; ++i) {
        for (int k = 0; k < kernel_count; ++k) {
          for (int j = 0; j < max_kernel_size; ++j) {
            bottom_diff[i][k][j] += top_diff[i][k][j*max_kernel_size+j];
          }
        }
      }
    } else if (kernel_mode == "permutation") {
      vector<int> cur_idx(max_kernel_size);
      for (int i = 0; i < nbatch; ++i) {
        for (int j = 0; j < max_kernel_size; ++j) {
          cur_idx[j] = max_kernel_size - j - 1;
        }
        int cur_channel = 0;
        do {
          for (int k = 0; k < kernel_count; ++k) {
            for (int j = 0; j < max_kernel_size; ++j) {
              int s = cur_idx[max_kernel_size - j - 1];
              bottom_diff[i][k][j] += top_diff[i][cur_channel][s*max_kernel_size+j];
            }
            cur_channel += 1;
          }
        } while ( std::prev_permutation(cur_idx.begin(), cur_idx.end()) );
      }
    }

  }

 protected:
  /*! \brief random number generator */
  int nbatch;
  int max_kernel_size;
  int channel_out;
  int kernel_count;
  string kernel_mode;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_GEN_KERNEL_LAYER_INL_HPP_
