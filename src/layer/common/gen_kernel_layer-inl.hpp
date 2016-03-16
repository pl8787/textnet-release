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
    max_kernel_size = shape_in[2];
    
	utils::Check(shape_in[1] == 1, "GenKernelLayer: kernel out must be 1.");
	utils::Check(shape_in[3] == 1, "GenKernelLayer: last dimension must be 1.");
    
	if (kernel_mode == "single") {
      channel_out = max_kernel_size;
      top[0]->Resize(nbatch, channel_out, max_kernel_size, 1, nbatch, 3, true);
    } else if (kernel_mode == "diag") {
      channel_out = 1;
      top[0]->Resize(nbatch, channel_out, max_kernel_size * max_kernel_size, 1, nbatch, 3, true);
    } else if (kernel_mode == "permutation") {
      channel_out = 1;
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
    top_len = F<op::identity>(bottom_len);

	if (kernel_mode == "single") {
      for (int i = 0; i < nbatch; ++i) {
        for (int j = 0; j < max_kernel_size; ++j) {
          top_data[i][j][j] = bottom_data[i][0][j];
        }
      }
	} else if (kernel_mode == "diag") {
      for (int i = 0; i < nbatch; ++i) {
        for (int j = 0; j < max_kernel_size; ++j) {
          top_data[i][0][j*max_kernel_size+j] = bottom_data[i][0][j];
        }
        top_len[i][2] = bottom_len[i][1];
      }
    } else if (kernel_mode == "permutation") {
      vector<int> cur_idx(max_kernel_size);
      for (int i = 0; i < nbatch; ++i) {
        for (int j = 0; j < max_kernel_size; ++j) {
          cur_idx[j] = max_kernel_size - j - 1;
        }
        int cur_channel = 0;
        do {
          for (int j = 0; j < max_kernel_size; ++j) {
            int k = cur_idx[max_kernel_size - j - 1];
            top_data[i][cur_channel][k*max_kernel_size+j] = bottom_data[i][0][j];
          }
          cur_channel += 1;
        } while ( std::prev_permutation(cur_idx.begin(), cur_idx.end()) );
        top_len[i][2] = bottom_len[i][1];
      }
    }

  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

  }

 protected:
  /*! \brief random number generator */
  int nbatch;
  int max_kernel_size;
  int channel_out;
  string kernel_mode;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_GEN_KERNEL_LAYER_INL_HPP_
