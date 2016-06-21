#ifndef TEXTNET_LAYER_GATING_LAYER_INL_HPP_
#define TEXTNET_LAYER_GATING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class GatingLayer : public Layer<xpu> {
 public:
  GatingLayer(LayerType type) { this->layer_type = type; }
  virtual ~GatingLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 1; }
  
  virtual void Require() {
    // default value, just set the value you want
	this->defaults["gate_type"] = SettingV("word-wise");
	// word-wise : each word has a gate weight
	// word-share : weight depended on word vector
	this->defaults["activefun_type"] = SettingV("linear");
	// linear : indentity
	// sigmoid : sigmoid * 2
	// tanh : tanh + 1

    // require value, set to SettingV(),
    // it will force custom to set in config
	this->defaults["word_count"] = SettingV();
	this->defaults["feat_size"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "GatingLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GatingLayer: top size problem.");
                            
    gate_type = setting["gate_type"].sVal();
	word_count = setting["word_count"].iVal();
	feat_size = setting["feat_size"].iVal();
	activefun_type = setting["activefun_type"].sVal();

	utils::Check(feat_size == bottom[0]->data.size(3), "GatingLayer: feat size not fit");
	utils::Check(gate_type == "word-wise" || gate_type == "word-share", 
			"GatingLayer: Only support word-wise or word-share.");
	utils::Check(activefun_type == "linear" || activefun_type == "sigmoid" 
			     || activefun_type == "tanh" || activefun_type == "relu",
			"GatingLayer: Only support linear or sigmoid or tanh or relu.");

    this->params.resize(1);
	if (gate_type == "word-wise") {
      this->params[0].Resize(word_count, 1, 1, 1, true);
	} else if (gate_type == "word-share") {
      this->params[0].Resize(feat_size, 1, 1, 1, true);
	}
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
          w_setting, this->prnd_);
    this->params[0].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
          w_updater, this->prnd_);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "GatingLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "GatingLayer: top size problem.");
    
	utils::Check(feat_size == bottom[0]->data.size(3), "GatingLayer: feat_size error.");

    nbatch = bottom[0]->data.size(0);
    int num_seq    = bottom[0]->data.size(1);
    int max_length = bottom[0]->data.size(2);
    
    top[0]->Resize(nbatch, num_seq, max_length, feat_size, true);
	
	total_words = nbatch * num_seq * max_length;
    gate_value.Resize(mshadow::Shape2(total_words, 1));
    gate_grad.Resize(mshadow::Shape2(1, total_words));
	word_bias.Resize(mshadow::Shape2(total_words, 1));	
	word_p_diff.Resize(mshadow::Shape2(total_words, feat_size));
	word_sum.Resize(mshadow::Shape1(total_words));
	
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

	// propagate sentence length
    top[0]->length = F<op::identity>(bottom[0]->length);

	// Word wise vector / matrix
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2_reverse();
	mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 2> top_data    = top[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> gate_data   = this->params[0].data_d2();
    
	int word_idx = -1;

	if (gate_type == "word-wise") {
      for (int i = 0; i < total_words; ++i) {
		word_idx = static_cast<int>(bottom1_data[i]);
		if (word_idx == -1) continue;
		gate_value[i][0] = gate_data[word_idx][0];
	  }
	} else if (gate_type == "word-share") {
      gate_value  = dot(bottom0_data,  gate_data);
	}

	// Apply active function
	if (activefun_type == "sigmoid") {
      gate_value = F<gate_sigmoid>(gate_value, 2.0f);
	} else if (activefun_type == "tanh") {
      gate_value = F<gate_tanh>(gate_value);
	} else if (activefun_type == "relu") {
	  gate_value = F<gate_relu>(gate_value);
	}

	for (int i = 0; i < total_words; ++i) {
	  word_idx = static_cast<int>(bottom1_data[i]);
	  if (word_idx == -1) continue;
	  top_data[i] = bottom0_data[i] * gate_value[i][0];
	}
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> top_diff     = top[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom0_diff = bottom[0]->diff_d2_reverse();
	mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 2> gate_data    = this->params[0].data_d2();
	mshadow::Tensor<xpu, 2> gate_diff    = this->params[0].diff_d2();
	
	gate_diff = 0;

	if (this->prop_error[0] || this->prop_grad[0]) {
		word_p_diff = bottom0_data * top_diff;
		word_sum = sumall_except_dim<0>(word_p_diff);
		if (activefun_type == "sigmoid") {
			gate_grad = F<gate_sigmoid_grad>(gate_value.T(), 2.0f); 
			word_sum *= gate_grad[0];
		} else if (activefun_type == "tanh") {
			gate_grad = F<gate_tanh_grad>(gate_value.T());
			word_sum *= gate_grad[0];
		} else if (activefun_type == "relu") {
			gate_grad = F<gate_relu_grad>(gate_value.T());
			word_sum *= gate_grad[0];
		}
	}

	if (this->prop_error[0]) {
	  for (int i = 0; i < total_words; ++i) {
		int word_idx = static_cast<int>(bottom1_data[i]);
		if (word_idx == -1) continue;
        bottom0_diff[i] += top_diff[i] * gate_value[i][0];
	  }
	  if (gate_type == "word-share") {
		// dot(gate_data, top_diff)
		for (int i = 0; i < total_words; ++i) {
		  int word_idx = static_cast<int>(bottom1_data[i]);
		  if (word_idx == -1) continue;
          bottom0_diff.Slice(i, i+1) += word_sum[i] * gate_data.T();
		}
	  }
	}

	if (this->prop_grad[0]) {
	  if (gate_type == "word-wise") {
        for (int i = 0; i < total_words; ++i) {
		  int word_idx = static_cast<int>(bottom1_data[i]);
		  if (word_idx == -1) continue;
          gate_diff[word_idx][0] += word_sum[i];
		}
	  } else if (gate_type == "word-share") {
		for (int i = 0; i < total_words; ++i) {
		  int word_idx = static_cast<int>(bottom1_data[i]);
		  if (word_idx == -1) continue;
          gate_diff += word_sum[i] * bottom0_data.Slice(i, i+1).T();
		}
	  }
	}
  }
  // Active function for gate only
  /*! \brief sigmoid unit */
  struct gate_sigmoid {
    MSHADOW_XINLINE static real_t Map(real_t a, real_t rate) {
      return rate / (1.0f + expf(-a));
    }
  };
  struct gate_sigmoid_grad {
    MSHADOW_XINLINE static real_t Map(real_t a, real_t rate) {
      return a * (1.0f - a / rate);
    }
  };
  /*! \brief Rectified Linear Operation */
  struct gate_relu {
    MSHADOW_XINLINE static real_t Map(real_t a) {
      using namespace std;
      return max(a, 0.0f);
    }
  };
  struct gate_relu_grad {
    MSHADOW_XINLINE static real_t Map(real_t a) {
      return a > 0.0f ? 1.0f : 0.0f;
    }
  };
  /*! \brief Rectified Linear Operation */
  struct gate_tanh {
    MSHADOW_XINLINE static real_t Map(real_t a) {
      return tanhf( a ) + 1.0f;
    }
  };
  
  struct gate_tanh_grad {
    MSHADOW_XINLINE static real_t Map(real_t a) {
      return 1.0f - (a-1.0f) * (a-1.0f);
    }
  };



 protected:
  /*! \brief random number generator */
  std::string gate_type;
  std::string activefun_type;
  int feat_size;
  int word_count;
  int nbatch;
  // Temp var
  mshadow::TensorContainer<xpu, 2> gate_value;
  mshadow::TensorContainer<xpu, 2> gate_grad;
  mshadow::TensorContainer<xpu, 2> word_bias;
  mshadow::TensorContainer<xpu, 2> word_p_diff;
  mshadow::TensorContainer<xpu, 1> word_sum;
  int total_words;
  
};
}  // namespace layer
}  // namespace textnet
#endif
