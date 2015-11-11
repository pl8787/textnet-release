#ifndef TEXTNET_LAYER_SELECT_SUB_REP_BY_TOKEN_LAYER_INL_HPP_
#define TEXTNET_LAYER_SELECT_SUB_REP_BY_TOKEN_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

// this layer is built for char lstm, select the representations by space
namespace textnet {
namespace layer {

template<typename xpu>
class SelectSubRepByTokenLayer : public Layer<xpu>{
 public:
  SelectSubRepByTokenLayer(LayerType type) { this->layer_type = type; }
  virtual ~SelectSubRepByTokenLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; } // token sequence, representation sequence
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
	this->defaults["max_length"] = SettingV(0);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["token"] = SettingV();

    Layer<xpu>::Require();
  }
  
  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;

  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "SelectSubRepByTokenLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SelectSubRepByTokenLayer:top size problem.");

    token = setting["token"].iVal();
	max_length = setting["max_length"].iVal();
    utils::Check(token >= 0, "SelectSubRepByTokenLayer: token setting error.");
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "SelectSubRepByTokenLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SelectSubRepByTokenLayer:top size problem.");
                  
    // the shape of bottom 0: (batch_size, doc_num, 1, doc_len)
    // the shape of bottom 1: (batch_size, doc_num, doc_len, feat_dim)
    utils::Check(bottom[0]->data.size(2) == 1, "SelectSubRepByTokenLayer: input size problem.");
    utils::Check(bottom[0]->data.size(0) == bottom[1]->data.size(0), "SelectSubRepByTokenLayer: input size problem.");
    utils::Check(bottom[0]->data.size(1) == bottom[1]->data.size(1), "SelectSubRepByTokenLayer: input size problem.");
    utils::Check(bottom[0]->data.size(3) == bottom[1]->data.size(2), "SelectSubRepByTokenLayer: input size problem.");
                  
	if (max_length == 0) {
	  doc_len = bottom[0]->data.size(3);
	} else {
      doc_len = max_length;
	}
    top[0]->Resize(bottom[1]->data.size(0), bottom[1]->data.size(1), doc_len, bottom[1]->data.size(3), true);

	if (show_info) {
	  bottom[0]->PrintShape("bottom0");
	  bottom[1]->PrintShape("bottom1");
	  top[0]->PrintShape("top0");
	}
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (top[0]->data.size(0) != bottom[0]->data.size(0)) {
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
	Tensor2D bottom0_len = bottom[0]->length;
	Tensor2D bottom1_len = bottom[1]->length;
    for (int i = 0; i < bottom0_len.size(0); ++i) 
      for (int j = 0; j < bottom0_len.size(1); ++j) 
        utils::Check(bottom0_len[i][j] == bottom1_len[i][j], "SelectSubRepByTokenLayer: input length error.");

    Tensor4D bottom0_data = bottom[0]->data;
    Tensor4D bottom1_data = bottom[1]->data;
    Tensor4D top_data     = top[0]->data;
	Tensor2D top_len      = top[0]->length;

	top_data = 0.f, top_len = -1;
    for (int batch_idx = 0; batch_idx < bottom0_data.size(0); ++batch_idx) {
      for (int doc_idx = 0; doc_idx < bottom0_data.size(1); ++doc_idx) {
        int b_len = bottom0_len[batch_idx][doc_idx];
        utils::Check(b_len > 0, "SelectSubRepByTokenLayer: bottom doc length error.");
        
        int t_len = 0;
        for (int i = 0; i < b_len; ++i) {
          if (int(bottom0_data[batch_idx][doc_idx][0][i]) == token) {
            top_data[batch_idx][doc_idx][t_len++] = F<op::identity>(bottom1_data[batch_idx][doc_idx][i]);
          }
        }
        utils::Check(t_len > 0, "SelectSubRepByTokenLayer: no token in one sentence. t_len:%d", t_len);
        utils::Check(t_len < doc_len, "SelectSubRepByTokenLayer: token more than max_length. t_len:%d", t_len);
        top_len[batch_idx][doc_idx] = t_len;
		// utils::Printf("b %d d %d : len %d\n", batch_idx, doc_idx, t_len);
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor2D bottom0_len  = bottom[0]->length;
    Tensor4D bottom0_data = bottom[0]->data;
    Tensor4D bottom1_diff = bottom[1]->diff;
    Tensor4D top_diff     = top[0]->diff;
	Tensor2D top_len      = top[0]->length;

    for (int batch_idx = 0; batch_idx < bottom0_data.size(0); ++batch_idx) {
      for (int doc_idx = 0; doc_idx < bottom0_data.size(1); ++doc_idx) {
        int b_len = bottom0_len[batch_idx][doc_idx];
        utils::Check(b_len > 0, "SelectSubRepByTokenLayer: bottom doc length error.");
        
        int t_len = 0;
        for (int i = 0; i < b_len; ++i) {
          if (int(bottom0_data[batch_idx][doc_idx][0][i]) == token) {
            bottom1_diff[batch_idx][doc_idx][i] += top_diff[batch_idx][doc_idx][t_len++];
          }
        }
        utils::Check(t_len == top_len[batch_idx][doc_idx], "SelectSubRepByTokenLayer: token number not match. t_len:%d, top:%d", t_len, top_len[batch_idx][doc_idx]);
        utils::Check(t_len < doc_len, "SelectSubRepByTokenLayer: token more than max_length. t_len:%s", t_len);
      }
    }
  }
  
 protected:
  int token;
  int max_length;
  int doc_len;
};
}  // namespace layer
}  // namespace textnet
#endif  

