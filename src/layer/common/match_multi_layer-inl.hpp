#ifndef TEXTNET_LAYER_MATCH_MULTI_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_MULTI_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class MatchMultiLayer : public Layer<xpu>{
 public:
  MatchMultiLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchMultiLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { 
	  if (output_len) {
		return 3;
	  } else {
		return 1; 
	  }
  }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["op"] = SettingV("xor"); 
	this->defaults["output_len"] = SettingV(false);
    // xor: can not bp
    // mul: can bp
    // plus: can bp
    // cos: can bp
    // euc: can bp
    // euc_exp: can bp, see wenpeng's paper
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["candids"] = SettingV();
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    op = setting["op"].sVal();
    candids = setting["candids"].iVal();
	output_len = setting["output_len"].bVal();

    utils::Check(op=="xor" || op=="mul" || op=="plus" || op=="cos" || op == "minus" ||\
                 op=="elemwise_product" || op=="elemwise_plus" || op=="euc" || op=="euc_exp",
                 "MatchMultiLayer: one of xor, mul, plus, cos, minus, elemwise_product, elemwise_plus, euc or euc_exp.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MatchMultiLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MatchMultiLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    utils::Check(nbatch % (candids + 1) == 0,
                  "MatchMultiLayer: prev nbatch must div (candids + 1).");

    nbatch = nbatch / (candids + 1);

    if (op == "xor") {
      doc_len = bottom[0]->data.size(3);
    } else {
      doc_len = bottom[0]->data.size(2);
      feat_size = bottom[0]->data.size(3);
    }        
    
	if (op == "elemwise_product" || op == "elemwise_plus") {
	  // Set data shape to (nbatch * candids, feat_size, doc_len, doc_len)
	  // Set length shape to (nbatch * candids, 2)
	  top[0]->Resize(nbatch * candids, feat_size, doc_len, doc_len, nbatch * candids, 2, true);
	} else {
	  // Set data shape to (nbatch * candids, 1, doc_len, doc_len)
	  // Set length shape to (nbatch * candids, 2)
	  top[0]->Resize(nbatch * candids, 1, doc_len, doc_len, nbatch * candids, 2, true);
	}

	if (output_len) {
	  top[1]->Resize(nbatch * candids, 1, 1, 1);
	  top[2]->Resize(nbatch * candids, 1, 1, 1);
	}
    
    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }

    if (op == "cos") {
      m_norm.Resize(mshadow::Shape2(nbatch * (candids+1), doc_len), 0.f);
      m_dot.Resize(mshadow::Shape3(nbatch * candids, doc_len, doc_len), 0.f);
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (nbatch != bottom[0]->data.size(0) / (candids+1)) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }

  void ForwardOne(int s1_idx, int s2_idx, int out_idx, 
                    mshadow::Tensor<xpu, 2> &bottom_data2, mshadow::Tensor<xpu, 4> &bottom_data4,
                    mshadow::Tensor<xpu, 1> &bottom_len, mshadow::Tensor<xpu, 4> &top_data) {
    int len_0 = -1, len_1 = -1;
    len_0 = bottom_len[s1_idx];
    len_1 = bottom_len[s2_idx];
    utils::Check(len_0 >= 0 && len_1 >= 0, "MatchMultiLayer: length error.");
    utils::Check(len_0 <= doc_len && len_1 <= doc_len, "MatchMultiLayer: length error.");

    for (int j = 0; j < len_0; j++) {
      for (int k = 0; k < len_1; k++) {
          if (op == "xor") {
            utils::Check(bottom_data2[s1_idx][j]!=-1 && bottom_data2[s2_idx][k]!=-1, 
              "In Match Layer: please check length setting. (%d, %d, %d, %d)", s1_idx, s2_idx, j, k);
            top_data[out_idx][0][j][k] = (bottom_data2[s1_idx][j] == bottom_data2[s2_idx][k]) ? 1 : 0;
          } else if (op == "mul") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[out_idx][0][j][k] += bottom_data4[s1_idx][0][j][m] * bottom_data4[s2_idx][0][k][m];
            }
          } else if (op == "plus") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[out_idx][0][j][k] += bottom_data4[s1_idx][0][j][m] + bottom_data4[s2_idx][0][k][m];
            }
          } else if (op == "minus") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[out_idx][0][j][k] += bottom_data4[s1_idx][0][j][m] - bottom_data4[s2_idx][0][k][m];
            }
          } else if (op == "cos") {
            for (int m = 0; m < feat_size; ++m) {
              m_dot[out_idx][j][k] += bottom_data4[s1_idx][0][j][m] * bottom_data4[s2_idx][0][k][m];
            }
            top_data[out_idx][0][j][k] = m_dot[out_idx][j][k] / (m_norm[s1_idx][j] * m_norm[s2_idx][k]);    
          } else if (op =="euc") {
            float sum_elem_square = 0.f;
            for (int m = 0; m < feat_size; ++m) {
              float sub_elem = bottom_data4[s1_idx][0][j][m] - bottom_data4[s2_idx][0][k][m];
              sum_elem_square += sub_elem * sub_elem;
            }
            // top_data[out_idx][0][j][k] = pow(sum_elem_square, 0.5);
            top_data[out_idx][0][j][k] = sum_elem_square;
          } else if (op =="euc_exp") { // by wengpeng ying, no sqrt
            float sum_elem_square = 0.f;
            for (int m = 0; m < feat_size; ++m) {
              float sub_elem = bottom_data4[s1_idx][0][j][m] - bottom_data4[s2_idx][0][k][m];
              sum_elem_square += sub_elem * sub_elem;
            }
            top_data[out_idx][0][j][k] = exp(-(sum_elem_square)/(2*2.f)); // beta is set to 2.f
		  } else if (op =="elemwise_product") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[out_idx][m][j][k] = bottom_data4[s1_idx][0][j][m] * bottom_data4[s2_idx][0][k][m];
            }
          } else if (op =="elemwise_plus") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[out_idx][m][j][k] = bottom_data4[s1_idx][0][j][m] + bottom_data4[s2_idx][0][k][m];
            }
          }  else {
            utils::Error("In Match Layer: no op named %s.\n", op.c_str());
          }
        }
      }
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom_data2 = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 4> bottom_data4 = bottom[0]->data;
    mshadow::Tensor<xpu, 1> bottom_len = bottom[0]->length_d1();
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
	mshadow::Tensor<xpu, 2> top_len = top[0]->length;
	mshadow::Tensor<xpu, 1> top_len1;
	mshadow::Tensor<xpu, 1> top_len2;

	if (output_len) {
		top_len1 = top[1]->length_d1();
		top_len2 = top[2]->length_d1();
	}

    top_data = 0.0f;
    m_norm = 0.0f;
    m_dot = 0.0f;
    
    if (op == "cos") {
      for (int i = 0; i < nbatch*(candids+1); i++) {
        int len_0 = -1;
        len_0 = bottom_len[i];
        utils::Check(len_0 >= 0, "MatchMultiLayer: length error.");
        utils::Check(len_0 <= doc_len, "MatchMultiLayer: length error.");

        for (int j = 0; j < len_0; j++) {
          for (int m = 0; m < feat_size; ++m) {
            m_norm[i][j] += bottom_data4[i][0][j][m] * bottom_data4[i][0][j][m];
          }
        }
      }
      m_norm = F<op::square_root >(m_norm);
    }

    for (int i = 0; i < nbatch; ++i) {
      for (int c = 1; c < candids+1; ++c) {
        ForwardOne(i*(candids+1), i*(candids+1)+c, i*candids+c-1, bottom_data2, bottom_data4, bottom_len, top_data);
		top_len[i*candids+c-1][0] = bottom_len[i*(candids+1)];
		top_len[i*candids+c-1][1] = bottom_len[i*(candids+1)+c];
		if (output_len) {
		  top_len1[i*candids+c-1] = bottom_len[i*(candids+1)];
		  top_len2[i*candids+c-1] = bottom_len[i*(candids+1)+c];
		}
      }
    }
  }
  
  void BackpropOne(int s1_idx, int s2_idx, int in_idx, 
                    mshadow::Tensor<xpu, 4> &top_diff, mshadow::Tensor<xpu, 4> &top_data, 
                    mshadow::Tensor<xpu, 4> &bottom_data, mshadow::Tensor<xpu, 4> &bottom_diff, mshadow::Tensor<xpu, 1> &bottom_len) {
    int len_0 = -1, len_1 = -1;
    len_0 = bottom_len[s1_idx];
    len_1 = bottom_len[s2_idx];
    utils::Check(len_0 >= 0 && len_1 >= 0, "MatchMultiLayer: length error.");
    utils::Check(len_0 <= doc_len && len_1 <= doc_len, "MatchMultiLayer: length error.");
  
    for (int j = 0; j < len_0; ++j) {
      for (int k = 0; k < len_1; ++k) {
        for (int m = 0; m < feat_size; ++m) {
          if (op == "mul") {  
              bottom_diff[s1_idx][0][j][m] += bottom_data[s2_idx][0][k][m] * top_diff[in_idx][0][j][k];
                  bottom_diff[s2_idx][0][k][m] += bottom_data[s1_idx][0][j][m] * top_diff[in_idx][0][j][k];
            } else if (op == "plus") {
                bottom_diff[s1_idx][0][j][m] += top_diff[in_idx][0][j][k];
                bottom_diff[s2_idx][0][k][m] += top_diff[in_idx][0][j][k];
            } else if (op == "minus") {
                bottom_diff[s1_idx][0][j][m] += top_diff[in_idx][0][j][k];
                bottom_diff[s2_idx][0][k][m] -= top_diff[in_idx][0][j][k];
            } else if (op == "cos") {
                bottom_diff[s1_idx][0][j][m] += (bottom_data[s2_idx][0][k][m] / (m_norm[s1_idx][j] * m_norm[s2_idx][k]) 
                                           - bottom_data[s1_idx][0][j][m] * m_dot[in_idx][j][k] / (pow(m_norm[s1_idx][j], 3) * m_norm[s2_idx][k]))
                                          * top_diff[in_idx][0][j][k];
                bottom_diff[s2_idx][0][k][m] += (bottom_data[s1_idx][0][j][m] / (m_norm[s1_idx][j] * m_norm[s2_idx][k]) 
                                           - bottom_data[s2_idx][0][k][m] * m_dot[in_idx][j][k] / (m_norm[s1_idx][j] * pow(m_norm[s2_idx][k], 3)))
                                          * top_diff[in_idx][0][j][k];
            } else if (op == "euc") {
            float distance = top_data[in_idx][0][j][k];
            float sub_elem = bottom_data[s1_idx][0][j][m] - bottom_data[s2_idx][0][k][m];
              // bottom0_diff[i][0][j][m] += top_diff[i][0][j][k] * (1/(2*distance)) * 2*(sub_elem);
              bottom_diff[s1_idx][0][j][m] += top_diff[in_idx][0][j][k] * 2*(sub_elem);
                // bottom1_diff[i][0][k][m] += top_diff[i][0][j][k] * (1/(2*distance)) * 2*(-sub_elem);
                bottom_diff[s2_idx][0][k][m] += top_diff[in_idx][0][j][k] * 2*(-sub_elem);
            } else if (op == "euc_exp") {
              float distance = top_data[in_idx][0][j][k];
              float sub_elem = bottom_data[s1_idx][0][j][m] - bottom_data[s2_idx][0][k][m];
              bottom_diff[s1_idx][0][j][m] += top_diff[in_idx][0][j][k] * distance * (-1/(2*2.f)) * 2*(sub_elem);
              bottom_diff[s2_idx][0][k][m] += top_diff[in_idx][0][j][k] * distance * (-1/(2*2.f)) * 2*(-sub_elem);
            } else if (op == "elemwise_product") {
                bottom_diff[s1_idx][0][j][m] += bottom_data[s2_idx][0][k][m] * top_diff[in_idx][m][j][k];
                bottom_diff[s2_idx][0][k][m] += bottom_data[s1_idx][0][j][m] * top_diff[in_idx][m][j][k];
            } else if (op == "elemwise_plus") {
                bottom_diff[s1_idx][0][j][m] += top_diff[in_idx][m][j][k];
                bottom_diff[s2_idx][0][k][m] += top_diff[in_idx][m][j][k];
            }
          }
        }
    }

  }

  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 1> bottom_len = bottom[0]->length_d1();

    if (op == "xor") {
      // do nothing
      return;
    }
      
    if (!this->prop_error[0]) return;

    for (int i = 0; i < nbatch; ++i) {
      for (int c = 1; c < candids+1; ++c) {
        BackpropOne(i*(candids+1), i*(candids+1)+c, i*candids+c-1, top_diff, top_data, bottom_data, bottom_diff, bottom_len);
      }
    }
  }
  
 protected:
  int doc_len;
  int feat_size;
  int nbatch;
  int candids;
  std::string op;
  bool output_len;

  mshadow::TensorContainer<xpu, 2> m_norm;
  mshadow::TensorContainer<xpu, 3> m_dot;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MATCH_MULTI_LAYER_INL_HPP_

