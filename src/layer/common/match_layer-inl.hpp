#ifndef TEXTNET_LAYER_MATCH_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_LAYER_INL_HPP_

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
class MatchLayer : public Layer<xpu>{
 public:
  MatchLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["op"] = SettingV("xor"); 
    this->defaults["interval"] = SettingV(1); 
    this->defaults["max_element"] = SettingV(0); 
    this->defaults["is_var_len"] = SettingV(true); 
    // xor: can not bp
    // mul: can bp
    // plus: can bp
    // cos: can bp
    // euc: can bp
    // euc_exp: can bp, see wenpeng's paper
    
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
                  "MatchLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MatchLayer:top size problem.");
    op = setting["op"].sVal();
    is_var_len = setting["is_var_len"].bVal();
    interval = setting["interval"].iVal();
	max_element = setting["max_element"].iVal();
    if (interval != 1 || max_element != 0) {
      utils::Check(op != "cos", "MatchLayer: does not support cos when interval is set");
    }

    utils::Check(op=="xor" || op=="mul" || op=="plus" || op=="cos" || op == "minus" ||\
                 op=="elemwise_product" || op=="elemwise_plus" || op=="elemwise_cat" ||\
                 op=="euc" || op=="euc_exp" || op=="order",
                 "MatchLayer: one of xor, mul, plus, cos, minus, elemwise_product, euc or euc_exp.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MatchLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MatchLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    if (op == "xor") {
      doc0_len = bottom[0]->data.size(3);
      doc1_len = bottom[1]->data.size(3);
    } else if (op == "elemwise_cat") {
      doc0_len = bottom[0]->data.size(2);
      doc1_len = bottom[1]->data.size(2);
      feat0_size = bottom[0]->data.size(3);
      feat1_size = bottom[1]->data.size(3);
      feat_size = feat0_size + feat1_size;
    } else {
      doc0_len = bottom[0]->data.size(2);
      doc1_len = bottom[1]->data.size(2);
      feat0_size = bottom[0]->data.size(3);
      feat1_size = bottom[1]->data.size(3);
      utils::Check(feat0_size == feat1_size, "MatchLayer: feature size not equal.");
      feat_size = feat0_size;
    }        
                  
    if (op == "elemwise_product" || op == "elemwise_plus") {
	  // Set data shape to (nbatch, feat_size, doc0_len, doc1_len)
	  // Set length shape to (nbatch, 2)
      top[0]->Resize(nbatch, feat_size, doc0_len, doc1_len, nbatch, 2, true);
    } else if (op == "elemwise_cat") {
	  // Set data shape to (nbatch, feat_size, doc0_len, doc1_len)
	  // Set length shape to (nbatch, 2)
      top[0]->Resize(nbatch, feat_size, doc0_len, doc1_len, nbatch, 2, true);
    } else {
	  // Set data shape to (nbatch, 1, doc0_len, doc1_len)
	  // Set length shape to (nbatch, 2)
      top[0]->Resize(nbatch, 1, doc0_len, doc1_len, nbatch, 2, true);
    }

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        bottom[1]->PrintShape("bottom1");
        top[0]->PrintShape("top0");
    }

    if (op == "cos") {
      m_norm.Resize(mshadow::Shape3(nbatch, 2, max(doc0_len, doc1_len)), 0.f);
      pow_3_m_norm.Resize(mshadow::Shape3(nbatch, 2, max(doc0_len, doc1_len)), 0.f);
      m_dot.Resize(mshadow::Shape3(nbatch, doc0_len, doc1_len), 0.f);
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
    mshadow::Tensor<xpu, 2> bottom0_data2 = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> bottom1_data2 = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 4> bottom0_data4 = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom1_data4 = bottom[1]->data;
    mshadow::Tensor<xpu, 1> bottom0_len = bottom[0]->length_d1();
    mshadow::Tensor<xpu, 1> bottom1_len = bottom[1]->length_d1();
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
	mshadow::Tensor<xpu, 2> top_len = top[0]->length;
	int interval_0 = 1;
	int interval_1 = 1;

    top_data = 0.0f;
    m_norm = 0.0f;
    pow_3_m_norm = 0.0f;
    m_dot = 0.0f;

    if (op == "cos") {
      for (int i = 0; i < nbatch; i++) {
        int len_0 = -1, len_1 = -1;
        if (is_var_len) {
          len_0 = bottom0_len[i];
          len_1 = bottom1_len[i];
		  if (max_element != 0) {
			interval_0 = len_0 / max_element + 1;
			interval_1 = len_1 / max_element + 1;
		  } else if (interval != 1) {
			interval_0 = interval;
			interval_1 = interval;
		  }
        } else {
          len_0 = doc0_len;
          len_1 = doc1_len;
		  if (max_element != 0) {
			interval_0 = len_0 / max_element + 1;
			interval_1 = len_1 / max_element + 1;
		  } else if (interval != 1) {
			interval_0 = interval;
			interval_1 = interval;
		  }
        }
        utils::Check(len_0 >= 0 && len_1 >= 0, 
				"MatchLayer: length error negative. len_0=%d, len_1=%d.", len_0, len_1);
        utils::Check(len_0 <= doc0_len && len_1 <= doc1_len, 
				"MatchLayer: length error large. len_0=%d, len_1=%d, max=%d, max=%d.", len_0, len_1, doc0_len, doc1_len);
        for (int j = 0; j < len_0; j++) {
          for (int m = 0; m < feat_size; ++m) {
            m_norm[i][0][j] += bottom0_data4[i][0][j][m] * bottom0_data4[i][0][j][m];
          }
        }
        for (int k = 0; k < len_1; k++) {
          for (int m = 0; m < feat_size; ++m) {
            m_norm[i][1][k] += bottom1_data4[i][0][k][m] * bottom1_data4[i][0][k][m];
          }
        }
      }
      m_norm = F<op::square_root >(m_norm);
      pow_3_m_norm = F<op::pow_3>(m_norm);
    }

    for (int i = 0; i < nbatch; i++) {
      int len_0 = -1, len_1 = -1;
      if (is_var_len) {
        len_0 = bottom0_len[i];
        len_1 = bottom1_len[i];
		top_len[i][0] = len_0;
		top_len[i][1] = len_1;
		if (max_element != 0) {
		  interval_0 = len_0 / max_element + 1;
		  interval_1 = len_1 / max_element + 1;
		} else if (interval != 1) {
		  interval_0 = interval;
		  interval_1 = interval;
		}
		// utils::Printf("Mat: %d x %d, interval: %d x %d\n", len_0, len_1, interval_0, interval_1);
      } else {
        len_0 = doc0_len;
        len_1 = doc1_len;
		if (max_element != 0) {
		  interval_0 = len_0 / max_element + 1;
		  interval_1 = len_1 / max_element + 1;
		} else if (interval != 1) {
		  interval_0 = interval;
		  interval_1 = interval;
		}
      }
      utils::Check(len_0 >= 0 && len_1 >= 0, 
				"MatchLayer: length error negative. len_0=%d, len_1=%d.", len_0, len_1);
      utils::Check(len_0 <= doc0_len && len_1 <= doc1_len, 
				"MatchLayer: length error large. len_0=%d, len_1=%d, max=%d, max=%d.", len_0, len_1, doc0_len, doc1_len);

      // for (int j = 0; j < len_0; j++) {
      //   for (int k = 0; k < len_1; k++) {
      for (int j = 0; j < len_0; j+=interval_0) {
        for (int k = 0; k < len_1; k+=interval_1) {
          if (op == "xor") {
            utils::Check(bottom0_data2[i][j]!=-1 && bottom1_data2[i][k]!=-1, 
              "In Match Layer: please check length setting. (%d, %d, %d)", i, j, k);
            top_data[i][0][j][k] = (bottom0_data2[i][j] == bottom1_data2[i][k]) ? 1 : 0;
          } else if (op == "mul") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][0][j][k] += bottom0_data4[i][0][j][m] * bottom1_data4[i][0][k][m];
            }
          } else if (op == "plus") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][0][j][k] += bottom0_data4[i][0][j][m] + bottom1_data4[i][0][k][m];
            }
          } else if (op == "minus") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][0][j][k] += bottom0_data4[i][0][j][m] - bottom1_data4[i][0][k][m];
            }
          } else if (op == "cos") {
            for (int m = 0; m < feat_size; ++m) {
              m_dot[i][j][k] += bottom0_data4[i][0][j][m] * bottom1_data4[i][0][k][m];
            }
            top_data[i][0][j][k] = m_dot[i][j][k] / (m_norm[i][0][j] * m_norm[i][1][k]);    
          } else if (op == "order") {
            float sum_elem_square = 0.f;
            for (int m = 0; m < feat_size; ++m) {
              float sub_elem = bottom0_data4[i][0][j][m] - bottom1_data4[i][0][k][m];
              if (sub_elem <= 0) continue;
              sum_elem_square += sub_elem * sub_elem;
            }
            top_data[i][0][j][k] = sum_elem_square;
          } else if (op =="elemwise_product") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][m][j][k] = bottom0_data4[i][0][j][m] * bottom1_data4[i][0][k][m];
            }
          } else if (op =="elemwise_plus") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][m][j][k] = bottom0_data4[i][0][j][m] + bottom1_data4[i][0][k][m];
            }
          } else if (op =="elemwise_cat") {
            for (int m = 0; m < feat0_size; ++m) {
              top_data[i][m][j][k] = bottom0_data4[i][0][j][m];
            }
            for (int m = 0; m < feat1_size; ++m) {
              top_data[i][m + feat0_size][j][k] = bottom1_data4[i][0][k][m];
            }
          } else if (op =="euc") {
            float sum_elem_square = 0.f;
            for (int m = 0; m < feat_size; ++m) {
              float sub_elem = bottom0_data4[i][0][j][m] - bottom1_data4[i][0][k][m];
              sum_elem_square += sub_elem * sub_elem;
            }
            // top_data[i][0][j][k] = pow(sum_elem_square, 0.5);
            top_data[i][0][j][k] = sum_elem_square;
          } else if (op =="euc_exp") { // by wengpeng ying, no sqrt
            float sum_elem_square = 0.f;
            for (int m = 0; m < feat_size; ++m) {
              float sub_elem = bottom0_data4[i][0][j][m] - bottom1_data4[i][0][k][m];
              sum_elem_square += sub_elem * sub_elem;
            }
            // top_data[i][0][j][k] = exp(-(sum_elem_square)/(2*2.f)); // beta is set to 2.f
            top_data[i][0][j][k] = -(sum_elem_square)/(2*2.f); // beta is set to 2.f
          }  else {
            utils::Error("In Match Layer: no op named %s.\n", op.c_str());
          }
        }
      }
    }
    if (op == "euc_exp") {
      top_data = F<op::exp_lookup>(top_data);
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> bottom0_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom1_data = bottom[1]->data;
    mshadow::Tensor<xpu, 4> bottom0_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> bottom1_diff = bottom[1]->diff;
    mshadow::Tensor<xpu, 1> bottom0_len = bottom[0]->length_d1();
    mshadow::Tensor<xpu, 1> bottom1_len = bottom[1]->length_d1();

	int interval_0 = 1;
	int interval_1 = 1;

    if (op == "xor") {
      // do nothing
      return;
    }
      
    if (!this->prop_error[0] && !this->prop_error[1]) return;

    for (int i = 0; i < nbatch; ++i) {
      int len_0 = -1, len_1 = -1;
      if (is_var_len) {
        len_0 = bottom0_len[i];
        len_1 = bottom1_len[i];
		if (max_element != 0) {
		  interval_0 = len_0 / max_element + 1;
		  interval_1 = len_1 / max_element + 1;
		} else if (interval != 1) {
		  interval_0 = interval;
		  interval_1 = interval;
		}
      } else {
        len_0 = doc0_len;
        len_1 = doc1_len;
		if (max_element != 0) {
		  interval_0 = len_0 / max_element + 1;
		  interval_1 = len_1 / max_element + 1;
		} else if (interval != 1) {
		  interval_0 = interval;
		  interval_1 = interval;
		}
      }
      utils::Check(len_0 >= 0 && len_1 >= 0, 
				"MatchLayer: length error negative. len_0=%d, len_1=%d.", len_0, len_1);
      utils::Check(len_0 <= doc0_len && len_1 <= doc1_len, 
				"MatchLayer: length error large. len_0=%d, len_1=%d, max=%d, max=%d.", len_0, len_1, doc0_len, doc1_len);

      // for (int j = 0; j < len_0; ++j) {
      //   for (int k = 0; k < len_1; ++k) {
      for (int j = 0; j < len_0; j+=interval_0) {
        for (int k = 0; k < len_1; k+=interval_1) {
          for (int m = 0; m < feat_size; ++m) {
            if (op == "mul") {  
              if (this->prop_error[0])
                bottom0_diff[i][0][j][m] += bottom1_data[i][0][k][m] * top_diff[i][0][j][k];
              if (this->prop_error[1])
                bottom1_diff[i][0][k][m] += bottom0_data[i][0][j][m] * top_diff[i][0][j][k];
            } else if (op == "plus") {
              if (this->prop_error[0])
                bottom0_diff[i][0][j][m] += top_diff[i][0][j][k];
              if (this->prop_error[1])
                bottom1_diff[i][0][k][m] += top_diff[i][0][j][k];
            } else if (op == "minus") {
              if (this->prop_error[0])
                bottom0_diff[i][0][j][m] += top_diff[i][0][j][k];
              if (this->prop_error[1])
                bottom1_diff[i][0][k][m] -= top_diff[i][0][j][k];
            } else if (op == "cos") {
              if (this->prop_error[0])
                bottom0_diff[i][0][j][m] += (bottom1_data[i][0][k][m] / (m_norm[i][0][j] * m_norm[i][1][k]) 
                                             - bottom0_data[i][0][j][m] * m_dot[i][j][k] / (pow_3_m_norm[i][0][j] * m_norm[i][1][k]))
                                            * top_diff[i][0][j][k];
              if (this->prop_error[1])
                bottom1_diff[i][0][k][m] += (bottom0_data[i][0][j][m] / (m_norm[i][0][j] * m_norm[i][1][k]) 
                                             - bottom1_data[i][0][k][m] * m_dot[i][j][k] / (m_norm[i][0][j] * pow_3_m_norm[i][1][k]))
                                            * top_diff[i][0][j][k];
            } else if (op == "elemwise_product") {
              if (this->prop_error[0])
                bottom0_diff[i][0][j][m] += bottom1_data[i][0][k][m] * top_diff[i][m][j][k];
              if (this->prop_error[1])
                bottom1_diff[i][0][k][m] += bottom0_data[i][0][j][m] * top_diff[i][m][j][k];
            } else if (op == "elemwise_plus") {
              if (this->prop_error[0])
                bottom0_diff[i][0][j][m] += top_diff[i][m][j][k];
              if (this->prop_error[1])
                bottom1_diff[i][0][k][m] += top_diff[i][m][j][k];
            } else if (op =="elemwise_cat") {
              if (m < feat0_size) {
                bottom0_diff[i][0][j][m] += top_diff[i][m][j][k];
              } else {
                bottom1_diff[i][0][k][m - feat0_size] += top_diff[i][m][j][k];
              }
            } else if (op == "euc") {
              float distance = top_data[i][0][j][k];
              float sub_elem = bottom0_data[i][0][j][m] - bottom1_data[i][0][k][m];
              if (this->prop_error[0]) {
                // bottom0_diff[i][0][j][m] += top_diff[i][0][j][k] * (1/(2*distance)) * 2*(sub_elem);
                bottom0_diff[i][0][j][m] += top_diff[i][0][j][k] * 2*(sub_elem);
              }
              if (this->prop_error[1]) {
                // bottom1_diff[i][0][k][m] += top_diff[i][0][j][k] * (1/(2*distance)) * 2*(-sub_elem);
                bottom1_diff[i][0][k][m] += top_diff[i][0][j][k] * 2*(-sub_elem);
              }
            } else if (op == "order") {
              float sub_elem = bottom0_data[i][0][j][m] - bottom1_data[i][0][k][m];
              if (sub_elem <= 0) continue;
              if (this->prop_error[0]) {
                bottom0_diff[i][0][j][m] += top_diff[i][0][j][k] * 2*(sub_elem);
              }
              if (this->prop_error[1]) {
                bottom1_diff[i][0][k][m] += top_diff[i][0][j][k] * 2*(-sub_elem);
              }
            } else if (op == "euc_exp") {
              float distance = top_data[i][0][j][k];
              float sub_elem = bottom0_data[i][0][j][m] - bottom1_data[i][0][k][m];
              if (this->prop_error[0]) {
                bottom0_diff[i][0][j][m] += top_diff[i][0][j][k] * distance * (-1/(2*2.f)) * 2*(sub_elem);
              }
              if (this->prop_error[1]) {
                bottom1_diff[i][0][k][m] += top_diff[i][0][j][k] * distance * (-1/(2*2.f)) * 2*(-sub_elem);
              }
            }
          }
        }
      }
    }
  }
  
 protected:
  int doc0_len;
  int doc1_len;
  int feat_size;
  int feat0_size;
  int feat1_size;
  int nbatch;
  int interval;
  int max_element;
  bool is_var_len;
  std::string op;
  mshadow::TensorContainer<xpu, 3> m_norm, pow_3_m_norm;
  mshadow::TensorContainer<xpu, 3> m_dot;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MATCH_LAYER_INL_HPP_

