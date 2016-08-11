#ifndef TEXTNET_LAYER_BATCH_COMBINE_LAYER_INL_HPP_
#define TEXTNET_LAYER_BATCH_COMBINE_LAYER_INL_HPP_

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
class BatchCombineLayer : public Layer<xpu>{
 public:
  BatchCombineLayer(LayerType type) { this->layer_type = type; }
  virtual ~BatchCombineLayer(void) {}
  
  virtual int BottomNodeNum() { return nbottom; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["element"] = SettingV(false);
	this->defaults["nbottom"] = SettingV(2);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["candids"] = SettingV();
    this->defaults["op"] = SettingV(); 
    // mul: can bp
    // plus: can bp
    // cos: can bp
    // euc: can bp
    // euc_exp: can bp, see wenpeng's paper
    // cat: can bp
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
	nbottom = setting["nbottom"].iVal(); // pay attention!!! nbottom should be set before SetupLayer!
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    op = setting["op"].sVal();
    candids = setting["candids"].iVal();
    element = setting["element"].bVal();

    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchCombineLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchCombineLayer:top size problem.");

	if (nbottom == 1) {
	  utils::Check(op=="ind", 
			     "BatchCombineLayer: bottom num equal 1, only ind op support.");
	} else {
      if (element)
        utils::Check(op=="mul" || op=="plus" || op=="minus",
                 "BatchCombineLayer: only mul, plus, minus support element.");
      else
        utils::Check(op=="cat" || op=="mul" || op=="plus" || op=="cos" || op == "minus" ||\
                 op=="euc" || op=="euc_neg" || op=="euc_exp",
                 "BatchCombineLayer: one of cat, mul, plus, cos, minus, euc, euc_neg or euc_exp.");
	}
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchCombineLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchCombineLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
    utils::Check(nbatch % candids == 0, 
                  "BatchCombineLayer: nbatch div candids.");
    nbatch = nbatch / candids;

    feat_size = bottom[0]->data.size(1) * bottom[0]->data.size(2) * bottom[0]->data.size(3);
                  
    if (op == "cat") {
      top[0]->Resize(nbatch, 2, candids, feat_size, true);
	} else if (op == "ind") {
      top[0]->Resize(nbatch, 1, candids, feat_size, true);
    } else if (element) {
      top[0]->Resize(nbatch, 1, candids, feat_size, true);
    } else {
      top[0]->Resize(nbatch, 1, 1, candids, true);
    }

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
		if (nbottom == 2)
		  bottom[1]->PrintShape("bottom1");
        top[0]->PrintShape("top0");
    }

    if (op == "cos") {
      m_norm.Resize(mshadow::Shape3(nbatch, 2, candids), 0.f);
      m_dot.Resize(mshadow::Shape2(nbatch, candids), 0.f);
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (nbatch != bottom[0]->data.size(0) / candids) {
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
    mshadow::Tensor<xpu, 4> bottom0_data4 = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;

	if (op == "ind") {
	  for (int i = 0; i < nbatch; ++i) {
		for (int c = 0; c < candids; ++c) {
		  int j = i * candids + c;
		  for (int m = 0; m < feat_size; ++m) {
			top_data[i][0][c][m] = bottom0_data2[j][m];
		  }
		}
	  }
	  // if nbottm == 1 just change the shape of the node
	  return;
	}

    mshadow::Tensor<xpu, 2> bottom1_data2 = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 4> bottom1_data4 = bottom[1]->data;

    top_data = 0.0f;
    m_norm = 0.0f;
    m_dot = 0.0f;
    
    if (op == "cos") {
      for (int i = 0; i < nbatch; ++i) {
        for (int c = 0; c < candids; ++c) {
          int j = i * candids + c;
          for (int m = 0; m < feat_size; ++m) {
            m_norm[i][0][c] += bottom0_data2[j][m] * bottom0_data2[j][m];
            m_norm[i][1][c] += bottom1_data2[j][m] * bottom1_data2[j][m];
          }
        }
      }
      m_norm = F<op::square_root >(m_norm);
    }

    if (op == "cat") {
      for (int i = 0; i < nbatch; ++i) {
        for (int c = 0; c < candids; ++c) {
          int j = i * candids + c;
          for (int m = 0; m < feat_size; ++m) {
            top_data[i][0][c][m] = bottom0_data2[j][m];
            top_data[i][1][c][m] = bottom1_data2[j][m];
          }
        }
      }
    } else if (element) {
      for (int i = 0; i < nbatch; ++i) {
        for (int c = 0; c < candids; ++c) {
          int j = i * candids + c;
          for (int m = 0; m < feat_size; ++m) {
            if (op == "plus") {
              top_data[i][0][c][m] = bottom0_data2[j][m] + bottom1_data2[j][m];
            } else if (op == "minus") {
              top_data[i][0][c][m] = bottom0_data2[j][m] - bottom1_data2[j][m];
            } else if (op == "mul") {
              top_data[i][0][c][m] = bottom0_data2[j][m] * bottom1_data2[j][m];
            }
          }
        }
      }
    } else { // element == false
      for (int i = 0; i < nbatch; i++) {
        for (int c = 0; c < candids; ++c) {
          int j = i * candids + c;
          if (op == "plus") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][0][0][c] += bottom0_data2[j][m] + bottom1_data2[j][m];
            }
          }
          else if (op == "minus") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][0][0][c] += bottom0_data2[j][m] - bottom1_data2[j][m];
            }
          }
          else if (op == "mul") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][0][0][c] += bottom0_data2[j][m] * bottom1_data2[j][m];
            }
          }
          else if (op == "cos") {
            for (int m = 0; m < feat_size; ++m) {
              m_dot[i][c] += bottom0_data2[j][m] * bottom1_data2[j][m];
            }
            top_data[i][0][0][c] = m_dot[i][c] / (m_norm[i][0][c] * m_norm[i][1][c]);    
          } 
          else if (op =="euc") {
            float sum_elem_square = 0.f;
            for (int m = 0; m < feat_size; ++m) {
              float sub_elem = bottom0_data2[j][m] - bottom1_data2[j][m];
              sum_elem_square += sub_elem * sub_elem;
            }
            // top_data[i][0][j][k] = pow(sum_elem_square, 0.5);
            top_data[i][0][0][c] = sum_elem_square;
          } 
          else if (op =="euc_neg") {
            float sum_elem_square = 0.f;
            for (int m = 0; m < feat_size; ++m) {
              float sub_elem = bottom0_data2[j][m] - bottom1_data2[j][m];
              sum_elem_square += sub_elem * sub_elem;
            }
            // top_data[i][0][j][k] = pow(sum_elem_square, 0.5);
            top_data[i][0][0][c] = -sum_elem_square;
          } 
          else if (op =="euc_exp") { // by wengpeng ying, no sqrt
            float sum_elem_square = 0.f;
            for (int m = 0; m < feat_size; ++m) {
              float sub_elem = bottom0_data2[j][m] - bottom1_data2[j][m];
              sum_elem_square += sub_elem * sub_elem;
            }
            top_data[i][0][0][c] = exp(-(sum_elem_square)/(2*2.f)); // beta is set to 2.f
          }  
          else {
            utils::Error("In Match Layer: no op named %s.\n", op.c_str());
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
    mshadow::Tensor<xpu, 2> bottom0_data2 = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> bottom0_diff2 = bottom[0]->diff_d2();

	if (op == "ind") {
      if (!this->prop_error[0]) return;
      for (int i = 0; i < nbatch; ++i) {
		for (int c = 0; c < candids; ++c) {
		  int j = i * candids + c;
		  for (int m = 0; m < feat_size; ++m) {
			bottom0_diff2[j][m] += top_diff[i][0][c][m];
		  }
		}
	  }
	  return;
	}

    if (!this->prop_error[0] && !this->prop_error[1]) return;

    mshadow::Tensor<xpu, 2> bottom1_data2 = bottom[1]->data_d2();
    mshadow::Tensor<xpu, 2> bottom1_diff2 = bottom[1]->diff_d2();

    for (int i = 0; i < nbatch; ++i) {
      for (int c = 0; c < candids; ++c) {
        int j = i * candids + c;
        for (int m = 0; m < feat_size; ++m) {
          if (op == "cat") {
            if (this->prop_error[0])
              bottom0_diff2[j][m] += top_diff[i][0][c][m];
            if (this->prop_error[1])
              bottom1_diff2[j][m] += top_diff[i][1][c][m];
          } else if (element) {
            if (op == "plus") {
              if (this->prop_error[0])
                bottom0_diff2[j][m] += top_diff[i][0][c][m];
              if (this->prop_error[1])
                bottom1_diff2[j][m] += top_diff[i][0][c][m];
            } else if (op == "minus") {
              if (this->prop_error[0])
                bottom0_diff2[j][m] += top_diff[i][0][c][m];
              if (this->prop_error[1])
                bottom1_diff2[j][m] -= top_diff[i][0][c][m];
            } else if (op == "mul") {  
              if (this->prop_error[0])
                bottom0_diff2[j][m] += bottom1_data2[j][m] * top_diff[i][0][c][m];
              if (this->prop_error[1])
                bottom1_diff2[j][m] += bottom0_data2[j][m] * top_diff[i][0][c][m];
            } 
          } else { // element == false
            if (op == "plus") {
              if (this->prop_error[0])
                bottom0_diff2[j][m] += top_diff[i][0][0][c];
              if (this->prop_error[1])
                bottom1_diff2[j][m] += top_diff[i][0][0][c];
            } else if (op == "minus") {
              if (this->prop_error[0])
                bottom0_diff2[j][m] += top_diff[i][0][0][c];
              if (this->prop_error[1])
                bottom1_diff2[j][m] -= top_diff[i][0][0][c];
            } else if (op == "mul") {
              if (this->prop_error[0])
                bottom0_diff2[j][m] += bottom1_data2[j][m] * top_diff[i][0][0][c];
              if (this->prop_error[1])
                bottom1_diff2[j][m] += bottom0_data2[j][m] * top_diff[i][0][0][c];
            } else if (op == "cos") {
              if (this->prop_error[0])
                bottom0_diff2[j][m] += (bottom1_data2[j][m] / (m_norm[i][0][c] * m_norm[i][1][c]) 
                                             - bottom0_data2[j][m] * m_dot[i][c] / (pow(m_norm[i][0][c], 3) * m_norm[i][1][c]))
                                            * top_diff[i][0][0][c];
              if (this->prop_error[1])
                bottom1_diff2[j][m] += (bottom0_data2[j][m] / (m_norm[i][0][c] * m_norm[i][1][c]) 
                                             - bottom1_data2[j][m] * m_dot[i][c] / (m_norm[i][0][c] * pow(m_norm[i][1][c], 3)))
                                            * top_diff[i][0][0][c];
            } else if (op == "euc") {
              float distance = top_data[i][0][0][c];
              float sub_elem = bottom0_data2[j][m] - bottom1_data2[j][m];
              if (this->prop_error[0]) {
                // bottom0_diff[i][0][j][m] += top_diff[i][0][j][k] * (1/(2*distance)) * 2*(sub_elem);
                bottom0_diff2[j][m] += top_diff[i][0][0][c] * 2*(sub_elem);
              }
              if (this->prop_error[1]) {
                // bottom1_diff[i][0][k][m] += top_diff[i][0][j][k] * (1/(2*distance)) * 2*(-sub_elem);
                bottom1_diff2[j][m] += top_diff[i][0][0][c] * 2*(-sub_elem);
              }
            } else if (op == "euc_neg") {
              float distance = top_data[i][0][0][c];
              float sub_elem = bottom0_data2[j][m] - bottom1_data2[j][m];
              if (this->prop_error[0]) {
                // bottom0_diff[i][0][j][m] += top_diff[i][0][j][k] * (1/(2*distance)) * 2*(sub_elem);
                bottom0_diff2[j][m] += top_diff[i][0][0][c] * 2*(-sub_elem);
              }
              if (this->prop_error[1]) {
                // bottom1_diff[i][0][k][m] += top_diff[i][0][j][k] * (1/(2*distance)) * 2*(-sub_elem);
                bottom1_diff2[j][m] += top_diff[i][0][0][c] * 2*(sub_elem);
              }
            } else if (op == "euc_exp") {
              float distance = top_data[i][0][0][c];
              float sub_elem = bottom0_data2[j][m] - bottom1_data2[j][m];
              if (this->prop_error[0]) {
                bottom0_diff2[j][m] += top_diff[i][0][0][c] * distance * (-1/(2*2.f)) * 2*(sub_elem);
              }
              if (this->prop_error[1]) {
                bottom1_diff2[j][m] += top_diff[i][0][0][c] * distance * (-1/(2*2.f)) * 2*(-sub_elem);
              }
            }
          }
        }
      }
    }
  }
  
 protected:
  int doc_len;
  int feat_size;
  int nbatch;
  int candids;
  bool element;
  int nbottom;
  std::string op;
  mshadow::TensorContainer<xpu, 3> m_norm;
  mshadow::TensorContainer<xpu, 2> m_dot;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_BATCH_COMBINE_LAYER_INL_HPP_

