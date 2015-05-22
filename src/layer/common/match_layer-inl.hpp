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
    this->defaults["is_var_len"] = SettingV(true); 
	// xor: can not bp
	// mul: can bp
	// plus: can bp
	// cos: can bp
    
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

	utils::Check(op=="xor" || op=="mul" || op=="plus" || op=="cos" || op=="elemwise_product", 
			"MatchLayer: one of xor, mul, plus or cos.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MatchLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MatchLayer:top size problem.");
                  
    nbatch = bottom[0]->data.size(0); 
	if (op == "xor") {
      doc_len = bottom[0]->data.size(3);
	} else {
      doc_len = bottom[0]->data.size(2);
	  feat_size = bottom[0]->data.size(3);
	}		
                  
    if (op == "elemwise_product") {
      top[0]->Resize(nbatch, feat_size, doc_len, doc_len, true);
    } else {
      top[0]->Resize(nbatch, 1, doc_len, doc_len, true);
    }
    bottom[0]->PrintShape("bottom0");
    bottom[1]->PrintShape("bottom1");
    top[0]->PrintShape("top0");

	if (op == "cos") {
      m_norm.Resize(mshadow::Shape3(nbatch, 2, doc_len), 0.f);
	  m_dot.Resize(mshadow::Shape3(nbatch, doc_len, doc_len), 0.f);
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

	top_data = 0.0f;
	m_norm = 0.0f;
	m_dot = 0.0f;
    
    if (op == "cos") {
      for (int i = 0; i < nbatch; i++) {
        int len_0 = -1, len_1 = -1;
        if (is_var_len) {
          len_0 = bottom0_len[i];
          len_1 = bottom1_len[i];
        } else {
          len_0 = doc_len;
          len_1 = doc_len;
        }
        utils::Check(len_0 > 0 && len_1 > 0, "MatchLayer: length error.");
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
	}

	for (int i = 0; i < nbatch; i++) {
      int len_0 = -1, len_1 = -1;
      if (is_var_len) {
        len_0 = bottom0_len[i];
        len_1 = bottom1_len[i];
      } else {
        len_0 = doc_len;
        len_1 = doc_len;
      }
      utils::Check(len_0 > 0 && len_1 > 0, "MatchLayer: length error.");

      for (int j = 0; j < len_0; j++) {
        for (int k = 0; k < len_1; k++) {
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
		  } else if (op =="cos") {
			for (int m = 0; m < feat_size; ++m) {
              m_dot[i][j][k] += bottom0_data4[i][0][j][m] * bottom1_data4[i][0][k][m];
			}
		    top_data[i][0][j][k] = m_dot[i][j][k] / (m_norm[i][0][j] * m_norm[i][1][k]);	
		  } else if (op =="elemwise_product") {
            for (int m = 0; m < feat_size; ++m) {
              top_data[i][m][j][k] = bottom0_data4[i][0][j][m] + bottom1_data4[i][0][k][m];
			}
		  }  else {
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
	mshadow::Tensor<xpu, 4> bottom0_data = bottom[0]->data;
	mshadow::Tensor<xpu, 4> bottom1_data = bottom[1]->data;
	mshadow::Tensor<xpu, 4> bottom0_diff = bottom[0]->diff;
	mshadow::Tensor<xpu, 4> bottom1_diff = bottom[1]->diff;
	mshadow::Tensor<xpu, 1> bottom0_len = bottom[0]->length_d1();
	mshadow::Tensor<xpu, 1> bottom1_len = bottom[1]->length_d1();

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
      } else {
        len_0 = doc_len;
        len_1 = doc_len;
      }
      utils::Check(len_0 > 0 && len_1 > 0, "MatchLayer: length error.");

      for (int j = 0; j < len_0; ++j) {
	    for (int k = 0; k < len_1; ++k) {
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
			} else if (op == "cos") {
              if (this->prop_error[0])
				bottom0_diff[i][0][j][m] += (bottom1_data[i][0][k][m] / (m_norm[i][0][j] * m_norm[i][1][k]) 
						                     - bottom0_data[i][0][j][m] * m_dot[i][j][k] / (pow(m_norm[i][0][j], 3) * m_norm[i][1][k]))
											* top_diff[i][0][j][k];
			  if (this->prop_error[1])
			    bottom1_diff[i][0][k][m] += (bottom0_data[i][0][j][m] / (m_norm[i][0][j] * m_norm[i][1][k]) 
						                     - bottom1_data[i][0][k][m] * m_dot[i][j][k] / (m_norm[i][0][j] * pow(m_norm[i][1][k], 3)))
											* top_diff[i][0][j][k];
		    } else if (op == "elemwise_product") {
			  if (this->prop_error[0])
                bottom0_diff[i][0][j][m] += bottom1_data[i][0][k][m] * top_diff[i][m][j][k];
			  if (this->prop_error[1])
				bottom1_diff[i][0][k][m] += bottom0_data[i][0][j][m] * top_diff[i][m][j][k];
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
  bool is_var_len;
  std::string op;
  mshadow::TensorContainer<xpu, 3> m_norm;
  mshadow::TensorContainer<xpu, 3> m_dot;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MATCH_LAYER_INL_HPP_

