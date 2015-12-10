#ifndef TEXTNET_LAYER_GAUSSIAN_MASK_LAYER_INL_HPP_
#define TEXTNET_LAYER_GAUSSIAN_MASK_LAYER_INL_HPP_

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
class GaussianMaskLayer : public Layer<xpu>{
 public:
  GaussianMaskLayer(LayerType type) { this->layer_type = type; }
  virtual ~GaussianMaskLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
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

  virtual void Require() {
    // default value, just set the value you want
    this->defaults["is_norm"] = SettingV(true);
	this->defaults["is_symmetric"] = SettingV(true);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
	this->defaults["channel"] = SettingV();
    this->defaults["dim"] = SettingV();
    this->defaults["n_size"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), 
		"GaussianMaskLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), 
		"GaussianMaskLayer:top size problem.");

    is_norm = setting["is_norm"].bVal();
	is_symmetric = setting["is_symmetric"].bVal();
    dim = setting["dim"].iVal();
	channel = setting["channel"].iVal();
    std::string n_size_str = setting["n_size"].sVal();

	utils::Check(dim == 1 || dim == 2, 
		"GaussianMaskLayer: dim now only support 1 and 2.");

	utils::Check(bottom[0]->data.size(1) * bottom[0]->data.size(2) == channel * dim,
		"GaussianMaskLayer: Mean input %d must equal to channel %d and dim %d.", 
		bottom[0]->data.size(1) * bottom[0]->data.size(2), channel, dim);
    if (is_symmetric) {
	  utils::Check(bottom[1]->data.size(1) * bottom[1]->data.size(2) * bottom[1]->data.size(3) == channel*(dim+1)*dim/2,
		"GaussianMaskLayer: Var input %d must compatible with channel %d and dim %d.", 
		bottom[1]->data.size(1) * bottom[1]->data.size(2) * bottom[1]->data.size(3), channel, dim);
	} else {
	  utils::Check(bottom[1]->data.size(1) * bottom[1]->data.size(2) * bottom[1]->data.size(3) == channel*dim*dim,
		"GaussianMaskLayer: Var input %d must compatible with channel %d and dim %d.", 
		bottom[1]->data.size(1) * bottom[1]->data.size(2) * bottom[1]->data.size(3), channel, dim);
	}

    int s = 0;
	istringstream iss;
	iss.str(n_size_str);
	for (int i = 0; i < dim; ++i) {
      iss >> s;
	  n_size.push_back(s);
	}
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), 
		"GaussianMaskLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), 
		"GaussianMaskLayer:top size problem.");
                  
    batch_size = bottom[0]->data.size(0); 
    
	if (dim == 1) {
      top[0]->Resize(batch_size, channel, 1, n_size[0], batch_size, 1, true);
	  position_.Resize(mshadow::Shape2(n_size[0], 1));
	  centered_.Resize(mshadow::Shape2(n_size[0], 1));
	  for (int i = 0; i < n_size[0]; ++i) {
        position_[i][0] = 1.0 * i / n_size[0];
	  }
	  a_size = n_size[0];
	} else if (dim == 2) {
      top[0]->Resize(batch_size, channel, n_size[0], n_size[1], batch_size, 2, true);
	  position_.Resize(mshadow::Shape2(n_size[0] * n_size[1], 2));
	  centered_.Resize(mshadow::Shape2(n_size[0] * n_size[1], 2));
	  int p = 0;
	  for (int i = 0; i < n_size[0]; ++i) {
		for (int j = 0; j < n_size[1]; ++j) {
		  position_[p][0] = 1.0 * i / n_size[0];
		  position_[p][1] = 1.0 * j / n_size[1];
		  ++p;
		}
	  }
	  a_size = n_size[0] * n_size[1];
	} else {
	  utils::Check(dim == 1 || dim == 2, 
		"GaussianMaskLayer: dim now only support 1 and 2.");  
	}
	var_.Resize(mshadow::Shape2(dim, dim));
	var_diff_.Resize(mshadow::Shape2(dim, dim));

	PrintTensor("position", position_);

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
    if (batch_size != bottom[0]->data.size(0)) {
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
	mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
	mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2();
	mshadow::Tensor<xpu, 3> top_data = top[0]->data_d3();
	for (int i = 0; i < batch_size; ++i) {
	  for (int c = 0; c < channel; ++c) {
		int c_start = 0; 

	    if (dim == 1) {
          // Construct Var Matrix
		  c_start = c * dim;
		  var_[0][0] = bottom1_data[i][c_start];
          // Compute Gaussian Map
		  centered_ = position_ - bottom0_data[i][c];
		  for (int k = 0; k < a_size; ++k) {
            top_data[i][c][k] = exp(-centered_[k][0]*var_[0][0]*centered_[k][0]);
		  }
		} else if (dim == 2) {
          // Construct Var Matrix
	      if (is_symmetric) {
			c_start = c * (dim+1)*dim/2;
			var_[0][0] = bottom1_data[i][c_start];
			var_[0][1] = bottom1_data[i][c_start+1];
			var_[1][0] = bottom1_data[i][c_start+1];
			var_[1][1] = bottom1_data[i][c_start+2];
	      } else {
			c_start = c * dim*dim;
			var_[0][0] = bottom1_data[i][c_start];
			var_[0][1] = bottom1_data[i][c_start+1];
			var_[1][0] = bottom1_data[i][c_start+2];
			var_[1][1] = bottom1_data[i][c_start+3];
	      }
          // Compute Gaussian Map
		  for (int k = 0; k < a_size; ++k) {
		    centered_[k][0] = position_[k][0] - bottom0_data[i][2*c];
		    centered_[k][1] = position_[k][1] - bottom0_data[i][2*c+1];
            top_data[i][c][k] = exp(-var_[0][0]*centered_[k][0]*centered_[k][0]
					                -(var_[1][0]+var_[0][1])*centered_[k][0]*centered_[k][1]
									-var_[1][1]*centered_[k][1]*centered_[k][1]);
		  }
		}
	  }
	}
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
	mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
	mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2();
	mshadow::Tensor<xpu, 2> bottom0_diff = bottom[0]->diff_d2();
	mshadow::Tensor<xpu, 2> bottom1_diff = bottom[1]->diff_d2();
	mshadow::Tensor<xpu, 3> top_data = top[0]->data_d3();
	mshadow::Tensor<xpu, 3> top_diff = top[0]->diff_d3();

	top_diff *= top_data;

    if (dim == 1) {
	  for (int i = 0; i < batch_size; ++i) {
        for (int c = 0; c < channel; ++c) {
		  var_[0][0] = bottom1_data[i][c];
		  centered_ = position_ - bottom0_data[i][c];
	      if (this->prop_error[0]) {
			for (int k = 0; k < a_size; ++k) {
			  bottom0_diff[i][c] += 2 * var_[0][0] * centered_[k][0] * top_diff[i][c][k];
			}
	      }
	      if (this->prop_error[1]) {
			for (int k = 0; k < a_size; ++k) {
              bottom1_diff[i][c] += -centered_[k][0] * centered_[k][0] * top_diff[i][c][k];
			}
	      }
		}
	  }
    } else if (dim == 2) {
      for (int i = 0; i < batch_size; ++i) {
		for (int c = 0; c < channel; ++c) {
		  int c_start = 0; 
          // Construct Var Matrix
	      if (is_symmetric) {
			c_start = c * (dim+1)*dim/2;
			var_[0][0] = bottom1_data[i][c_start];
			var_[0][1] = bottom1_data[i][c_start+1];
			var_[1][0] = bottom1_data[i][c_start+1];
			var_[1][1] = bottom1_data[i][c_start+2];
	      } else {
			c_start = c * dim*dim;
			var_[0][0] = bottom1_data[i][c_start];
			var_[0][1] = bottom1_data[i][c_start+1];
			var_[1][0] = bottom1_data[i][c_start+2];
			var_[1][1] = bottom1_data[i][c_start+3];
	      }
		  for (int k = 0; k < a_size; ++k) {
			centered_[k][0] = position_[k][0] - bottom0_data[i][2*c];
			centered_[k][1] = position_[k][1] - bottom0_data[i][2*c+1];
            if (this->prop_error[0]) {
			  bottom0_diff[i][c*2] += (2 * var_[0][0] * centered_[k][0] + (var_[0][1]+var_[1][0]) * centered_[k][1]) * top_diff[i][c][k];
			  bottom0_diff[i][c*2+1] += (2 * var_[1][1] * centered_[k][1] + (var_[0][1]+var_[1][0]) * centered_[k][0]) * top_diff[i][c][k];
		    }
		    if (this->prop_error[1]) {
			  var_diff_ = dot(centered_.Slice(k,k+1).T(), centered_.Slice(k,k+1));
			  var_diff_ *= -top_diff[i][c][k];
	          if (is_symmetric) {
			    c_start = c * (dim+1)*dim/2;
			    bottom1_diff[i][c_start] += var_diff_[0][0];
			    bottom1_diff[i][c_start+1] += var_diff_[0][1] + var_diff_[1][0];
			    bottom1_diff[i][c_start+2] += var_diff_[1][1];
	          } else {
			    c_start = c * dim*dim;
			    bottom1_diff[i][c_start] += var_diff_[0][0];
			    bottom1_diff[i][c_start+1] += var_diff_[0][1];
			    bottom1_diff[i][c_start+2] += var_diff_[1][0];
			    bottom1_diff[i][c_start+3] += var_diff_[1][1];
	          }
		    }
		  }
		}
	  }
	}
  }
  
 protected:
  int batch_size;
  bool is_norm;
  bool is_symmetric;
  int dim;
  int channel;
  std::vector<int> n_size;
  int a_size;

  mshadow::TensorContainer<xpu, 2> position_;
  mshadow::TensorContainer<xpu, 2> centered_;
  mshadow::TensorContainer<xpu, 2> var_;
  mshadow::TensorContainer<xpu, 2> var_diff_;
};
}  // namespace layer
}  // namespace textnet
#endif  

