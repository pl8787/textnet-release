#ifndef TEXTNET_LAYER_RESHAPE_LAYER_INL_HPP_
#define TEXTNET_LAYER_RESHAPE_LAYER_INL_HPP_

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
class ReshapeLayer : public Layer<xpu>{
 public:
  ReshapeLayer(LayerType type) { this->layer_type = type; }
  virtual ~ReshapeLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
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
    this->defaults["D0"] = SettingV(0);
    this->defaults["D1"] = SettingV(0);
    this->defaults["D2"] = SettingV(0);
    this->defaults["D3"] = SettingV(0);
    this->defaults["L0"] = SettingV(0);
    this->defaults["L1"] = SettingV(0);
    
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
        "ReshapeLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), 
        "ReshapeLayer:top size problem.");

    D0 = setting["D0"].iVal();
    D1 = setting["D1"].iVal();
    D2 = setting["D2"].iVal();
    D3 = setting["D3"].iVal();
    L0 = setting["L0"].iVal();
    L1 = setting["L1"].iVal();

  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), 
        "ReshapeLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), 
        "ReshapeLayer:top size problem.");
                  
    in_data_shape = bottom[0]->data.shape_;
    in_len_shape = bottom[0]->length.shape_;

    out_data_shape[0] = D0==0 ? in_data_shape[0] : D0;
    out_data_shape[1] = D1==0 ? in_data_shape[1] : D1;
    out_data_shape[2] = D2==0 ? in_data_shape[2] : D2;
    out_data_shape[3] = D3==0 ? in_data_shape[3] : D3;
    out_len_shape[0] = L0==0 ? in_len_shape[0] : L0;
    out_len_shape[1] = L1==0 ? in_len_shape[1] : L1;

    if (D0 == -1) {
      out_data_shape[0] = 1;
      out_data_shape[0] = in_data_shape.Size() / out_data_shape.Size();
    } else if (D1 == -1) {
      out_data_shape[1] = 1;
      out_data_shape[1] = in_data_shape.Size() / out_data_shape.Size();
    } else if (D2 == -1) {
      out_data_shape[2] = 1;
      out_data_shape[2] = in_data_shape.Size() / out_data_shape.Size();
    } else if (D3 == -1) {
      out_data_shape[3] = 1;
      out_data_shape[3] = in_data_shape.Size() / out_data_shape.Size();
    }

    if (L0 == -1) {
      out_len_shape[0] = 1;
      out_len_shape[0] = in_len_shape.Size() / out_len_shape.Size();
    } else if (L1 == -1) {
      out_len_shape[1] = 1;
      out_len_shape[1] = in_len_shape.Size() / out_len_shape.Size();
    }

    utils::Check(out_data_shape.Size() == in_data_shape.Size(), 
        "ReshapeLayer: data shape mismatch.");
    utils::Check(out_len_shape.Size() == in_len_shape.Size(), 
        "ReshapeLayer: length shape mismatch.");

    top[0]->Resize(out_data_shape, out_len_shape, true);

    if (show_info) {
      bottom[0]->PrintShape("bottom0");
      top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (in_data_shape.Size() != bottom[0]->data.shape_.Size()) {
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
    mshadow::Tensor<xpu, 1> bottom0_data = bottom[0]->data_d1();
    mshadow::Tensor<xpu, 1> bottom0_len = bottom[0]->length_d1();
    mshadow::Tensor<xpu, 1> top0_data = top[0]->data_d1();
    mshadow::Tensor<xpu, 1> top0_len = top[0]->length_d1();
    
    top0_data = F<op::identity>(bottom0_data);
    top0_len = F<op::identity>(bottom0_len);

  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> bottom0_diff = bottom[0]->diff_d1();
    mshadow::Tensor<xpu, 1> top0_diff = top[0]->diff_d1();

    bottom0_diff = F<op::identity>(top0_diff);
  }
  
 protected:
    int D0;
    int D1;
    int D2;
    int D3;
    int L0;
    int L1;
    mshadow::Shape<4> in_data_shape;
    mshadow::Shape<2> in_len_shape;
    mshadow::Shape<4> out_data_shape;
    mshadow::Shape<2> out_len_shape;

};
}  // namespace layer
}  // namespace textnet
#endif  

