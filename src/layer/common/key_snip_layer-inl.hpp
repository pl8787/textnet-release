#ifndef TEXTNET_LAYER_KEY_SNIP_LAYER_INL_HPP_
#define TEXTNET_LAYER_KEY_SNIP_LAYER_INL_HPP_

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
class KeySnipLayer : public Layer<xpu>{
 public:
  KeySnipLayer(LayerType type) { this->layer_type = type; }
  virtual ~KeySnipLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 2; }
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
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["max_snip"] = SettingV();
    this->defaults["snip_size"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), 
        "KeySnipLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), 
        "KeySnipLayer:top size problem.");

    max_snip = setting["max_snip"].iVal();
    snip_size = setting["snip_size"].iVal();

  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), 
        "KeySnipLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), 
        "KeySnipLayer:top size problem.");
                  
    batch_size = bottom[0]->data.size(0); 
    
    top[0]->Resize(batch_size, max_snip, 1, snip_size*2+1, batch_size, max_snip, true);
    top[1]->Resize(batch_size, max_snip, 1, 1, batch_size, max_snip, true);
    snip_center_.Resize(mshadow::Shape2(batch_size, max_snip)); 

    if (show_info) {
      bottom[0]->PrintShape("bottom0");
      bottom[1]->PrintShape("bottom1");
      top[0]->PrintShape("top0");
      top[1]->PrintShape("top1");
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
    mshadow::Tensor<xpu, 1> bottom0_len = bottom[0]->length_d1();
    mshadow::Tensor<xpu, 1> bottom1_len = bottom[1]->length_d1();
    mshadow::Tensor<xpu, 3> top0_data = top[0]->data_d3();
    mshadow::Tensor<xpu, 2> top1_data = top[1]->data_d2();
    mshadow::Tensor<xpu, 2> top0_len = top[0]->length;
    mshadow::Tensor<xpu, 2> top1_len = top[1]->length;

    top0_data = -1;
    top1_data = -1;

    top0_len = 0;
    top1_len = 0;

    for (int i = 0; i < batch_size; ++i) {
      int s = 0;
      int d_cnt = bottom0_len[i];
      int q_cnt = bottom1_len[i];

      for (int j = 0; j < q_cnt; ++j) {
        for (int k = 0; k < d_cnt; ++k) {
          if ( fabs(bottom0_data[i][k]-bottom1_data[i][j]) < 1e-5 ) {
            top1_data[i][s] = j;
            top1_len[i][s] = 1;
            snip_center_[i][s] = k;
            ++s;
            if (s >= max_snip) break;
          }
        }
        if (s >= max_snip) break;
      }
       
      // crop snipping
      for (int j = 0; j < s; ++j) {
        for (int k = snip_center_[i][j] - snip_size, pk = 0; 
             k <= snip_center_[i][j] + snip_size; ++k, ++pk) {
          if (k < 0 || k >= d_cnt) continue;
          top0_data[i][j][pk] = bottom0_data[i][k];
        }
        top0_len[i][j] = snip_size * 2 + 1;
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

  }
  
 protected:
  int batch_size;
  int snip_size;
  int max_snip;
  mshadow::TensorContainer<xpu, 2> snip_center_;

};
}  // namespace layer
}  // namespace textnet
#endif  

