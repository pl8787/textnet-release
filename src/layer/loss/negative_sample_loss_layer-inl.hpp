#ifndef TEXTNET_LAYER_NEGATIVE_SAMPLE_LOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_NEGATIVE_SAMPLE_LOSS_LAYER_INL_HPP_

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
class NegativeSampleLossLayer : public Layer<xpu>{
 public:
  NegativeSampleLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~NegativeSampleLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 3; } // pred_rep, word_rep, label
  virtual int TopNodeNum() { return 2; } // softmax_prob and loss
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(), "Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "NegativeSampleLossLayer:top size problem.");

    // bottom[0], pred_rep, (batch_size, position_num, 1, feat_size)
    // bottom[1], word_rep, (batch_size, position_num, sample_num, feat_size+1)
    // bottom[2], label,    (batch_size, position_num, sample_num, 1)
    utils::Check(bottom[0]->data.size(0) == bottom[1]->data.size(0), "NegativeSampleLossLayer: input error.");
    utils::Check(bottom[0]->data.size(0) == bottom[2]->data.size(0), "NegativeSampleLossLayer: input error.");
    utils::Check(bottom[0]->data.size(1) == bottom[1]->data.size(1), "NegativeSampleLossLayer: input error.");
    utils::Check(bottom[0]->data.size(1) == bottom[2]->data.size(1), "NegativeSampleLossLayer: input error.");
    utils::Check(bottom[0]->data.size(3)+1 == bottom[1]->data.size(3), "NegativeSampleLossLayer: input error.");
    utils::Check(bottom[1]->data.size(2) == bottom[2]->data.size(2), "NegativeSampleLossLayer: input error.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "NegativeSampleLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "NegativeSampleLossLayer:top size problem.");
    // nbatch = bottom[0]->data.size(0);  
    int batch_size   = bottom[0]->data.size(0);
    int position_num = bottom[0]->data.size(1);
    int sample_num   = bottom[1]->data.size(2);
    top[0]->Resize(batch_size, position_num, sample_num, 2, true);
    top[1]->Resize(1, 1, 1, 1, true);
  }
  void checkNan(float *p, int l) {
      for (int i = 0; i < l; ++i) {
          assert(!isnan(p[i]));
      }
  }
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom0_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom1_data = bottom[1]->data;
    mshadow::Tensor<xpu, 4> bottom2_data = bottom[2]->data;
    mshadow::Tensor<xpu, 4> top0_data    = top[0]->data;
    mshadow::Tensor<xpu, 2> top0_data_d2 = top[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 1> top1_data    = top[1]->data_d1();

    top0_data = 0.f;
    top1_data = 0.f;

    for (int i = 0; i < bottom1_data.size(0); ++i) {       // example
      for (int j = 0; j < bottom1_data.size(1); ++j) {     // position
        for (int k = 0; k < bottom1_data.size(2); ++k) {   // sample
          for (int f = 0; f < bottom0_data.size(3); ++f) { // feature and bias
            top0_data[i][j][k][1] += bottom0_data[i][j][0][f] * bottom1_data[i][j][k][f]; 
          }
          top0_data[i][j][k][1] += bottom1_data[i][j][k][bottom1_data.size(3)-1]; // bias
        }
      }
    }

    mshadow::Softmax(top0_data_d2, top0_data_d2);
	
    // loss
    int sample_cnt = 0;
    for (int i = 0; i < bottom1_data.size(0); ++i) {
      for (int j = 0; j < bottom1_data.size(1); ++j) {
        for (int k = 0; k < bottom1_data.size(2); ++k) {
          sample_cnt += 1;
          int c = static_cast<int>(bottom2_data[i][j][k][0]);
          utils::Check(c == 0 || c == 1, "NegativeSampleLossLayer: label error");
          if (top0_data[i][j][k][c] == 0.) {
            top1_data[0] += 88; // by min float number
          } else { 
            top1_data[0] += -log(top0_data[i][j][k][c]);
          }
        }
      }
    }
    top1_data[0] /= sample_cnt;
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom0_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom1_data = bottom[1]->data;
    mshadow::Tensor<xpu, 4> bottom2_data = bottom[2]->data;
    mshadow::Tensor<xpu, 4> bottom0_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> bottom1_diff = bottom[1]->diff;
    mshadow::Tensor<xpu, 4> top0_data    = top[0]->data;
    mshadow::Tensor<xpu, 4> top0_diff    = top[0]->diff;

    top0_diff = F<op::identity>(top0_data);
    // if (this->prop_error[0]) {
      // bp loss and softmax
      for (int i = 0; i < top0_data.size(0); ++i) {
        for (int j = 0; j < top0_data.size(1); ++j) {
          for (int k = 0; k < top0_data.size(2); ++k) {
            int c = static_cast<int>(bottom2_data[i][j][k][0]);
            utils::Check(c == 0 || c == 1, "NegativeSampleLossLayer: label error");
            top0_diff[i][j][k][c] -= 1.0f;
          }
        }
      }
      // bp to score
      for (int i = 0; i < bottom1_data.size(0); ++i) {
        for (int j = 0; j < bottom1_data.size(1); ++j) {
          for (int k = 0; k < bottom1_data.size(2); ++k) {
            for (int f = 0; f < bottom0_data.size(3); ++f) {
              bottom0_diff[i][j][0][f] += top0_diff[i][j][k][1] * bottom1_data[i][j][k][f];
              bottom1_diff[i][j][k][f] += top0_diff[i][j][k][1] * bottom0_data[i][j][0][f];
            }
            bottom1_diff[i][j][k][bottom1_data.size(3)-1] += top0_diff[i][j][k][1];
          }
        }
      }
    // }
  }
 // protected:
};
}  // namespace layer
}  // namespace textnet
#endif
