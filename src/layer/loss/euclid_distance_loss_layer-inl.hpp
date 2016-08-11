#ifndef TEXTNET_LAYER_EUCLID_DISTANCE_LOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_EUCLID_DISTANCE_LOSS_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

typedef float orc_real;

// this layer is originally designed for pretrainning char LSTM
// it can be also used for models such as stacked autoencoders
// ATTENTION: the distance is computed elem-wise
template<typename xpu>
class EuclidDistanceLossLayer : public Layer<xpu>{
 public:
  EuclidDistanceLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~EuclidDistanceLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; } // pred_rep, target_rep 
  virtual int TopNodeNum() { return 1; }    // loss
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["temperature"] = SettingV(); // tranditional LSTM constrains the representations to -1 to 1

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(), "Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "EuclidDistanceLossLayer:top size problem.");

    temperature = setting["temperature"].fVal();

    // bottom[0], pred_rep,   (batch_size, 1, 1, feat_size)
    // bottom[1], target_rep, (batch_size, 1, 1, feat_size)
    utils::Check(bottom[0]->data.size(0) == bottom[1]->data.size(0), "EuclidDistanceLossLayer: input error.");
    utils::Check(bottom[0]->data.size(1) == bottom[1]->data.size(1), "EuclidDistanceLossLayer: input error.");
    utils::Check(bottom[0]->data.size(2) == bottom[1]->data.size(2), "EuclidDistanceLossLayer: input error.");
    utils::Check(bottom[0]->data.size(3) == bottom[1]->data.size(3), "EuclidDistanceLossLayer: input error.");
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "EuclidDistanceLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "EuclidDistanceLossLayer:top size problem.");

    top[0]->Resize(1, 1, 1, 1, true);
	if (show_info) {
      top[0]->PrintShape("top0");
	}
  }
  void checkNan(orc_real *p, int l) {
    for (int i = 0; i < l; ++i) {
      assert(!std::isnan(p[i]));
    }
  }
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> pred_rep   = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> target_rep = bottom[1]->data_d2();

    if (temperature != 1.f) {
      pred_rep = pred_rep * temperature;
    }

    float loss = 0.f;
    for (int i = 0; i < pred_rep.size(0); ++i) {
      float l = 0.f;
      for (int j = 0; j < pred_rep.size(1); ++j) {
        float d = pred_rep[i][j] - target_rep[i][j];
        l += d * d; 
      }
      loss += l;
    }
    top[0]->data[0][0][0][0] = loss/float(pred_rep.shape_.Size());
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> pred_rep   = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 2> pred_diff  = bottom[0]->diff_d2();
    mshadow::Tensor<xpu, 2> target_rep = bottom[1]->data_d2();

    pred_diff = 0.f;
    for (int i = 0; i < pred_rep.size(0); ++i) {
      for (int j = 0; j < pred_rep.size(1); ++j) {
        float d = pred_rep[i][j] - target_rep[i][j];
        pred_diff[i][j] = 2 * d;
      }
    }
  }

protected:
  float temperature;
};
}  // namespace layer
}  // namespace textnet

#endif
