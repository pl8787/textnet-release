#ifndef TEXTNET_LAYER_LSTM_AUTOENCODER_SOFTMAX_LOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_LSTM_AUTOENCODER_SOFTMAX_LOSS_LAYER_INL_HPP_

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

// this layer is originally designed to train LSTM autoencoder in (Jiwei Li, 2015)
// it also can be used to train a tranditional language models
template<typename xpu>
class LmSoftmaxLossLayer : public Layer<xpu>{
 public:
  LmSoftmaxLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~LmSoftmaxLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; } // pred_rep, label
  virtual int TopNodeNum() { return 2; }    // final_prob, loss
  virtual int ParamNodeNum() { return 2; }
  
  virtual void Require() {
    // default value, just set the value you want
    // this->defaults["word_embed_file"] = SettingV("");
    this->defaults["no_bias"] = SettingV(false);
    // this->defaults["w_file"] = SettingV("");
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["feat_size"] = SettingV();
    this->defaults["vocab_size"] = SettingV();
    this->defaults["temperature"] = SettingV();

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(), "Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LmSoftmaxLossLayer:top size problem.");

    feat_size = setting["feat_size"].iVal();
    temperature = setting["temperature"].fVal();
    vocab_size= setting["vocab_size"].iVal();
    no_bias = setting["no_bias"].bVal();
    // w_file = setting["w_file"].sVal();

    // bottom[0], pred_rep, (batch_size, position_num, 1, feat_size)
    // bottom[1], label,    (batch_size, position_num, 1, 1)
    utils::Check(bottom[0]->data.size(0) == bottom[1]->data.size(0), "LmSoftmaxLossLayer: input error.");
    utils::Check(bottom[0]->data.size(2) == bottom[1]->data.size(2), "LmSoftmaxLossLayer: input error 1.");
    utils::Check(bottom[0]->data.size(3) == feat_size, "LmSoftmaxLossLayer: input error 2.");
    utils::Check(bottom[0]->data.size(1) == 1, "LmSoftmaxLossLayer: input error 3.");
    utils::Check(bottom[1]->data.size(1) == 1, "LmSoftmaxLossLayer: input error 4.");
    utils::Check(bottom[1]->data.size(3) == 1, "LmSoftmaxLossLayer: input error 5.");

    this->params.resize(2); 
    this->params[0].Resize(1, 1, vocab_size, feat_size, true); // W
    this->params[1].Resize(1, 1, 1, vocab_size, true);         // b

    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
                                                                          w_setting, this->prnd_);
    this->params[1].initializer_ = initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(),
                                                                          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
                                                              w_updater, this->prnd_);
    this->params[1].updater_ = updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
                                                              b_updater, this->prnd_);
    // if (!w_file.empty()) {
    //    this->params[0].LoadDataSsv(w_file.c_str());
    // }
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "LmSoftmaxLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LmSoftmaxLossLayer:top size problem.");

    int batch_size   = bottom[0]->data.size(0);
    int position_num = bottom[0]->data.size(2);
    top[0]->Resize(batch_size, position_num, 1, vocab_size, true);
    top[1]->Resize(1, 1, 1, 1, true);
	if (show_info) {
      top[0]->PrintShape("top0");
      top[1]->PrintShape("top1");
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
    mshadow::Tensor<xpu, 2> pred_rep = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 1> label    = bottom[1]->data_d1_reverse();
    mshadow::Tensor<xpu, 2> prob     = top[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 1> loss     = top[1]->data_d1_reverse();
    mshadow::Tensor<xpu, 2> w        = this->params[0].data_d2_reverse();
    mshadow::Tensor<xpu, 1> b        = this->params[1].data_d1_reverse();
    prob = 0.f, loss = 0.f;

    // **** compute class prob
    if (!no_bias) {
      prob += repmat(b, pred_rep.size(0));
    }
    prob += dot(pred_rep, w.T());
    if (temperature != 1.f) {
      prob *= temperature;
    }
    mshadow::Softmax(prob,  prob);
    // ====

    int cnt = 0;
    for (int i = 0; i < pred_rep.size(0); ++i) {
      int y = static_cast<int>(label[i]);
      if (y == -1) continue;
      cnt += 1;

      float final_p = prob[i][y];
      // loss
      if (final_p == 0.) {
        loss[0] += 88; // by min float number
      } else { 
        loss[0] += -log(final_p);
      }
    }
    loss[0] /= cnt;
  }
  
  // ATTENTION, log(p_c*p_word) = log(p_c) + log(p_word), the deviation can be bp separatelly
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> prob_data = top[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> prob_diff = top[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 1> label     = bottom[1]->data_d1_reverse();

    prob_diff = F<op::identity>(prob_data);
    for (int i = 0; i < prob_data.size(0); ++i) {
      int y = static_cast<int>(label[i]); // origin word idx
      if (y == -1) {
        prob_diff[i] = 0.f;
        continue;
      }
      prob_diff[i][y] -= 1.0f;
    }

    if (temperature != 1.f) {
      prob_diff *= temperature;
    }

    // **** bp to param and bottom class
    mshadow::Tensor<xpu, 2> pred_rep_data = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> pred_rep_diff = bottom[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> w_data        = this->params[0].data_d2_reverse();
    mshadow::Tensor<xpu, 2> w_diff        = this->params[0].diff_d2_reverse();
    mshadow::Tensor<xpu, 1> b_diff        = this->params[1].diff_d1_reverse();

    w_diff += dot(prob_diff.T(), pred_rep_data);
    if (!no_bias) {
      b_diff += sum_rows(prob_diff);
    }
    pred_rep_diff += dot(prob_diff, w_data);
    // ====
  }

protected:
  bool no_bias;
  int feat_size, vocab_size;
  float temperature;
  // string w_file;
};
}  // namespace layer
}  // namespace textnet

#endif
