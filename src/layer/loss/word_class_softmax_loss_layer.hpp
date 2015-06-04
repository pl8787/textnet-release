#ifndef TEXTNET_LAYER_WORD_CLASS_SOFTMAX_LOSS_LAYER_INL_HPP_
#define TEXTNET_LAYER_WORD_CLASS_SOFTMAX_LOSS_LAYER_INL_HPP_

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

template<typename xpu>
class WordClassSoftmaxLossLayer : public Layer<xpu>{
 public:
  WordClassSoftmaxLossLayer(LayerType type) { this->layer_type = type; }
  virtual ~WordClassSoftmaxLossLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; } // pred_rep, label
  virtual int TopNodeNum() { return 4; } // class_prob, word_prob, final_prob, loss
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["word_embed_file"] = SettingV("");
    this->defaults["class_embed_file"] = SettingV("");
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["feat_size"] = SettingV();
    this->defaults["class_num"] = SettingV();
    this->defaults["vocab_size"] = SettingV();
    this->defaults["word_class_file"] = SettingV();

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(), "Layer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "WordClassSoftmaxLossLayer:top size problem.");

    feat_size = setting["feat_size"].iVal();
    class_num = setting["class_num"].iVal();
    vocab_size= setting["vocab_size"].iVal();
    word_class_file = setting["word_class_file"].sVal();
    word_embed_file = setting["word_embed_file"].sVal();
    class_embed_file = setting["class_embed_file"].sVal();

    // bottom[0], pred_rep, (batch_size, position_num, 1, feat_size)
    // bottom[1], label,    (batch_size, position_num, 1, 1)
    utils::Check(bottom[0]->data.size(0) == bottom[1]->data.size(0), "WordClassSoftmaxLossLayer: input error.");
    utils::Check(bottom[0]->data.size(1) == bottom[1]->data.size(1), "WordClassSoftmaxLossLayer: input error.");
    utils::Check(bottom[0]->data.size(3) == feat_size, "WordClassSoftmaxLossLayer: input error.");
    utils::Check(bottom[0]->data.size(2) == 1, "WordClassSoftmaxLossLayer: input error.");
    utils::Check(bottom[1]->data.size(2) == 1, "WordClassSoftmaxLossLayer: input error.");
    utils::Check(bottom[1]->data.size(3) == 1, "WordClassSoftmaxLossLayer: input error.");

    this->params.resize(4); // two embed and bias matrix
    this->params[0].Resize(1, 1, feat_size, class_num, true);
    this->params[1].Resize(1, 1, 1, class_num, true);
    this->params[2].Resize(1, 1, feat_size, vocab_size, true);
    this->params[3].Resize(1, 1, 1, vocab_size, true);

    std::map<std::string, SettingV> &w_class_setting = *setting["w_class_filler"].mVal();
    std::map<std::string, SettingV> &b_class_setting = *setting["b_class_filler"].mVal();
    std::map<std::string, SettingV> &w_word_setting = *setting["w_word_filler"].mVal();
    std::map<std::string, SettingV> &b_word_setting = *setting["b_word_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_class_setting["init_type"].iVal(),
                                               w_class_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_class_setting["init_type"].iVal(),
                                               b_class_setting, this->prnd_);
    this->params[2].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_word_setting["init_type"].iVal(),
                                               w_word_setting, this->prnd_);
    this->params[3].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_word_setting["init_type"].iVal(),
                                               b_word_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    this->params[2].Init();
    this->params[3].Init();
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "WordClassSoftmaxLossLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "WordClassSoftmaxLossLayer:top size problem.");

    int batch_size   = bottom[0]->data.size(0);
    int position_num = bottom[0]->data.size(1);
    top[0]->Resize(batch_size, position_num, 1, position_num, true);
    top[1]->Resize(batch_size, position_num, 1, vocab_size, true);
    top[2]->Resize(batch_size, position_num, 1, 1, true);
    top[3]->Resize(1, 1, 1, 1, true);
  }
  void checkNan(orc_real *p, int l) {
      for (int i = 0; i < l; ++i) {
          assert(!isnan(p[i]));
      }
  }
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> label      = bottom[1]->data;
    mshadow::Tensor<xpu, 4> class_prob = top[0]->data;
    mshadow::Tensor<xpu, 4> word_prob  = top[1]->data;
    mshadow::Tensor<xpu, 4> final_prob = top[2]->data;
    mshadow::Tensor<xpu, 1> loss       = top[3]->data_d1();

    class_prob = 0.f, word_prob = 0.f, final_prob = 0.f, loss = 0.f;

    mshadow::Tensor<xpu, 2> pred_rep_d2   = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> class_prob_d2 = top[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> w_class       = params[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 1> b_class       = params[1]->data_d1_reverse();

    // **** compute class prob
    if (!no_bias) {
        class_prob_d2 = repmat(b_class, pred_rep_d2.size(0));
    }
    class_prob_d2 += dot(class_prob_d2, w_class);
    mshadow::Softmax(class_prob_d2, class_prob_d2);
    // ==== 

    // **** compute word prob, for that label words belong to different classes, need to process one by one
    int batch_size = label.size(0);
    int position_num = label.size(1);
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      for (int pos_idx = 0; pos_idx < position_num; ++pos_idx) {
        int y = static_cast<int>(label[batch_idx][pos_idx][0][0]);
        utils::Check(y >= 0 && y < vocab_size, "WordClassSoftmaxLossLayer: label error.");
        int c = word_2_class[y];
        utils::Check(c >= 0 && c < class_num, "WordClassSoftmaxLossLayer: label error.");

        int class_begin = class_begins[c];
        int class_end   = class_ends[c];
        mshadow::Tensor<xpu, 2> w_word_one_class    = params[2]->data_d2_reverse().Slice(class_beg, class_end);
        mshadow::Tensor<xpu, 1> b_word_one_class    = params[3]->data_d1_reverse().Slice(class_beg, class_end);
        mshadow::Tensor<xpu, 2> pred_rep            = bottom[0]->data[batch_idx][pos_idx];
        mshadow::Tensor<xpu, 2> word_prob_one_class = top[1]->data_d2_reverse().Slice(class_beg, class_end);
        if (!no_bias) {
          word_prob_one_class = repmat(b_word_one_class, 1);
        } 
        word_prob_one_class += dot(pred_rep, w_word_one_class);
        mshadow::Softmax(word_prob_one_class, word_prob_one_class);

        // final label prob
        int new_word_idx_by_class = word_2_new_idx[y];
        orc_real final_p  = class_prob[batch_idx][pos_idx][0][c] * \
                               word_prob[batch_idx][pos_idx][0][new_word_idx_by_class];
        final_prob[batch_idx][pos_idx][0][0] = final_p;

        // loss
        if (final_p == 0.) {
          loss[0] += 88; // by min float number
        } else { 
          loss[0] += -log(final_p);
        }
      }
    }
    loss[0] /= (batch_size * position_num);
    // ====
  }
  
  // ATTENTION, log(p_c*p_word) = log(p_c) + log(p_word), the deviation can be bp separatelly
 virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> class_prob_data = top[0]->data;
    mshadow::Tensor<xpu, 4> class_prob_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> word_prob_data  = top[1]->data;
    mshadow::Tensor<xpu, 4> word_prob_diff  = top[1]->diff;

    mshadow::Tensor<xpu, 4> pred_rep_data  = bottom[0]->data;
    // mshadow::Tensor<xpu, 4> pred_rep_diff  = bottom[0]->diff;

    mshadow::Tensor<xpu, 4> label          = bottom[1]->data;

    class_prob_diff = F<op::identity>(class_prob_data);
    word_prob_diff  = F<op::identity>(word_prob_data);

    // **** generate top diff
    int batch_size = label.size(0);
    int position_num = label.size(1);
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      for (int pos_idx = 0; pos_idx < position_num; ++pos_idx) {
        int y = static_cast<int>(label[batch_idx][pos_idx][0][0]);
        int c = word_2_class[y];
        int new_word_idx_by_class = word_2_new_idx[y];
        
        class_prob_diff[batch_idx][pos_idx][0][c] -= 1.0f;
        word_prob_diff[batch_idx][pos_idx][0][new_word_idx_by_class] -= 1.0f;
      }
    }
    // ====
    // **** bp to param and bottom class
    mshadow::Tensor<xpu, 2> pred_rep_data_d2   = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> pred_rep_diff_d2   = bottom[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> class_prob_diff_d2 = top[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> w_data_class       = params[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> w_diff_class       = params[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 1> b_data_class       = params[1]->data_d1_reverse();
    mshadow::Tensor<xpu, 1> b_diff_class       = params[1]->diff_d1_reverse();

    w_diff_class += dot(pred_rep_data_d2.T(), class_prob_diff_d2);
    if (!no_bias) {
      b_diff_class += sum_rows(class_prob_diff_d2);
    }
    pred_rep_diff_d2 += dot(class_prob_diff_d2, w_data_class.T());
    // ====
    // **** bp to param and bottom word
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      for (int pos_idx = 0; pos_idx < position_num; ++pos_idx) {
        int y = static_cast<int>(label[batch_idx][pos_idx][0][0]);
        int c = word_2_class[y];
        int class_begin = class_begins[c];
        int class_end   = class_ends[c];

        mshadow::Tensor<xpu, 2> w_word_one_class_data = params[2]->data_d2_reverse().Slice(class_beg, class_end);
        mshadow::Tensor<xpu, 2> w_word_one_class_diff = params[2]->data_d2_reverse().Slice(class_beg, class_end);
        mshadow::Tensor<xpu, 1> b_word_one_class_data = params[3]->data_d1_reverse().Slice(class_beg, class_end);
        mshadow::Tensor<xpu, 1> b_word_one_class_diff = params[3]->data_d1_reverse().Slice(class_beg, class_end);

        mshadow::Tensor<xpu, 2> pred_rep_data            = bottom[0]->data[batch_idx][pos_idx];
        mshadow::Tensor<xpu, 2> pred_rep_diff            = bottom[0]->data[batch_idx][pos_idx];
        mshadow::Tensor<xpu, 2> word_prob_one_class_diff = top[1]->diff_d2_reverse().Slice(class_beg, class_end);

        w_word_one_class_diff += dot(pred_rep_data.T(), word_prob_one_class_diff);
        if (!no_bias) {
          b_diff_class += sum_rows(class_prob_diff_d2);
        }
        pred_rep_diff += dot(word_prob_one_class_diff, w_word_one_class_data.T());
      }
    }
    // ====
  }

  void load_word_class_file(void) {
    ifstream ifs(word_class_file);
    utils::Check(ifs.is_open(), "WordClassSoftmaxLossLayer: open word class file error.");
    word_2_class = vector<int>(vocab_size, -1);
    int cnt = 0;
    while (!ifs.eof()) {
      int word_idx = -1, class_idx = -1;
      ifs >> word_idx >> class_idx;
      if (word_idx == -1) break;
      word2class[word_idx] = class_idx;
      cnt += 1;
    }
    utils::Check(cnt == vocab_size, "WordClassSoftmaxLossLayer: word class error.");
    for (size_t i = 0; i < word2class.size(); ++i) {
      utils::Check(word2class[i] >= 0 && word2class[i] < class_num, "WordClassSoftmaxLossLayer: word class error.");
    }
  }

  // this will reorder word embeddings
  // must load word embeddings first
  void construct_class(void) {
    class_word_num = vector<int>(class_num, 0);
    for (size_t i = 0; i < word2class.size(); ++i) {
      class_word_num[word2class[i]] += 1;
    }

    class_begins.push_back(0);
    int cnt = 0;
    for (size_t i = 1; i < class_word_num.size(); ++i) {
      cnt += class_word_num[i-1];
      class_begins.push_back(cnt);
      class_ends.push_back(cnt);
    }
    class_ends.push_back(vocab_size);

    word_2_new_idx[i] = vector<int>(vocab_size, -1);
    // word 2 new idx
    vector<int> class_word_idx(class_num, 0);
    for (size_t i = 0; i < word2class.size(); ++i) {
      int c = word2class[i];
      int new_idx = class_word_idx[c]+class_begins[i];
      ++class_word_idx[c];
      word_2_new_idx[i] = new_idx;
    }
  }

  void reoragnize_word_embed() {
    mshadow::TensorContainer<xpu, 4> w_ori_data;
    for (int i = 0) {
    }
  }

  void prepare(void) {
    load_word_class_file();
    construct_class();
    reoragnize_word_embed();
  }
  
protected:
  int feat_size, vocab_size, class_num;
  string word_class_file, word_embed_file, class_embed_file;

  vector<int> word_2_class, word_2_new_idx, class_word_num, class_begins, class_ends;
};
}  // namespace layer
}  // namespace textnet
#endif
