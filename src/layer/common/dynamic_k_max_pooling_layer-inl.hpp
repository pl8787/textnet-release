#ifndef TEXTNET_LAYER_DYNAMIC_K_MAX_POOLING_LAYER_INL_HPP_
#define TEXTNET_LAYER_DYNAMIC_K_MAX_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

// ref: Yin, W., et al., "MultiGranCNN: An Architecture for General Matching of Text Chunks on Multiple Levels of Granularity", ACL'14
//
// this layer is different from dynamic pooling layer
// 1. k is not fixed and is rescaled by the input length
// 2. this layer selects top k values of one dimension which are position independent
template<typename xpu>
class DynamicKMaxPoolingLayer : public Layer<xpu>{
 public:
  DynamicKMaxPoolingLayer(LayerType type) { this->layer_type = type; }
  virtual ~DynamicKMaxPoolingLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; } // input_rep for pooling, origin_word_embedding_rep for length info
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["L"] = SettingV();                   // L in the num of pooling layers used in the whole model
    this->defaults["l"] = SettingV();                   // l is begin from 1
    this->defaults["max_sentence_length"] = SettingV(); // the max length of the original sentence
    this->defaults["min_rep_length"] = SettingV();      // avoid too short representations in middle layer

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    L = setting["L"].iVal();
    l = setting["l"].iVal();
    max_sentence_length = setting["max_sentence_length"].iVal();
    min_rep_length      = setting["min_rep_length"].iVal();
    utils::Check(l > 0 && l <= L, "DynamicKMaxPoolingLayer: parameter error.");
  }

  int get_dynamic_k(int sentence_length, int min_rep_length, int L, int l) {
    if (l == L) 
        return 1;
    int k = ((L-l)*sentence_length + L-1) / L;
    if (k < min_rep_length) 
        k = min_rep_length;
    return k;
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "DynamicKMaxPoolingLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "DynamicKMaxPoolingLayer: top size problem.");
    
    int max_k = get_dynamic_k(max_sentence_length, min_rep_length, L, l);
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    int batch_size = shape_in[0];
    int sentence_num = shape_in[1];
    int max_len = shape_in[2];
    int feat_size = shape_in[3];
    mshadow::Shape<4> shape_out = mshadow::Shape4(batch_size, sentence_num, max_k, feat_size);

    top[0]->Resize(shape_out, true);
    pos.Resize(shape_out, true);

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

    if (top[0]->data.shape_[0] != bottom[0]->data.shape_[0]) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }
 
  typedef mshadow::Tensor<xpu,2> Tensor2D;
  typedef mshadow::Tensor<xpu,2,int> Tensor2DInt;

  void pooling_one_matrix(Tensor2D t_in, Tensor2D t_out, Tensor2DInt pos, int len, int k) {
    utils::Check(t_out.size(0) == pos.size(0) && t_out.size(1) == pos.size(1), "DynamicKMaxPoolingLayer: size error.");
    utils::Check(t_out.size(1) == t_in.size(1), "DynamicKMaxPoolingLayer: size error.");
    utils::Check(k <= t_out.size(0) && len <= t_in.size(0), "DynamicKMaxPoolingLayer: size error.");

    pos = -1; t_out = 0.f;
    if (k > len) {
      // if input is shorter than output, the val will be padded with 0 and pos will be set to -1
      for (int j = 0; j < t_in.size(1); ++j) {
        for (int i = 0; i < len; ++i) {
          t_out[i][j] = t_in[i][j];
          pos[i][j] = i;
        }
      }
    } else {
      vector<float> v(len, 0);
      for (int j = 0; j < t_in.size(1); ++j) {
        for (int i = 0; i < len; ++i) {
          v[i] = t_in[i][j];
        }
        std::partial_sort(v.begin(), v.begin() + k, v.end(), std::greater<float>());
        float topk_threshold = v[k-1];

        int p = 0;
        for (int i = 0; i < len; ++i) {
          if (t_in[i][j] >= topk_threshold) {
            t_out[p][j] = t_in[i][j];
            pos[p][j] = i;
            ++p;
            if (p == k) break;
          }
        }
      }
    }
  }

  void unpooling_one_matrix(Tensor2D t_in, Tensor2D t_out, Tensor2DInt pos, int len, int k) {
    for (int j = 0; j < pos.size(1); ++j) {
      for (int i = 0; i < k; ++i) {
        if (pos[i][j] == -1) {
          utils::Check(len < k && i == len, "DynamicKMaxPoolingLayer: length error.");
          break;
        }
        t_in[pos[i][j]][j] += t_out[i][j];
      }
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data  = bottom[0]->data;
    mshadow::Tensor<xpu, 2> rep_len      = bottom[0]->length;
    mshadow::Tensor<xpu, 2> sentence_len = bottom[1]->length;
    mshadow::Tensor<xpu, 4> top_data     = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len      = top[0]->length;

    top_data = 0; pos = -1; top_len = -1;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      for (index_t sen_idx = 0; sen_idx < bottom_data.size(1); ++sen_idx) {
        int sen_len = sentence_len[batch_idx][sen_idx];
        int len     = rep_len[batch_idx][sen_idx];
        int k = get_dynamic_k(sen_len, min_rep_length, L, l);
        top_len[batch_idx][sen_idx] = k;
        pooling_one_matrix(bottom_data[batch_idx][sen_idx], 
                           top_data[batch_idx][sen_idx],
                           pos[batch_idx][sen_idx],
                           len, 
                           k);
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff  = bottom[0]->diff;
    mshadow::Tensor<xpu, 2> rep_len      = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_diff     = top[0]->diff;
    mshadow::Tensor<xpu, 2> top_len      = top[0]->length;

    for (index_t batch_idx = 0; batch_idx < bottom_diff.size(0); ++batch_idx) {
      for (index_t sen_idx = 0; sen_idx < bottom_diff.size(1); ++sen_idx) {
        int len = rep_len[batch_idx][sen_idx];
        int k   = top_len[batch_idx][sen_idx];
        unpooling_one_matrix(bottom_diff[batch_idx][sen_idx], top_diff[batch_idx][sen_idx], pos[batch_idx][sen_idx], len, k);
      }
    }
  }
 protected:
  mshadow::TensorContainer<xpu, 4, int> pos;
  int L, l, max_sentence_length, min_rep_length;
};
}  // namespace layer
}  // namespace textnet
#endif

