#ifndef TEXTNET_LAYER_PHRASE_AVE_REP_LAYER_INL_HPP_
#define TEXTNET_LAYER_PHRASE_AVE_REP_LAYER_INL_HPP_

#include <iostream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../../utils/utils.h"
#include "../../io/json/json.h"
#include <cassert>

namespace textnet {
namespace layer {

template<typename xpu>
class PhraseAveRepLayer : public Layer<xpu> {
 public:
  PhraseAveRepLayer(LayerType type) { this->layer_type = type; }
  virtual ~PhraseAveRepLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["window"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);                        
    
    utils::Check(bottom.size() == BottomNodeNum(), "PhraseAveRepLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "PhraseAveRepLayer:top size problem.");
                  
    window = setting["window"].iVal();
  }

  // bottom should be padded with only one zero on both sides
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "PhraseAveRepLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "PhraseAveRepLayer:top size problem.");
    
    mshadow::Shape<4> shape_in  = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], shape_in[2], shape_in[3]);

    top[0]->Resize(shape_out, true);

	bottom[0]->PrintShape("bottom0");
	top[0]->PrintShape("top0");
  }

  // compose phrase rep by ave word reps
  void ComposeOnePosition(Tensor2D seq, int pos, int len, Tensor2D phrase_ave_rep) {
    phrase_ave_rep = 0.f;
    for (int i = pos; i < pos+window; ++i) {
      if (i < len) {
        phrase_ave_rep += seq.Slice(i, i+1);
      } 
    }
    phrase_ave_rep /= window;
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    Tensor4D bottom_data = bottom[0]->data;
    Tensor4D top_data = top[0]->data;
    top[0]->length = mshadow::expr::F<op::identity>(bottom[0]->length);

    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < bottom_data.size(1); ++seq_idx) {
        int len = bottom[0]->length[batch_idx][seq_idx];
        utils::Assert(len >= 0, "PhraseAveRepLayer: sequence length error.");
        for (int pos = 0; pos < len; ++pos) {
          Tensor2D phrase_rep = top_data[batch_idx][seq_idx].Slice(pos, pos+1);
          ComposeOnePosition(bottom_data[batch_idx][seq_idx], pos, len, phrase_rep);
        }
      }
    }
  }

  // compose phrase rep by ave word reps
  void BpOnePosition(Tensor2D seq, int pos, int len, Tensor2D phrase_ave_rep) {
    for (int i = pos; i < pos+window; ++i) {
      if (i < len) {
        seq.Slice(i, i+1) += phrase_ave_rep/window;
      } 
    }
  }

  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
        
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < bottom_data.size(1); ++seq_idx) {
        int len = bottom[0]->length[batch_idx][seq_idx];
        utils::Assert(len >= 0, "PhraseAveRepLayer: sequence length error.");
        for (int pos = 0; pos < len; ++pos) {
          Tensor2D phrase_rep_diff = top_diff[batch_idx][seq_idx].Slice(pos, pos+1);
          ComposeOnePosition(bottom_diff[batch_idx][seq_idx], pos, len, phrase_rep_diff);
        }
      }
    }
  }

 public:
// protected:
  int window;
};
}  // namespace layer
}  // namespace textnet
#endif 
