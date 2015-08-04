#ifndef TEXTNET_LAYER_ONE_HOT_LAYER_INL_HPP_
#define TEXTNET_LAYER_ONE_HOT_LAYER_INL_HPP_

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
class OneHotLayer : public Layer<xpu>{
 public:
  OneHotLayer(LayerType type) { this->layer_type = type; }
  virtual ~OneHotLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["pad_value"] = SettingV(0.f);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["word_count"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(), "OneHotLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "OneHotLayer:top size problem.");
                  
    word_count = setting["word_count"].iVal();
    pad_value = setting["pad_value"].fVal();
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "OneHotLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "OneHotLayer:top size problem.");
    
    max_doc_len = bottom[0]->data.size(3);
    doc_count = bottom[0]->data.size(1);
    nbatch = bottom[0]->data.size(0);
                  
    top[0]->Resize(nbatch, doc_count, max_doc_len, word_count, true);

	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len  = top[0]->length;
    
    // fill all top data to pad_value
    top_data = pad_value;
    top_len = F<op::identity>(bottom_len);

    int w_idx = -1;
    for (int i = 0; i < nbatch; ++i) {
      for (int j = 0; j < doc_count; ++j) {
        int doc_len = bottom_len[i][j];
        utils::Check(doc_len >= 0, "Embedding layer: length must be inited.");
        for (int k = 0; k < doc_len; ++k) {
          w_idx = (int)bottom_data[i][j][0][k];
          if (w_idx != -1) {
            top_data[i][j][k] = 0.;
            top_data[i][j][k][w_idx] = 1;
          }
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
  }

 protected:
  int word_count, doc_count, nbatch, max_doc_len;
  float pad_value;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_EMBEDDING_LAYER_INL_HPP_

