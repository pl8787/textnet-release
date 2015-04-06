#ifndef TEXTNET_CONCAT_LAYER_INL_HPP_
#define TEXTNET_CONCAT_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class ConcatLayer : public Layer<xpu>{
 public:
  ConcatLayer(LayerType type) { this->layer_type = type; }
  virtual ~ConcatLayer(void) {}
  
  virtual int BottomNodeNum() { return nBottomNode; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }

  virtual void Require() {
    // default value, just set the value you want

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["bottom_node_num"] = SettingV();

    
    Layer<xpu>::Require();
  }

  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    utils::Check(setting.count("bottom_node_num"), "ConcatLayer: setting problem."); 
    nBottomNode = setting["bottom_node_num"].i_val;
  }
  
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                 "ConcatLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                 "ConcatLayer:top size problem.");
    int out_size = 0, batch_size = 0;
    for (int i = 0; i < BottomNodeNum(); ++i) {
      mshadow::Shape<4> shape_in = bottom[i]->data.shape_;
      utils::Check(shape_in[1] == 1 && shape_in[2] == 1, "ConcatLayer: bottom size problem");
      out_size += shape_in[3];
      if (i == 0) {
        batch_size = shape_in[0];
      } else {
        utils::Check(shape_in[0] == batch_size, "ConcatLayer: bottom size problem");
      }
    }
    mshadow::Shape<4> shape_out = mshadow::Shape4(batch_size, 1, 1, out_size);
    top[0]->Resize(shape_out, true);
  }

  // void checkNan(float *p, int l) {
  //     for (int i = 0; i < l; ++i) {
  //         assert(!isnan(p[i]));
  //     }
  // }
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

    for (int i = 0; i < top[0]->data.size(0); ++i) { // for each batch
      int cnt = 0;
      mshadow::Tensor<xpu, 1> top_data = top[0]->data[i][0][0];
      for (int j = 0; j < BottomNodeNum(); ++j) { // for each input node
        mshadow::Tensor<xpu, 1> bottom_data = bottom[j]->data[i][0][0];
        top_data.Slice(cnt, cnt+bottom_data.size(0)) = F<op::identity>(bottom_data);
        cnt += bottom_data.size(0);
      }
      utils::Assert(cnt == top[0]->data.size(3), "ConcatLayer: forward error.");
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    for (int i = 0; i < top[0]->diff.size(0); ++i) { // for each batch
      int cnt = 0;
      mshadow::Tensor<xpu, 1> top_diff = top[0]->diff[i][0][0];
      for (int j = 0; j < BottomNodeNum(); ++j) { // for each input node
        mshadow::Tensor<xpu, 1> bottom_diff = bottom[j]->diff[i][0][0];
        bottom_diff = F<op::identity>(top_diff.Slice(cnt, cnt+bottom_diff.size(0)));
        cnt += bottom_diff.size(0);
      }
      utils::Assert(cnt == top[0]->diff.size(3), "ConcatLayer: bp error.");
    }
  }
 protected:
  int nBottomNode;
};
}  // namespace layer
}  // namespace textnet
#endif
