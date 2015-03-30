#ifndef TEXTNET_LAYER_ACTIVATION_LAYER_INL_HPP_
#define TEXTNET_LAYER_ACTIVATION_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu,typename ForwardOp, typename BackOp>
class WholePoolingLayer : public Layer<xpu>{
 public:
  WholePoolingLayer(LayerType type) { this->layer_type = type; }
  virtual ~WholePoolingLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "WholePoolingLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "WholePoolingLayer:top size problem.");
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], shape_in[1], 1, shape_in[3]);
    top[0]->Resize(shape_out);
    pos.Resize(shape_in);
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    // top[0]->data = F<ForwardOp>(bottom[0]->data);
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    // if (this->prop_error[0]) {
    //   bottom[0]->diff = F<BackOp>(top[0]->data) * top[0]->diff;
    // }
  }
 protected:
  mshadow::TensorContainer<xpu, 4> pos;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_ACTIVATION_LAYER_INL_HPP_

