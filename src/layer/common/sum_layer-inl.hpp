#ifndef TEXTNET_LAYER_SUM_LAYER_INL_HPP_
#define TEXTNET_LAYER_SUM_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

// sum aross one axis
template<typename xpu>
class SumLayer : public Layer<xpu> {
 public:
  SumLayer(LayerType type) { this->layer_type = type; }
  virtual ~SumLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["axis"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "SumLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SumLayer:top size problem.");
    axis = setting["axis"].iVal();
    utils::Check(0 < axis && axis < 4, "SumLayer: axis error.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "SumLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SumLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out= shape_in;
    shape_out[axis] = 1;

    top[0]->Resize(shape_out, true);

	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		top[0]->PrintShape("top0");
	}
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (! (bottom[0]->data.size(0) == top[0]->data.size(0))) {
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
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Shape<4>       bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 4> top_data    = top[0]->data;

    if (axis == 3) { 
      top[0]->length = F<op::identity>(bottom[0]->length); 
    } else if (axis == 1) {
      for (int i = 0; i < bottom_shape[0]; ++i) {
        top[0]->length[i][0] = bottom[0]->length[i][0]; 
      }
    } else if (axis == 0) {
      top[0]->length[0] = F<op::identity>(bottom[0]->length[0]); 
    }

    //if (axis == 2 && bottom[0]->length[0][0] > -1) { // var len, this is just for check, not for flag
    //  int length = bottom[0]->length[0][0];
    //  if (length < bottom_shape[2]) {
    //    utils::Check(bottom_data[0][0][length][0] == 0.f, "SumLayer: error, var len must be padding as zero.");
    //  }
    //}
    
    int left = 1, right = 1, middle = bottom_shape[axis];
    for (int i = 0; i < axis; ++i) {
        left *= int(bottom_shape[i]);
    }
    for (int i = axis+1; i < 4; ++i) {
        right *= int(bottom_shape[i]);
    }

    mshadow::Tensor<xpu, 3> input(bottom_data.dptr_, mshadow::Shape3(left, middle, right));
    mshadow::Tensor<xpu, 3> output(top_data.dptr_,   mshadow::Shape3(left, 1, right));
    output = 0.f;
    for (int l_idx = 0; l_idx < left; ++l_idx) {
      for (int r_idx = 0; r_idx < right; ++r_idx) {
        for (int m_idx = 0; m_idx < middle; ++m_idx) {
          output[l_idx][0][r_idx] += input[l_idx][m_idx][r_idx];
        }
      }
    } 
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Shape<4>       bottom_shape= bottom[0]->data.shape_;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_diff    = top[0]->diff;

    int left = 1, right = 1, middle = bottom_shape[axis];
    for (int i = 0; i < axis; ++i) {
        left *= int(bottom_shape[i]);
    }
    for (int i = axis+1; i < 4; ++i) {
        right *= int(bottom_shape[i]);
    }

    mshadow::Tensor<xpu, 3> input_diff(bottom_diff.dptr_, mshadow::Shape3(left, middle, right));
    mshadow::Tensor<xpu, 3> output_diff(top_diff.dptr_,   mshadow::Shape3(left, 1, right));
    for (int l_idx = 0; l_idx < left; ++l_idx) {
      for (int r_idx = 0; r_idx < right; ++r_idx) {
        for (int m_idx = 0; m_idx < middle; ++m_idx) {
          input_diff[l_idx][m_idx][r_idx] += output_diff[l_idx][0][r_idx];
        }
      }
    }
  }
 protected:
  int axis;
};
}  // namespace layer
}  // namespace textnet
#endif
