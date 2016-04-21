#ifndef TEXTNET_LAYER_PRODUCT_LAYER_INL_HPP_
#define TEXTNET_LAYER_PRODUCT_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

// this layer elem-wise products bottom representations on the last 2 dimensions
template<typename xpu>
class ProductLayer : public Layer<xpu> {
 public:
  ProductLayer(LayerType type) { this->layer_type = type; }
  virtual ~ProductLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["mu1"] = SettingV(0.0f);   // for activation mu1*bottom0_data
    this->defaults["mu2"] = SettingV(0.0f);   // for activation mu2*bootom1_data
	this->defaults["delta1"] = SettingV(0.0f);  // for L2 norm activation
	this->defaults["delta2"] = SettingV(0.0f);  // for L2 norm activation
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);

	mu1 = setting["mu1"].fVal();
	mu2 = setting["mu2"].fVal();
	delta1 = setting["delta1"].fVal();
	delta2 = setting["delta2"].fVal();
    
    utils::Check(bottom.size() == BottomNodeNum(), "ProductLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ProductLayer:top size problem.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "ProductLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ProductLayer:top size problem.");
    
    mshadow::Shape<4> shape0 = bottom[0]->data.shape_;
    mshadow::Shape<4> shape1 = bottom[1]->data.shape_;
	mshadow::Shape<2> shape_len = bottom[0]->length.shape_;

    utils::Check(shape0[0] == shape1[0], "ProductLayer: bottom sizes does not match.");
    utils::Check(shape0[1] == shape1[1], "ProductLayer: bottom sizes does not match.");
    utils::Check(shape0[2] == shape1[2], "ProductLayer: bottom sizes does not match.");

    int size_0_3 = shape0[3];
    int size_1_3 = shape1[3];
    utils::Check(size_0_3 == 1 || size_1_3 == 1 || size_0_3 == size_1_3, "ProductLayer: bottom sizes does not match.");
	if (mu1 != 0 || mu2 != 0) {
	  utils::Check(size_0_3 == size_1_3, "ProductLayer: mu1 or mu2 have value, only support same size.");
	}

    int output_size = size_0_3 > size_1_3 ? size_0_3 : size_1_3;
    top[0]->Resize(shape0[0], shape0[1], shape0[2], output_size, shape_len[0], shape_len[1], true);

	if (show_info) {
      bottom[0]->PrintShape("bottom0");
      bottom[1]->PrintShape("bottom1");
      top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    top[0]->length = F<op::identity>(bottom[0]->length); // bottom nodes should have the same length

    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> top_data     = top[0]->data_d2_reverse();

    if (bottom0_data.size(1) == bottom1_data.size(1)) {
      top_data = bottom0_data * bottom1_data;
	  if (mu1 != 0) {
		top_data -= mu1 * bottom0_data * bottom0_data;
	  } 
	  if (mu2 != 0) {
		top_data -= mu2 * bottom1_data * bottom1_data;
	  }
    } else {
      for (int i = 0; i < bottom0_data.size(0); ++i) {
        if (bottom0_data.size(1) == 1) {
          top_data[i] = bottom0_data[i][0] * bottom1_data[i];
        } else if (bottom1_data.size(1) == 1) {
          top_data[i] = bottom0_data[i]    * bottom1_data[i][0];
        } else {
          utils::Assert(false, "ProductLayer: ff data size error.");
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> top_diff     = top[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom0_diff = bottom[0]->diff_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom1_data = bottom[1]->data_d2_reverse();
    mshadow::Tensor<xpu, 2> bottom1_diff = bottom[1]->diff_d2_reverse();

    if (bottom0_data.size(1) == bottom1_data.size(1)) {
      bottom0_diff += top_diff * bottom1_data;
      bottom1_diff += top_diff * bottom0_data;
	  if (mu1 != 0) {
		bottom0_diff -= 2 * mu1 * top_diff * bottom0_data;
	  }
	  if (mu2 != 0) {
		bottom1_diff -= 2 * mu2 * top_diff * bottom1_data;
      }
	  if (delta1 != 0) {
		bottom0_diff += delta1 * bottom0_data;
	  }
	  if (delta2 != 0) {
		bottom1_diff += delta2 * bottom1_data;
	  }
    } else {
      for (int i = 0; i < bottom0_data.size(0); ++i) {
        if (bottom0_data.size(1) == 1) {
          bottom1_diff[i] += top_diff[i] * bottom0_data[i][0];
          for (int j = 0; j < bottom1_data.size(1); ++j) {
            bottom0_diff[i][0] += top_diff[i][j] * bottom1_data[i][j];
          }
        } else if (bottom1_data.size(1) == 1) {
          bottom0_diff[i] += top_diff[i] * bottom1_data[i][0];
          for (int j = 0; j < bottom0_data.size(1); ++j) {
            bottom1_diff[i][0] += top_diff[i][j] * bottom0_data[i][j];
          }
        } else {
          utils::Assert(false, "ProductLayer: bp data size error.");
        }
      }
    }
  }

 protected:
  float mu1;
  float mu2;
  float delta1;
  float delta2;

};
}  // namespace layer
}  // namespace textnet
#endif
