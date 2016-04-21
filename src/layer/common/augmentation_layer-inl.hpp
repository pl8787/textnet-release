#ifndef TEXTNET_LAYER_AUGMENTATION_LAYER_INL_HPP_
#define TEXTNET_LAYER_AUGMENTATION_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

// this layer elem-wise products bottom representations on the last 2 dimensions
template<typename xpu>
class AugmentationLayer : public Layer<xpu> {
 public:
  AugmentationLayer(LayerType type) { this->layer_type = type; }
  virtual ~AugmentationLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["scale_x"] = SettingV("1 1");  
    this->defaults["scale_y"] = SettingV("1 1");  
	this->defaults["shift_x"] = SettingV("0 0");  
	this->defaults["shift_y"] = SettingV("0 0");  
	this->defaults["rotate_x"] = SettingV("0 0"); 
	this->defaults["rotate_y"] = SettingV("0 0"); 
	this->defaults["eq_scale"] = SettingV(false);
	this->defaults["eq_shift"] = SettingV(false);
	this->defaults["batch_wise"] = SettingV(true);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    Layer<xpu>::Require();
  }
  
  vector<float> ReadRange(string str) {
	vector<float> rg(2);
	istringstream iss;
	iss.str(str);
	iss >> rg[0] >> rg[1];
	return rg;
  }

  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);

	scale_x_str = setting["scale_x"].sVal();
	scale_y_str = setting["scale_y"].sVal();
	shift_x_str = setting["shift_x"].sVal();
	shift_y_str = setting["shift_y"].sVal();
	rotate_x_str = setting["rotate_x"].sVal();
	rotate_y_str = setting["rotate_y"].sVal();

	eq_scale = setting["eq_scale"].bVal();
	eq_shift = setting["eq_shift"].bVal();
	batch_wise = setting["batch_wise"].bVal();

	scale_x = ReadRange(scale_x_str);
	scale_y = ReadRange(scale_y_str);
	shift_x = ReadRange(shift_x_str);
	shift_y = ReadRange(shift_y_str);
	rotate_x = ReadRange(rotate_x_str);
	rotate_y = ReadRange(rotate_y_str);
    
    utils::Check(bottom.size() == BottomNodeNum(), "AugmentationLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "AugmentationLayer:top size problem.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "AugmentationLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "AugmentationLayer:top size problem.");
    
    mshadow::Shape<4> shape0 = bottom[0]->data.shape_;
	mshadow::Shape<2> shape0_len = bottom[0]->length.shape_;

    top[0]->Resize(shape0, shape0_len);

	if (show_info) {
      bottom[0]->PrintShape("bottom0");
      top[0]->PrintShape("top0");
	}
  }
  
  float SampleUniform(vector<float> rg) {
    std::uniform_real_distribution<float> distribution(rg[0], rg[1]);
	return distribution(generator);
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    top[0]->length = F<op::identity>(bottom[0]->length); // bottom nodes should have the same length

    mshadow::Tensor<xpu, 4> bottom0_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data     = top[0]->data;

	max_w = bottom0_data.size(3);
	max_h = bottom0_data.size(2);

	top_data = 0;

	if (batch_wise) {
	  float c_scale_x = SampleUniform(scale_x);
	  float c_scale_y = SampleUniform(scale_y);
	  float c_shift_x = SampleUniform(shift_x);
	  float c_shift_y = SampleUniform(shift_y);
	  float c_rotate_x = SampleUniform(rotate_x);
	  float c_rotate_y = SampleUniform(rotate_y);

	  if (eq_scale) {
	    c_scale_y = c_scale_x;
	  }
	  if (eq_shift) {
	    c_shift_y = c_shift_x;
	  }

      for (int h = 0; h < bottom0_data.size(2); ++h) {
	    for (int w = 0; w < bottom0_data.size(3); ++w) {
	  	  int ori_w = c_scale_x * w + c_rotate_y * h + c_shift_x;
	  	  int ori_h = c_scale_y * h + c_rotate_x * w + c_shift_y;
	  	  if (ori_w < 0 || ori_w >= max_w || ori_h < 0 || ori_h >= max_h) {
	  	    continue;
	  	  }
          for (int i = 0; i < bottom0_data.size(0); ++i) {
	        for (int j = 0; j < bottom0_data.size(1); ++j) {
              top_data[i][j][h][w] = bottom0_data[i][j][ori_h][ori_w];
	  	    }
	  	  }
	    }
      }
	} else {
      for (int i = 0; i < bottom0_data.size(0); ++i) {
	    float c_scale_x = SampleUniform(scale_x);
	    float c_scale_y = SampleUniform(scale_y);
	    float c_shift_x = SampleUniform(shift_x);
	    float c_shift_y = SampleUniform(shift_y);
	    float c_rotate_x = SampleUniform(rotate_x);
	    float c_rotate_y = SampleUniform(rotate_y);

	    if (eq_scale) {
	      c_scale_y = c_scale_x;
	    }
	    if (eq_shift) {
	      c_shift_y = c_shift_x;
	    }


	    for (int j = 0; j < bottom0_data.size(1); ++j) {
          for (int h = 0; h < bottom0_data.size(2); ++h) {
	        for (int w = 0; w < bottom0_data.size(3); ++w) {
	  	      int ori_w = c_scale_x * w + c_rotate_y * h + c_shift_x;
	    	  int ori_h = c_scale_y * h + c_rotate_x * w + c_shift_y;
	  	      if (ori_w < 0 || ori_w >= max_w || ori_h < 0 || ori_h >= max_h) {
	  	        continue;
	  	      }
              top_data[i][j][h][w] = bottom0_data[i][j][ori_h][ori_w];
	  	    }
	  	  }
	    }
      }
	}
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

  }

 protected:
  std::default_random_engine generator;
  string scale_x_str, scale_y_str;
  string shift_x_str, shift_y_str;
  string rotate_x_str, rotate_y_str;
  vector<float> scale_x, scale_y;
  vector<float> shift_x, shift_y;
  vector<float> rotate_x, rotate_y;
  bool eq_scale;
  bool eq_shift;
  bool batch_wise;

  int max_w, max_h;

};
}  // namespace layer
}  // namespace textnet
#endif
