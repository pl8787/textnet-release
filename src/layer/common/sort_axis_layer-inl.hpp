#ifndef TEXTNET_LAYER_SORT_AXIS_LAYER_INL_HPP_
#define TEXTNET_LAYER_SORT_AXIS_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

// sum aross one axis
template<typename xpu>
class SortAxisLayer : public Layer<xpu> {
 public:
  SortAxisLayer(LayerType type) { this->layer_type = type; }
  virtual ~SortAxisLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["axis"] = SettingV(3);
    this->defaults["reverse"] = SettingV(false);
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "SortAxisLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SortAxisLayer:top size problem.");
    axis = setting["axis"].iVal();
    reverse = setting["reverse"].bVal();
    utils::Check(0 < axis && axis < 4, "SortAxisLayer: axis error.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "SortAxisLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "SortAxisLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out= shape_in;
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

  vector<size_t> sort_indexes(vector<float> & vecinput){
    vector<size_t> idx(vecinput.size());
    iota(idx.begin(), idx.end(), 0);
    if(reverse)
        sort(idx.begin(), idx.end(), [&vecinput](size_t k1,size_t k2){ return vecinput[k1] > vecinput[k2];});
    else
        sort(idx.begin(), idx.end(), [&vecinput](size_t k1,size_t k2){ return vecinput[k1] < vecinput[k2];});
    return std::move(idx);
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

    if (axis == 3) { 
      mshadow::Tensor<xpu, 2> bottom_data2 = bottom[0]->data_d2();
      mshadow::Tensor<xpu, 2> top_data2 = top[0]->data_d2();
      for(int i = 0 ; i < bottom_data2.size(0); ++ i){
        vector<float> vecscore(bottom_data2[i].dptr_, bottom_data2[i].dptr_ + bottom_data2.size(1));
        vecidx.push_back(sort_indexes(vecscore));
        for(int j  = 0 ; j < bottom_data2.size(1); ++ j)
          top_data2[i][j] = vecscore[vecidx[i][j]];
      }
      top[0]->length = F<op::identity>(bottom[0]->length);
    } else if (axis == 2) {
    } else if (axis == 1) {
    } else if (axis == 0) {
    }

  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    if(!this->prop_error[0]) return;
    using namespace mshadow::expr;

    if (axis == 3) {
      mshadow::Tensor<xpu, 2> bottom_diff2 = bottom[0]->diff_d2();
      mshadow::Tensor<xpu, 2> top_diff2 = top[0]->diff_d2();
      for(int i = 0 ; i < bottom_diff2.size(0); ++ i){
        for(int j = 0 ; j < bottom_diff2.size(1); ++ j){
          bottom_diff2[i][vecidx[i][j]] += top_diff2[i][j];
        }
      }
    } else if(axis == 2) {
    } else if(axis == 1) {
    } else if(axis == 0) {
    }
    return;
  }
 protected:
  vector<vector<size_t>> vecidx;
  int axis;
  bool reverse;
};
}  // namespace layer
}  // namespace textnet
#endif
