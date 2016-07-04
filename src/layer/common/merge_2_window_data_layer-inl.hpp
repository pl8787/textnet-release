#ifndef TEXTNET_LAYER_MERGE_2_WINDOWDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_MERGE_2_WINDOWDATA_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>
#include "stdlib.h"

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

using namespace std;

namespace textnet {
namespace layer {

/*
   * Merge2WindowDataLayer : split each document into several passages
   * param 0 : bottom[0] records query infomation: ( batch-size, 1, 1, 1)
   * param 1 : bottom[1] records doc information:  ( old-batch-size, 1, 1, 1)
   * out 0  :  top[0] : match infotmation:  (old-batch-size, 1, 1, k)
*/
template<typename xpu>
class Merge2WindowDataLayer : public Layer<xpu>{
 public:
  Merge2WindowDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~Merge2WindowDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "Merge2WindowDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "Merge2WindowDataLayer:top size problem.");
  }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "Merge2WindowDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "Merge2WindowDataLayer:top size problem.");
    Tensor2D bottom0_length = bottom[0]->length;
    Tensor2D bottom1_length = bottom[1]->length;

    int idim = 0;
    for(index_t i = 0 ; i < bottom[1]->data.size(0); ++ i){
        idim = idim > bottom[1]->data[i][0][0][0] ? idim : bottom[1]->data[i][0][0][0];
    }

    top[0]->Resize(bottom1_length.size(0), 1, 1, idim, true);
    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        bottom[1]->PrintShape("bottom1");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    this->Reshape(bottom, top);
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top0_data = top[0]->data;
    Tensor2D top0_length = top[0]->length;
    //printf("bottom[0]:%d,%d,%d,%d\n",bottom[0]->data.size(0),bottom[0]->data.size(1),bottom[0]->data.size(2),bottom[0]->data.size(3));
    utils::Check(bottom[0]->data.size(1) == 1 && bottom[0]->data.size(2) == 1, " Merge2WindowDataLayer:: bottom[0] size problem.");

    index_t k = 0;
    for(index_t i = 0 ; i < top0_data.size(0); ++ i){
        int curr_len = bottom[1]->data[i][0][0][0];
        utils::Check(curr_len > 0," Merge2WindowDataLayer: curr_len must be above 0.");
        top0_length[i][0] = curr_len;
        for(index_t j = 0 ; j < curr_len; ++ j){
            top0_data[i][0][0][j] = bottom[0]->data[k][0][0][0];
            ++k;
        }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top0_diff = top[0]->diff;
    index_t k = 0;
    for(index_t i = 0 ; i < top0_diff.size(0); ++ i){ // copy query diff info
        int curr_len = bottom[1]->data[i][0][0][0];
        for(index_t j = 0 ; j < curr_len; ++ j){
            bottom[0]->diff[k][0][0][0] += top0_diff[i][0][0][j];
            ++k;
        }
    }
  }
  
 protected:
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MAP_2_TEXTDATA_LAYER_INL_HPP_

