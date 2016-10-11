#ifndef TEXTNET_LAYER_MERGE_WINDOW_LAYER_INL_HPP_
#define TEXTNET_LAYER_MERGE_WINDOW_LAYER_INL_HPP_

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
   * MergeWindowLayer : split each document into several passages
   * param 0 : bottom[0] records query infomation: ( batch-size, 1, 1, 1)
   * param 1 : bottom[1] records doc information:  ( old-batch-size, 1, 1, 1)
   * out 0  :  top[0] : match infotmation:  (old-batch-size, 1, 1, k)
*/
template<typename xpu>
class MergeWindowLayer : public Layer<xpu>{
 public:
  MergeWindowLayer(LayerType type) { this->layer_type = type; }
  virtual ~MergeWindowLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    this->defaults["dim"] = SettingV(3);
    this->defaults["max_len"] = SettingV(100);
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MergeWindowLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MergeWindowLayer:top size problem.");
    max_len   = setting["max_len"].iVal();
    dim   = setting["dim"].iVal();
  }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MergeWindowLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MergeWindowLayer:top size problem.");
    Tensor2D bottom0_length = bottom[0]->length;

    int idim = max_len;
    for(index_t i = 0 ; i < bottom[1]->data.size(0); ++ i){
        utils::Check(bottom[1]->data[i][0][0][0] > 0,"MergeWindowLayer:: Reshape error, window size equals zero.");
        idim = idim > bottom[1]->data[i][0][0][0] ? idim : bottom[1]->data[i][0][0][0];
    }
    utils::Check(idim > 0, "MergeWindowLayer:: Reshape idim problem.");
    int batch_size  = bottom[1]->data.size(0);

    assert(bottom[0]->data.size(dim) == 1);
    if(dim == 1){
        top[0]->Resize(batch_size, idim, bottom[0]->data.size(2), bottom[0]->data.size(3), true);
    }else if(dim == 2){
        top[0]->Resize(batch_size, bottom[0]->data.size(1), idim, bottom[0]->data.size(3), true);
    }else if(dim == 3){
        top[0]->Resize(batch_size, bottom[0]->data.size(1), bottom[0]->data.size(2), idim, true);
    }else{
        utils::Check(0,"merge_2_window_data_layer::dim setting wrong,must be 1|2|3.");
    }
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
    Tensor1D bottom0_data1 = bottom[0]->data_d1();
    Tensor2D top0_data2 = top[0]->data_d2();
    //printf("bottom[0]:%d,%d,%d,%d\n",bottom[0]->data.size(0),bottom[0]->data.size(1),bottom[0]->data.size(2),bottom[0]->data.size(3));
    utils::Check(bottom[0]->data.size(dim) == 1, " MergeWindowLayer:: bottom[%d] size problem.",dim);

    top0_data = 0.0;
    int batch_size = top0_data.size(0);
    index_t k = 0;
    for(index_t i = 0 ; i < batch_size; ++ i){
        int curr_len = bottom[1]->data[i][0][0][0];
        utils::Check(curr_len > 0," MergeWindowLayer: curr_len must be above 0.");
        top0_length[i][0] = curr_len;
        if(dim == 1){
            for(index_t j = 0 ; j < curr_len; ++ j){
                top0_data[i][j] = mshadow::expr::F<op::identity>(bottom[0]->data[k][0]);
                ++ k;
            }
        }else if(dim == 2){
            for(index_t j = 0 ; j < curr_len; ++ j){
                for(index_t m = 0 ; m < bottom[0]->data.size(1); ++ m){
                    top0_data[i][m][j] = F<op::identity>(bottom[0]->data[k][m][0]);
                }
                ++ k;
            }
        }else if(dim == 3){
          if(bottom[0]->data.size(1) == 1 && bottom[0]->data.size(2) == 1){
            //top0_data[i].Slice(0,curr_len) = F<op::identity>(bottom[0]->data.Slice(k,curr_len));
            top0_data2[i].Slice(0,curr_len) = F<op::identity>(bottom0_data1.Slice(k,k + curr_len));
            k += curr_len;
          }else{
            for(index_t j = 0 ; j < curr_len; ++ j){
              for(index_t m = 0 ; m < bottom[0]->data.size(1); ++ m){
                for(index_t n = 0 ; n < bottom[0]->data.size(2); ++ n){
                  top0_data[i][m][n][j] = bottom[0]->data[k][m][n][0];
                }
              }
              ++ k;
            }
          }
        }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    if(!this->prop_error[0] && !this->prop_error[1]) return;
    using namespace mshadow::expr;
    Tensor4D top0_diff = top[0]->diff;
    index_t k = 0;
    for(index_t i = 0 ; i < top0_diff.size(0); ++ i){ // copy query diff info
        int curr_len = bottom[1]->data[i][0][0][0];
        if(dim == 1){
            for(index_t j = 0 ; j < curr_len; ++ j){
                bottom[0]->diff[k][0] += F<op::identity>(top0_diff[i][j]);
                ++k;
            }
        }else if(dim == 2){
            for(index_t j = 0 ; j < curr_len; ++ j){
                for(index_t m = 0 ; m < bottom[0]->data.size(1); ++ m){
                    bottom[0]->diff[k][m][0] += F<op::identity>(top0_diff[i][m][j]);
                }
                ++k;
            }
        }else if(dim == 3){
            for(index_t j = 0 ; j < curr_len; ++ j){
                for(index_t m = 0 ; m < bottom[0]->data.size(1); ++ m){
                    for(index_t n = 0 ; n < bottom[0]->data.size(2); ++ n){
                        bottom[0]->diff[k][m][n][0] += top0_diff[i][m][n][j];
                    }
                }
                ++k;
            }
        }
    }
  }
  
 protected:
  int dim;
  int max_len;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MAP_2_TEXTDATA_LAYER_INL_HPP_

