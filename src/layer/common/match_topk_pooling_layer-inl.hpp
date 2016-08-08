#ifndef TEXTNET_LAYER_MATCH_TOPK_POOLING_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_TOPK_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include <algorithm>

namespace textnet {
namespace layer {

template<typename xpu>
class MatchTopKPoolingLayer : public Layer<xpu>{
 public:
  MatchTopKPoolingLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchTopKPoolingLayer(void) {}
  
  virtual int BottomNodeNum() { return nbottom; } // matrix node, l_seq for length, r_seq for length
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    // this->defaults["dim"] = SettingV(2);
    // this->defaults["is_constraint"] = SettingV(false);

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["k"] = SettingV();

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    k = setting["k"].iVal();
    nbottom = bottom.size();
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "MatchTopKPoolingLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchTopKPoolingLayer: top size problem.");

    mshadow::Shape<4> shape_out  = bottom[0]->data.shape_;
    shape_out[2] = k;
    shape_out[3] = 1;
    top[0]->Resize(shape_out, true);

    shape_out[3] = 2;
    pos.Resize(shape_out, true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        if(nbottom == 3){
          bottom[1]->PrintShape("bottom1");
          bottom[2]->PrintShape("bottom2");
        }
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;

    if (top[0]->data.shape_[0] != bottom[0]->data.shape_[0]) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }
 
    
  typedef mshadow::Tensor<xpu,2> Tensor2D;
  typedef mshadow::Tensor<xpu,2,int> Tensor2DInt;

  struct ValidIdx {
    float val;
    int x, y;
    ValidIdx(void) {
      x = 0; y = 0; val = 0.f;
    }
  };

  class CmpVal {
    public:
      bool operator() (const ValidIdx &l, const ValidIdx &r) const {
         return l.val > r.val;
      }
  };
          
  // bool cmp_val(const ValidIdx &l, const ValidIdx &r) {
  //   return l.val > r.val;
  // }

  void pooling_one_matrix(Tensor2D t_in, Tensor2D t_out,
                          int input_row,  int input_col,
                          int k, Tensor2DInt pos) {
    utils::Check(t_in.size(0) >= input_row && t_in.size(1) >= input_col, "MatchTopKPoolingLayer: size error.");
    if(input_row * input_col < k){
        printf("input_row:%d,input_col:%d,k:%d\n",input_row,input_col,k);
    }
    utils::Check(input_row * input_col >= k, "MatchTopKPoolingLayer: size error.");
    utils::Check(pos.size(0) == k && pos.size(1) == 2, "MatchTopKPoolingLayer: size error.");
    utils::Check(t_out.size(0) == k && t_out.size(1) == 1, "MatchTopKPoolingLayer: size error.");

    vector<ValidIdx> all;
    int iindex = 0;
    for (int i = 0; i < input_row; ++i) {
      for (int j = 0; j < input_col; ++j) {
        ValidIdx val_idx;
        val_idx.val = t_in[i][j];
        val_idx.x = i;
        val_idx.y = j;
        all.push_back(val_idx);
        ++iindex;
      }
    }
    /*
    while(iindex < k){
        ValidIdx val_idx;
        val_idx.val = 0;
        val_idx.x = -1;
        val_idx.y = -1;
        ++iindex;
        all.push_back(val_idx);
    }
    assert(all.size() >= k);
    */
    std::sort(all.begin(), all.end(), CmpVal());
        
    for (int i = 0; i < k; ++i) {
      t_out[i][0] = all[i].val;
      pos[i][0] = all[i].x;
      pos[i][1] = all[i].y;
    }
  }

  void unpooling_one_matrix(Tensor2D t_in, Tensor2D t_out,
                            int k, Tensor2DInt pos) {
    for (int i = 0; i < k; ++i) {
      int x = pos[i][0];
      int y = pos[i][1];
      //if( x == -1 || y == -1) continue;
      t_in[x][y] += t_out[i][0];
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data  = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;

    top_data = 0;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      for (index_t channel_idx = 0; channel_idx < bottom_data.size(1); ++channel_idx) {
        int len_l = bottom_data[batch_idx][channel_idx].size(0);
        int len_r = bottom_data[batch_idx][channel_idx].size(1);
        if(nbottom == 3){
          len_l = bottom[1]->length[batch_idx][0];
          len_r = bottom[2]->length[batch_idx][0];
        }
        pooling_one_matrix(bottom_data[batch_idx][channel_idx], top_data[batch_idx][channel_idx],
                           len_l, len_r, k, pos[batch_idx][channel_idx]);
        //printf("label:%d\tbatch_idx:%d\tchannel_idx:%d\n",batch_idx%2,batch_idx,channel_idx);
        //for(int i = 0 ; i < k; ++ i){
            //printf(" %.4f",top_data[batch_idx][channel_idx][1][i]);
        //}
        //printf("\n");
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff  = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_diff     = top[0]->diff;

    for (index_t batch_idx = 0; batch_idx < bottom_diff.size(0); ++batch_idx) {
      for (index_t channel_idx = 0; channel_idx < bottom_diff.size(1); ++channel_idx) {
        unpooling_one_matrix(bottom_diff[batch_idx][channel_idx], top_diff[batch_idx][channel_idx],
                             k, pos[batch_idx][channel_idx]);
      }
    }
  }

 protected:
  mshadow::TensorContainer<xpu, 4, int> pos;
  int k;
  int nbottom;
};
}  // namespace layer
}  // namespace textnet
#endif  
