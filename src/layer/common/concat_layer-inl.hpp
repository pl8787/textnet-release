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
    
    // this is to concat on dim 2 by length, for var length cases
    // mutiple sentences with different lengths are concatenated into a consecutive memory
    // all lengths will be sumed to a new length
    this->defaults["is_concat_by_length"] = SettingV(false);

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["bottom_node_num"] = SettingV();
    this->defaults["concat_dim_index"] = SettingV(); // 0, 1, 2, 3
    
    Layer<xpu>::Require();
  }

  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    nBottomNode = setting["bottom_node_num"].i_val;
    concat_dim_index = setting["concat_dim_index"].i_val;
    is_concat_by_length = setting["is_concat_by_length"].b_val;
    utils::Check(concat_dim_index < 4, "ConcatLayer: setting problem."); 
    utils::Check((!is_concat_by_length) || concat_dim_index == 2, "ConcatLayer: length is used only on dim 2");
  }
  
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "ConcatLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ConcatLayer:top size problem.");

    int out_size = 0;
    mshadow::Shape<4> shape_in_0 = bottom[0]->data.shape_;
    for (int i = 0; i < BottomNodeNum(); ++i) {
      mshadow::Shape<4> shape_in = bottom[i]->data.shape_;
      out_size += shape_in[concat_dim_index];
      for (int dim = 0; dim < 4; ++dim) {
        if (dim == concat_dim_index) 
            continue;
        utils::Check(shape_in[dim] == shape_in_0[dim], "ConcatLayer: bottom size problem");
      }
    }
    mshadow::Shape<4> shape_out = shape_in_0;
    shape_out[concat_dim_index] = out_size;
    if (concat_dim_index >= 2) {
      // this is a patch, length is set to the sampe with bottom 0
      top[0]->Resize(shape_out, bottom[0]->length.shape_, true);
    } else {
      top[0]->Resize(shape_out, true);
    }

	if (show_info) {
      for (int i = 0; i < nBottomNode; ++i) {
        bottom[i]->PrintShape("bottom_i");
      }
	  top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    int out_size = 0;
    mshadow::Shape<4> shape_in_0 = bottom[0]->data.shape_;
    for (int i = 0; i < BottomNodeNum(); ++i) {
      mshadow::Shape<4> shape_in = bottom[i]->data.shape_;
      out_size += shape_in[concat_dim_index];
      for (int dim = 0; dim < 4; ++dim) {
        if (dim == concat_dim_index) 
            continue;
        utils::Check(shape_in[dim] == shape_in_0[dim], "ConcatLayer: bottom size problem");
      }
    }

    if ( !(shape_in_0 == top[0]->data.shape_) ) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;

  void ConcatDim3(const std::vector<Node<xpu>*> &bottom,
                  const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_data = top[0]->data;
    top[0]->length = F<op::identity>(bottom[0]->length); // all bottom nodes must have the same length

    // i, j, m, n
    for (index_t i = 0; i < top_data.size(0); ++i) {
      for (index_t j = 0; j < top_data.size(1); ++j) {
        for (index_t m = 0; m < top_data.size(2); ++m) {
          int cnt = 0;
          for (index_t n = 0; n < BottomNodeNum(); ++n){
            Tensor1D t = bottom[n]->data[i][j][m];
            top_data[i][j][m].Slice(cnt, cnt+t.size(0)) = F<op::identity>(t);
            cnt += t.size(0);
          }
        }
      }
    }
  }

  void ConcatDim2(const std::vector<Node<xpu>*> &bottom,
                  const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    if (!is_concat_by_length) { // not support variable length
      Tensor4D top_data = top[0]->data;

      for (index_t i = 0; i < top_data.size(0); ++i) {
        for (index_t j = 0; j < top_data.size(1); ++j) {
          int cnt = 0;
          for (index_t n = 0; n < BottomNodeNum(); ++n){
            Tensor2D t = bottom[n]->data[i][j];
            top_data[i][j].Slice(cnt, cnt+t.size(0)) = F<op::identity>(t);
            cnt += t.size(0);
          }
        }
      }
    } else { // support variable length 
      Tensor4D top_data = top[0]->data;
      Tensor2D top_len  = top[0]->length;

      for (index_t i = 0; i < top_data.size(0); ++i) {
        for (index_t j = 0; j < top_data.size(1); ++j) {
          int cnt = 0;
          for (index_t n = 0; n < BottomNodeNum(); ++n){
            Tensor2D t = bottom[n]->data[i][j];
            int l = bottom[n]->length[i][j];
            top_data[i][j].Slice(cnt, cnt+l) = F<op::identity>(t.Slice(0, l));
            cnt += l;
          }
          top_len[i][j] = cnt;
        }
      }
    }
  }

  void ConcatDim1(const std::vector<Node<xpu>*> &bottom,
                  const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_data   = top[0]->data;
    Tensor2D top_length = top[0]->length;

    for (index_t i = 0; i < top_data.size(0); ++i) {
      int cnt = 0;
      for (index_t n = 0; n < BottomNodeNum(); ++n){
        Tensor3D t = bottom[n]->data[i];
        top_data[i].Slice(cnt, cnt+t.size(0)) = F<op::identity>(t);
        Tensor1D t_length = bottom[n]->length[i];
        top_length[i].Slice(cnt, cnt+t.size(0)) = F<op::identity>(t_length);
        cnt += t.size(0);
      }
    }
  }

  void ConcatDim0(const std::vector<Node<xpu>*> &bottom,
                  const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_data   = top[0]->data;
    Tensor2D top_length = top[0]->length;

    int cnt = 0;
    for (index_t n = 0; n < BottomNodeNum(); ++n){
      Tensor4D t = bottom[n]->data;
      top_data.Slice(cnt, cnt+t.size(0)) = F<op::identity>(t);
      Tensor2D t_length = bottom[n]->length;
      top_length.Slice(cnt, cnt+t.size(0)) = F<op::identity>(t_length);
      cnt += t.size(0);
    }
  }

  void SplitDim3(const std::vector<Node<xpu>*> &bottom,
                 const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_diff = top[0]->diff;

    for (index_t i = 0; i < top_diff.size(0); ++i) {
      for (index_t j = 0; j < top_diff.size(1); ++j) {
        for (index_t m = 0; m < top_diff.size(2); ++m) {
          int cnt = 0;
          for (index_t n = 0; n < BottomNodeNum(); ++n){
            Tensor1D t = bottom[n]->diff[i][j][m];
            t += top_diff[i][j][m].Slice(cnt, cnt+t.size(0));
            cnt += t.size(0);
          }
        }
      }
    }
  }

  void SplitDim2(const std::vector<Node<xpu>*> &bottom,
                 const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_diff = top[0]->diff;

    if (!is_concat_by_length) {
      for (index_t i = 0; i < top_diff.size(0); ++i) {
        for (index_t j = 0; j < top_diff.size(1); ++j) {
          int cnt = 0;
          for (index_t n = 0; n < BottomNodeNum(); ++n){
            Tensor2D t = bottom[n]->diff[i][j];
            t += top_diff[i][j].Slice(cnt, cnt+t.size(0));
            cnt += t.size(0);
          }
        }
      }
    } else {
      for (index_t i = 0; i < top_diff.size(0); ++i) {
        for (index_t j = 0; j < top_diff.size(1); ++j) {
          int cnt = 0;
          for (index_t n = 0; n < BottomNodeNum(); ++n) {
            Tensor2D t = bottom[n]->data[i][j];
            int l = bottom[n]->length[i][j];
            t.Slice(0,l) += top_diff[i][j].Slice(cnt, cnt+l);
            cnt += l;
          }
        }
      }
    }
  }

  void SplitDim1(const std::vector<Node<xpu>*> &bottom,
                 const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_diff = top[0]->diff;

    for (index_t i = 0; i < top_diff.size(0); ++i) {
      int cnt = 0;
      for (index_t n = 0; n < BottomNodeNum(); ++n){
        Tensor3D t = bottom[n]->diff[i];
        t += top_diff[i].Slice(cnt, cnt+t.size(0));
        cnt += t.size(0);
      }
    }
  }

  void SplitDim0(const std::vector<Node<xpu>*> &bottom,
                 const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor4D top_diff = top[0]->diff;

    int cnt = 0;
    for (index_t n = 0; n < BottomNodeNum(); ++n){
      Tensor4D t = bottom[n]->diff;
      t += top_diff.Slice(cnt, cnt+t.size(0));
      cnt += t.size(0);
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    if (concat_dim_index == 0) {
      ConcatDim0(bottom, top); 
    } else if (concat_dim_index == 1) {
      ConcatDim1(bottom, top); 
    } else if (concat_dim_index == 2) {
      ConcatDim2(bottom, top); 
    } else if (concat_dim_index == 3) {
      ConcatDim3(bottom, top); 
    } else {
      utils::Assert(false, "");
    } 
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    if (concat_dim_index == 0) {
      SplitDim0(bottom, top); 
    } else if (concat_dim_index == 1) {
      SplitDim1(bottom, top); 
    } else if (concat_dim_index == 2) {
      SplitDim2(bottom, top); 
    } else if (concat_dim_index == 3) {
      SplitDim3(bottom, top); 
    } else {
      utils::Assert(false, "");
    } 
  }
 protected:
  int nBottomNode, concat_dim_index;
  bool is_concat_by_length;
};
}  // namespace layer
}  // namespace textnet
#endif
