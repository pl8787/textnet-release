#ifndef TEXTNET_LAYER_TOPK_POOLING_LAYER_INL_HPP_
#define TEXTNET_LAYER_TOPK_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

// this layer selects the top k position by gate data
// then fills the top nodes with rep values by the position
template<typename xpu>
class TopkPoolingLayer : public Layer<xpu>{
 public:
  TopkPoolingLayer(LayerType type) { this->layer_type = type; }
  virtual ~TopkPoolingLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; } // a gate node and a representation node
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["k"] = SettingV(1);

    // require value, set to SettingV(),
    // it will force custom to set in config

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    k = setting["k"].iVal();
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "TopkPoolingLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "TopkPoolingLayer: top size problem.");

    mshadow::Shape<4> gate_shape = bottom[0]->data.shape_;
    mshadow::Shape<4> rep_shape  = bottom[1]->data.shape_;
    utils::Check(gate_shape[2] >= 1 && gate_shape[3] == 1, "TopkPoolingLayer: gate size problem.");
    utils::Check(gate_shape[0] == rep_shape[0], "TopkPoolingLayer: gate & rep do not match.");
    utils::Check(gate_shape[1] == rep_shape[1], "TopkPoolingLayer: gate & rep do not match.");
    utils::Check(gate_shape[2] == rep_shape[2], "TopkPoolingLayer: gate & rep do not match.");

    mshadow::Shape<4> top_shape = mshadow::Shape4(rep_shape[0], rep_shape[1], k, rep_shape[3]);
    mshadow::Shape<4> pos_shape = mshadow::Shape4(rep_shape[0], rep_shape[1], k, 1);
    top[0]->Resize(top_shape, true);
    pos.Resize(pos_shape, -1);

	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		bottom[1]->PrintShape("bottom1");
		top[0]->PrintShape("top0");
	}
  }

  typedef mshadow::Tensor<xpu,2> Tensor2D;
  typedef mshadow::Tensor<xpu,2,int> Tensor2DInt;

  // select top k positions from gate and keep the relative order
  void get_topk_pos(Tensor2D gate, Tensor2DInt pos, int length) {
    utils::Assert(gate.shape_[1] == 1 && pos.shape_[1] == 1 && pos.shape_[0] == k, 
                  "TopkPoolingLayer: gate & pos size error.");
    utils::Assert(length >= pos.shape_[0] && length <= gate.shape_[0], 
                  "TopkPoolingLayer: k exceeds the length, we have not consider this case yet.");
    
    std::vector<float> v(gate.dptr_, gate.dptr_+length);
    std::partial_sort(v.begin(), v.begin() + k, v.end(), std::greater<float>());
    float topk_threshold = v[k-1];

    std::vector<int> topk_pos;
    for (int i = 0; i < length; ++i) {
      if (gate[i][0] > topk_threshold) {
        topk_pos.push_back(i);
      }
    }
    utils::Assert(topk_pos.size() < k, "TopkPoolingLayer: sort error.");
    for (int i = 0; i < length; ++i) {
      if (gate[i][0] == topk_threshold) {
        topk_pos.push_back(i);
        if (topk_pos.size() == k) break;
      }
    }
    utils::Assert(topk_pos.size() == k, "TopkPoolingLayer: sort error.");
    std::sort(topk_pos.begin(), topk_pos.end());
    for (int i = 0; i < k; ++i) {
      pos[i][0] = topk_pos[i];
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> gate_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> rep_data  = bottom[1]->data;
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;

    // top[0]->length = k; // var len to static len

    top_data = 0;
    for (index_t batch_idx = 0; batch_idx < gate_data.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < gate_data.size(1); ++seq_idx) {
        int length = bottom_len[batch_idx][seq_idx]; 
        get_topk_pos(gate_data[batch_idx][seq_idx], pos[batch_idx][seq_idx], length);
        for (int i = 0; i < k; ++i) {
          top_data[batch_idx][seq_idx][i] = F<op::identity>(rep_data[batch_idx][seq_idx][pos[batch_idx][seq_idx][i][0]]);
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> rep_diff = bottom[1]->diff;

    for (index_t batch_idx = 0; batch_idx < top_diff.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < top_diff.size(1); ++seq_idx) {
        for (int i = 0; i < k; ++i) {
          rep_diff[batch_idx][seq_idx][pos[batch_idx][seq_idx][i][0]] += top_diff[batch_idx][seq_idx][i];
        }
      }
    }
  }
 protected:
  mshadow::TensorContainer<xpu, 4, int> pos;
  int k;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_WHOLEPOOLING_LAYER_INL_HPP_

