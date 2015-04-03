#ifndef TEXTNET_LAYER_NODE_H_
#define TEXTNET_LAYER_NODE_H_

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include <mshadow/tensor_container.h>
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../initializer/initializer.h"
#include "../updater/updater.h"
#include "../io/json/json.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of layer defintiion */
namespace layer {
template<typename xpu>
struct Node {
  mshadow::TensorContainer<xpu, 4> data;
  mshadow::TensorContainer<xpu, 4> diff;
  mshadow::TensorContainer<xpu, 1> idx;
  bool must_contiguous;
  bool inited_data;
  bool inited_diff;
  std::string node_name;
  int node_idx;
  bool need_diff;
  updater::Updater<xpu, 4>* updater_;
  initializer::Initializer<xpu, 4>* initializer_;
  
  // constructor
  Node(bool need_diff_ = true) : must_contiguous(false), need_diff(need_diff_) {
    data.shape_ = mshadow::Shape4(0,0,0,0);
    diff.shape_ = mshadow::Shape4(0,0,0,0);
    // data.set_pad(false);
    // diff.set_pad(false);
    inited_data = false;
    inited_diff = false;
  }
  
  inline void FreeSpace(void) {
    if (inited_data){
      mshadow::FreeSpace(&data);
    }
    if (inited_diff){
      mshadow::FreeSpace(&diff);
    }
  }

  inline void PrintShape(std::string text) {
	mshadow::Shape<4> s = data.shape_;
	// utils::Printf("\t%s Shape: %d x %d x %d x %d\n", text.c_str(), s[0], s[1], s[2], s[3]);
  }

  inline void Resize(int d1, int d2, int d3, int d4, bool init=false) {
    mshadow::Shape<4> new_size = mshadow::Shape4(d1, d2, d3, d4);
    if (4 == data.shape_.kDimension && new_size == data.shape_ && !init) {
      // do nothing
    } else if (init) {
      data.Resize(new_size, 0.0);
      inited_data = true;
      if (need_diff) {
        diff.Resize(new_size, 0.0);
        inited_diff = true;
      }
    } else {
      data.Resize(new_size);
      inited_data = true;
      if (need_diff) {
        diff.Resize(new_size);
        inited_diff = true;
      }
    }
  }
  
  inline void Resize(mshadow::Shape<4> new_size, bool init=false) {
    if (4 == data.shape_.kDimension && new_size == data.shape_ && !init) {
      // do nothing
    } else if (init) {
      data.Resize(new_size, 0.0);
      inited_data = true;
      if (need_diff) {
        diff.Resize(new_size, 0.0);
        inited_diff = true;
      }
    } else {
      data.Resize(new_size);
      inited_data = true;
      if (need_diff) {
        diff.Resize(new_size);
        inited_diff = true;
      }
    }
  }
  
  void SaveNode(Json::Value &node_root, bool with_diff = false) {
    Json::Value data_root;
	Json::Value diff_root;
    Json::Value data_shape_root;
	Json::Value diff_shape_root;
	Json::Value data_value_root;
	Json::Value diff_value_root;

	// Save data Tensor in Node
    mshadow::Shape<4> shape = data.shape_;
	for (int i = 0; i < 4; ++i) {
	  data_shape_root.append(shape[i]);
	}
    for (int i = 0; i < data.shape_.Size(); ++i) {
	  data_value_root.append(data.dptr_[i]);
	}
	data_root["shape"] = data_shape_root;
	data_root["value"] = data_value_root;

	node_root["data"] = data_root;

	// if doesn't need diff just jump out
	if (!with_diff) return;

	// Save diff Tensor in Node
    shape = diff.shape_;
	for (int i = 0; i < 4; ++i) {
	  diff_shape_root.append(shape[i]);
	}
    for (int i = 0; i < diff.shape_.Size(); ++i) {
	  diff_value_root.append(diff.dptr_[i]);
	}
	diff_root["shape"] = diff_shape_root;
	diff_root["value"] = diff_value_root;

	node_root["diff"] = diff_root;
  }

  void LoadNode(Json::Value &node_root, bool with_diff = false) {

  }

  inline mshadow::Tensor<xpu, 1> data_d1() {
    return mshadow::Tensor<xpu, 1>(data.dptr_, mshadow::Shape1(data.shape_.Size()));
  }

  inline mshadow::Tensor<xpu, 1> diff_d1() {
    return mshadow::Tensor<xpu, 1>(diff.dptr_, mshadow::Shape1(diff.shape_.Size()));
  }

  inline mshadow::Tensor<xpu, 1> idx_d1() {
    return mshadow::Tensor<xpu, 1>(idx.dptr_, mshadow::Shape1(idx.shape_.Size()));
  }
  
  inline mshadow::Tensor<xpu, 2> data_d2() {
    mshadow::Shape<4> s = data.shape_;
    index_t  ymax = s[1]*s[2]*s[3];
    return mshadow::Tensor<xpu, 2>(data.dptr_, mshadow::Shape2(s[0], ymax));
  }

  inline mshadow::Tensor<xpu, 2> diff_d2() {
    mshadow::Shape<4> s = diff.shape_;
    index_t  ymax = s[1]*s[2]*s[3];
    return mshadow::Tensor<xpu, 2>(diff.dptr_, mshadow::Shape2(s[0], ymax));
  }
 
  inline mshadow::Tensor<xpu, 3> data_d3() {
    mshadow::Shape<4> s = data.shape_;
    index_t  ymax = s[2]*s[3];
    return mshadow::Tensor<xpu, 3>(data.dptr_, mshadow::Shape3(s[0], s[1], ymax));
  }

  inline mshadow::Tensor<xpu, 3> diff_d3() {
    mshadow::Shape<4> s = diff.shape_;
    index_t  ymax = s[2]*s[3];
    return mshadow::Tensor<xpu, 3>(diff.dptr_, mshadow::Shape3(s[0], s[1], ymax));
  }
  
  inline void Init(bool init_diff = false) {
    initializer_->DoInitialize(data);
    if (init_diff) {
      initializer_->DoInitialize(diff);
    }
  }
  
  inline void Update() {
    if (updater_->is_sparse) {
      updater_->UpdateSparse(data, diff, idx);
    } else {
      updater_->Update(data, diff);
    }
  }

}; // struct Node

}  // namespace layer
}  // namespace textnet
#endif  // CXXNET_LAYER_NODE_H
