#ifndef TEXTNET_LAYER_NODE_H_
#define TEXTNET_LAYER_NODE_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include <mshadow/tensor_container.h>
#include "op.h"
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
  // store the Node data
  mshadow::TensorContainer<xpu, 4> data;
  // store the Node diff
  mshadow::TensorContainer<xpu, 4> diff;
  // store the Node idx, if node is sparse
  mshadow::TensorContainer<xpu, 1> idx;
  // store sentence lenth
  mshadow::TensorContainer<xpu, 2> length;
  // to show whether Node has sparse diff
  bool is_sparse;
  // always true
  bool must_contiguous;
  bool inited_data;
  bool inited_diff;
  // for sharing parameters, if true, will not use local data and diff
  bool is_share;
  std::string node_name;
  int node_idx;

  // set this to false if we only need data 
  bool need_diff;

  // Updater interface
  updater::Updater<xpu, 4>* updater_;
  // Initializer interface
  initializer::Initializer<xpu, 4>* initializer_;

  // for sharing parameters, a pointer to the master node
  Node *master; 

  // constructor
  Node(bool need_diff_ = true) : must_contiguous(true), need_diff(need_diff_) {
    // data.shape_ = mshadow::Shape4(0,0,0,0);
    // diff.shape_ = mshadow::Shape4(0,0,0,0);
    // data.set_pad(false);
    // diff.set_pad(false);
    inited_data = false;
    inited_diff = false;
    is_share = false;
	is_sparse = false;
	updater_ = NULL;
    initializer_ = NULL;
    master = NULL;
    node_idx = -1;
  }
  
  inline void FreeSpace(void) {
    if (inited_data){
      mshadow::FreeSpace(&data);
      mshadow::FreeSpace(&length);
    }
    if (need_diff && inited_diff){
      mshadow::FreeSpace(&diff);
    }
	if (need_diff && is_sparse && inited_diff){
	  mshadow::FreeSpace(&idx);
	}
  }

  inline void PrintShape(std::string text) {
	mshadow::Shape<4> s = data.shape_;
	mshadow::Shape<2> len_s = length.shape_;
	utils::Printf("\tNode:[%s](%s) \tdata (%d x %d x %d x %d) \tlen (%d x %d)\n", 
			node_name.c_str(), text.c_str(), s[0], s[1], s[2], s[3], len_s[0], len_s[1]); 
  }

  // Clear Data data 
  void ClearData(void) {
    if (is_share) return;
	utils::Check(inited_data, "Must init data before clear.");
	data = 0.f;
	length = -1.f;
  }

  // Clear Diff data
  void ClearDiff(void) {
    if (is_share) return;
	utils::Check(inited_diff || !need_diff, "Must init diff before clear.");
    if (is_sparse) {
	  // if is_sparse we need delete its shape
	  diff.Resize(mshadow::Shape4(0,0,0,0));
	  idx.Resize(mshadow::Shape1(0));
	} else {
      diff = 0.f;
    }
  }

  // Share with other node
  void Share(Node &other) {
    is_share = true;
	is_sparse = other.is_sparse;
    data.Resize(mshadow::Shape4(0,0,0,0));
    diff.Resize(mshadow::Shape4(0,0,0,0));
    idx.Resize(mshadow::Shape1(0));
    length.Resize(mshadow::Shape2(0,0));
    // assert(!is_sparse);
    // utils::Check(!is_sparse, "Node: sparse parameter sharing is not supported yet");
    
    // use tensor container as a tensor without realloc space
    (*(mshadow::Tensor<xpu, 4> *)&data)   = other.data;
    (*(mshadow::Tensor<xpu, 2> *)&length) = other.length;
    if (!is_sparse)  {
      (*(mshadow::Tensor<xpu, 4> *)&diff) = other.diff;
      (*(mshadow::Tensor<xpu, 1> *)&idx)  = other.idx;
    }
    must_contiguous = other.must_contiguous;
    inited_data = false; // main node take charge of this
    inited_diff = false; // main node take charge of this
    node_name = other.node_name;
    node_idx = other.node_idx;
    need_diff = other.need_diff;
    
    updater_ = NULL;     // main node take charge of this
    initializer_ = NULL; // main node take charge of this 
    master = &other;
  }
 
  inline void Resize(int d1, int d2, int d3, int d4, bool init=false) {
    utils::Check(!is_share, "Node: Share node does not manage memory.");
    mshadow::Shape<4> new_size = mshadow::Shape4(d1, d2, d3, d4);
    Resize(new_size, init);
  }
  
  inline void Resize(int d1, int d2, int d3, int d4, int len_d1, int len_d2, bool init=false) {
    utils::Check(!is_share, "Node: Share node does not manage memory.");
    mshadow::Shape<4> new_size = mshadow::Shape4(d1, d2, d3, d4);
	mshadow::Shape<2> len_new_size = mshadow::Shape2(len_d1, len_d2);
    Resize(new_size, len_new_size, init);
  }

  inline void Resize(mshadow::Shape<4> new_size, bool init=false) {
    utils::Check(!is_share, "Node: Share node does not manage memory.");
	mshadow::Shape<2> len_new_size = mshadow::Shape2(new_size[0], new_size[1]);
    Resize(new_size, len_new_size, init);
  }

  inline void Resize(mshadow::Shape<4> new_size, mshadow::Shape<2> len_new_size, bool init=false) {
    utils::Check(!is_share, "Node: Share node does not manage memory.");
    if (4 == data.shape_.kDimension && new_size == data.shape_ && !init) {
      // do nothing
    } else if (init) {
      data.Resize(new_size, 0.0);
      length.Resize(mshadow::Shape2(len_new_size[0], len_new_size[1]), -1.f);
      inited_data = true;
      if (need_diff) {
        diff.Resize(new_size, 0.0);
        inited_diff = true;
      }
    } else {
      data.Resize(new_size);
      length.Resize(mshadow::Shape2(len_new_size[0], len_new_size[1]));
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
  // this load by ssv (space split) format, not json format, the first line is the shape info
  void LoadDataSsv(const char *data_file) {
    utils::Printf("Open data file: %s\n", data_file);     
    std::vector<std::string> lines; 
    std::ifstream fin(data_file); 
    std::string s; 
    utils::Check(fin, "Open data file problem."); 
    while (!fin.eof()) { 
      std::getline(fin, s);
      if (s.empty()) break;
      lines.push_back(s);
    }
    fin.close();
    std::istringstream iss;
    int row = 0, col = 0;
    iss.str(lines[0]);
    iss >> row >> col;
    utils::Check(row == lines.size()-1, "Data error 1.");
    utils::Check(this->data.shape_[0] == 1, "Data error 2.");
    utils::Check(this->data.shape_[1] == 1, "Data error 3.");
    utils::Check(this->data.shape_[2] == row, "Data error 4.");
    utils::Check(this->data.shape_[3] == col, "Data error 5.");

    for (int i = 0; i < row; ++i) {
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(lines[i+1]);
      for (int j = 0; j < col; ++j) {
        iss >> this->data[0][0][i][j];
      }
    }
  }

  void LoadNode(Json::Value &node_root, bool with_diff = false) {
    Json::Value data_root = node_root["data"];
    int s0 = data_root["shape"][0].asInt();
    int s1 = data_root["shape"][1].asInt();
    int s2 = data_root["shape"][2].asInt();
    int s3 = data_root["shape"][3].asInt();
    mshadow::Shape<4> shape = mshadow::Shape4(s0, s1, s2, s3);
    if (!(shape == data.shape_)) {
      Resize(shape);
    }
    int size = s0*s1*s2*s3;
    for (int i = 0; i < size; ++i) {
      data.dptr_[i] = data_root["value"][i].asFloat();
    }
    
    // if doesn't need diff just jump out
	if (!with_diff) return;
    utils::Check(need_diff, "Node: try to load diff but without need_diff");
    Json::Value diff_root = node_root["diff"];
    s0 = diff_root["shape"][0].asInt();
    s1 = diff_root["shape"][1].asInt();
    s2 = diff_root["shape"][2].asInt();
    s3 = diff_root["shape"][3].asInt();
    shape = mshadow::Shape4(s0, s1, s2, s3);
    if (!(shape == diff.shape_)) {
      Resize(shape);
    }
    size = s0*s1*s2*s3;
    for (int i = 0; i < size; ++i) {
      diff.dptr_[i] = diff_root["value"][i].asFloat();
    }
  }

  float AbsMean(float *p, size_t size) {
    utils::Check(size > 0, "Node: mean size error.");
    float sum = 0;
    for (size_t i = 0; i < size; ++i) {
      if (p[i] > 0) {
        sum += p[i];
      } else {
        sum -= p[i];
      }
    }
    return sum/size;
  }

  float AbsMax(float *p, size_t size) {
    utils::Check(size > 0, "Node: max size error.");
    float max = -10000.f;
    for (size_t i = 0; i < size; ++i) {
      float v = p[i]>0?p[i]:-p[i];
      if (max < v) 
        max = v;
    }
    return max;
  }

  void PrintStatistic(string prefix) {
    float data_mean = 0.f, data_max = 0.f, diff_mean = 0.f, diff_max = 0.f;
    data_mean = AbsMean(data.dptr_, data.shape_.Size());
    data_max  = AbsMax(data.dptr_, data.shape_.Size());
    diff_mean = AbsMean(diff.dptr_, diff.shape_.Size());
    diff_max  = AbsMax(diff.dptr_,  diff.shape_.Size());
    cout << "Stat of " << prefix << ":" << data_mean << ":" \
         << diff_mean << ":" << data_max << ":" << diff_max << endl;
  }

  Json::Value data_statistic(Json::Value &req_root) {
	using namespace mshadow::expr;
    Json::Value rtn_root;
	for (int i = 0; i < req_root.size(); ++i) {
      std::string req_tag = req_root[i].asString();
	  if (req_tag == "mean") {
		// rtn_root[req_tag] = sum_rows(data_d1())[0] / data.shape_.Size();
		float rtn = 0.0f;
		for (int j = 0; j < data.shape_.Size(); ++j) {
          rtn += data_d1()[j];
		}
		rtn /= data.shape_.Size();
		rtn_root[req_tag] = rtn;
	  } else if (req_tag == "var") {
		if (!rtn_root["mean"]) {
		  float rtn = 0.0f;
		  for (int j = 0; j < data.shape_.Size(); ++j) {
            rtn += data_d1()[j];
		  }
		  rtn /= data.shape_.Size();
		  rtn_root["mean"] = rtn;
	      // rtn_root["mean"] = sum_rows(data_d1());
		}
        //rtn_root[req_tag] = sum_rows(F<op::square>(data_d1()))[0] / data.shape_.Size() 
		//	- rtn_root["mean"].asFloat();
		float rtn = 0.0f;
		for (int j = 0; j < data.shape_.Size(); ++j) {
          rtn += data_d1()[j]*data_d1()[j];
		}
		rtn_root[req_tag] = rtn / data.shape_.Size() - rtn_root["mean"].asFloat()*rtn_root["mean"].asFloat();
	  } else if (req_tag == "min") {
        
	  } else if (req_tag == "max") {

	  }
	}
	return rtn_root;
  }

  Json::Value diff_statistic(Json::Value &req_root) {
 	using namespace mshadow::expr;
    Json::Value rtn_root;
	for (int i = 0; i < req_root.size(); ++i) {
      std::string req_tag = req_root[i].asString();
	  if (req_tag == "mean") {
		// rtn_root[req_tag] = sum_rows(data_d1())[0] / data.shape_.Size();
		float rtn = 0.0f;
		for (int j = 0; j < diff.shape_.Size(); ++j) {
          rtn += diff_d1()[j];
		}
		rtn /= diff.shape_.Size();
		rtn_root[req_tag] = rtn;
	  } else if (req_tag == "var") {
		if (!rtn_root["mean"]) {
		  float rtn = 0.0f;
		  for (int j = 0; j < diff.shape_.Size(); ++j) {
            rtn += diff_d1()[j];
		  }
		  rtn /= diff.shape_.Size();
		  rtn_root["mean"] = rtn;
	      // rtn_root["mean"] = sum_rows(data_d1());
		}
        //rtn_root[req_tag] = sum_rows(F<op::square>(data_d1()))[0] / data.shape_.Size() 
		//	- rtn_root["mean"].asFloat();
		float rtn = 0.0f;
		for (int j = 0; j < diff.shape_.Size(); ++j) {
          rtn += diff_d1()[j]*diff_d1()[j];
		}
		rtn_root[req_tag] = rtn / diff.shape_.Size() - rtn_root["mean"].asFloat()*rtn_root["mean"].asFloat();
	  } else if (req_tag == "min") {
        
	  } else if (req_tag == "max") {

	  }
	}
	return rtn_root;  
  }

  inline mshadow::Tensor<xpu, 1> data_d1() {
    return mshadow::Tensor<xpu, 1>(data.dptr_, mshadow::Shape1(data.shape_.Size()));
  }

  inline mshadow::Tensor<xpu, 1> diff_d1() {
    return mshadow::Tensor<xpu, 1>(diff.dptr_, mshadow::Shape1(diff.shape_.Size()));
  }

  inline mshadow::Tensor<xpu, 1> data_d1_reverse() {
    return mshadow::Tensor<xpu, 1>(data.dptr_, mshadow::Shape1(data.shape_.Size()));
  }

  inline mshadow::Tensor<xpu, 1> diff_d1_reverse() {
    return mshadow::Tensor<xpu, 1>(diff.dptr_, mshadow::Shape1(diff.shape_.Size()));
  }

  inline mshadow::Tensor<xpu, 1> idx_d1() {
    return mshadow::Tensor<xpu, 1>(idx.dptr_, mshadow::Shape1(idx.shape_.Size()));
  }

  inline mshadow::Tensor<xpu, 1> length_d1() {
    return mshadow::Tensor<xpu, 1>(length.dptr_, mshadow::Shape1(length.shape_.Size()));
  }
 
  inline mshadow::Tensor<xpu, 2> data_d2() {
    mshadow::Shape<4> s = data.shape_;
    index_t  ymax = s[1]*s[2]*s[3];
    return mshadow::Tensor<xpu, 2>(data.dptr_, mshadow::Shape2(s[0], ymax));
  }

  inline mshadow::Tensor<xpu, 2> data_d2_middle() {
    mshadow::Shape<4> s = data.shape_;
    index_t  xmax = s[0]*s[1];
    index_t  ymax = s[2]*s[3];
    return mshadow::Tensor<xpu, 2>(data.dptr_, mshadow::Shape2(xmax, ymax));
  }

  inline mshadow::Tensor<xpu, 2> data_d2_reverse() {
    mshadow::Shape<4> s = data.shape_;
    index_t  xmax = s[0]*s[1]*s[2];
    return mshadow::Tensor<xpu, 2>(data.dptr_, mshadow::Shape2(xmax, s[3]));
  }

  inline mshadow::Tensor<xpu, 2> diff_d2() {
    mshadow::Shape<4> s = diff.shape_;
    index_t  ymax = s[1]*s[2]*s[3];
    return mshadow::Tensor<xpu, 2>(diff.dptr_, mshadow::Shape2(s[0], ymax));
  }

  inline mshadow::Tensor<xpu, 2> diff_d2_middle() {
    mshadow::Shape<4> s = diff.shape_;
    index_t  xmax = s[0]*s[1];
    index_t  ymax = s[2]*s[3];
    return mshadow::Tensor<xpu, 2>(diff.dptr_, mshadow::Shape2(xmax, ymax));
  }

  inline mshadow::Tensor<xpu, 2> diff_d2_reverse() {
    mshadow::Shape<4> s = diff.shape_;
    index_t  xmax = s[0]*s[1]*s[2];
    return mshadow::Tensor<xpu, 2>(diff.dptr_, mshadow::Shape2(xmax, s[3]));
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
    if (!is_share) {
      initializer_->DoInitialize(data);
      if (init_diff) {
        initializer_->DoInitialize(diff);
      }
    }
  }
  
  inline void Update() {
    if (!is_share) {
      if (is_sparse) {
        updater_->UpdateSparse(data, diff, idx);
      } else {
        updater_->Update(data, diff);
      }
    }
  }
  void sparseAdd(mshadow::TensorContainer<xpu, 4> &l_data, 
                 mshadow::TensorContainer<xpu, 1> &l_idx, 
                 mshadow::TensorContainer<xpu, 4> &r_data, 
                 mshadow::TensorContainer<xpu, 1> &r_idx,
                 mshadow::TensorContainer<xpu, 4> &merge_data,
                 mshadow::TensorContainer<xpu, 1> &merge_idx) {
    utils::Assert(l_data.size(2) == 1 && r_data.size(2) == 1, "Merge Sparse Tensor: size problem");
    utils::Assert(l_data.size(3) == 1 && r_data.size(3) == 1, "Merge Sparse Tensor: size problem");
    utils::Assert(l_data.size(1) == r_data.size(1), "Merge Sparse Tensor: size problem");

    std::map<int, int> idx_map;
    int inc = 0;
    for (int i = 0; i < l_idx.size(0); ++i) {
      int w_idx = l_idx[i];
      if (!idx_map.count(w_idx)) {
        idx_map[w_idx] = inc++;
      }
    }
    for (int i = 0; i < r_idx.size(0); ++i) {
      int w_idx = r_idx[i];
      if (!idx_map.count(w_idx)) {
        idx_map[w_idx] = inc++;
      }
    }

    int feat_size = l_data.size(1);
    merge_data.Resize(mshadow::Shape4(inc, feat_size, 1, 1), 0);
    merge_idx.Resize(mshadow::Shape1(inc), 0);
    for (std::map<int,int>::iterator it=idx_map.begin(); it!=idx_map.end(); ++it) {
      merge_idx[it->second] = it->first;
    }
    for (int i = 0; i < l_data.size(0); ++i) {
      merge_data[idx_map[l_idx[i]]] += l_data[i];
    }
    for (int i = 0; i < r_data.size(0); ++i) {
      merge_data[idx_map[r_idx[i]]] += r_data[i];
    }
  }

  void CutOffGradient(float maxVal) {
    utils::Check(maxVal >= 0.f, "Node: cut off gradient error.");
    utils::Check(!is_share, "Node: cut off gradient error.");
    utils::Check(!is_sparse, "Node: cut off gradient error.");
    size_t size = diff.shape_.Size();
    for (size_t i = 0; i < size; ++i) {
      float *p = diff.dptr_ + i;
      if (*p > maxVal) {
        *p = maxVal;
      } else if (*p < -maxVal) {
        *p = -maxVal;
      }
    }
  }

  void sparseAdd2Left(mshadow::TensorContainer<xpu, 4> &l_data, 
                      mshadow::TensorContainer<xpu, 1> &l_idx, 
                      mshadow::TensorContainer<xpu, 4> &r_data, 
                      mshadow::TensorContainer<xpu, 1> &r_idx) {
    mshadow::TensorContainer<xpu, 4> merge_data;
    mshadow::TensorContainer<xpu, 1> merge_idx;
    sparseAdd(l_data, l_idx, r_data, r_idx, merge_data, merge_idx);
    l_data = merge_data;
    l_idx  = merge_idx;
  }

}; // struct Node

}  // namespace layer
}  // namespace textnet
#endif  // CXXNET_LAYER_NODE_H
