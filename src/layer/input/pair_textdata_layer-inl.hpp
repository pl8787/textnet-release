#ifndef TEXTNET_LAYER_PAIR_TEXTDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_PAIR_TEXTDATA_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include "stdlib.h"

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class PairTextDataLayer : public Layer<xpu>{
 public:
  PairTextDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~PairTextDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["min_doc_len"] = SettingV(1);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["data_file"] = SettingV();
    this->defaults["batch_size"] = SettingV();
    this->defaults["max_doc_len"] = SettingV();
    this->defaults["shuffle"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "TextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "TextDataLayer:top size problem.");

    data_file = setting["data_file"].sVal();
    batch_size = setting["batch_size"].iVal();
    max_doc_len = setting["max_doc_len"].iVal();
    min_doc_len = setting["min_doc_len"].iVal();
    shuffle = setting["shuffle"].bVal();
    
    ReadTextData();
	MakePair();
    
    line_ptr = 0;
  }

  void ReadTextData() {
    utils::Printf("Open data file: %s\n", data_file.c_str());    
    std::vector<std::string> lines;
    std::ifstream fin(data_file.c_str());
    std::string s;
    utils::Check(fin, "Open data file problem.");
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      lines.push_back(s);
    }
    fin.close();
    
    line_count = lines.size();

    std::istringstream iss;
	for (int i = 0; i < lines.size(); ++i) {
      int len_s1 = 0;
      int len_s2 = 0;
	  int label = -1;
	  int wid = -1;
	  vector<int> s1;
	  vector<int> s2;
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> label >> len_s1 >> len_s2;
	  label_set.push_back(label);
      for (int j = 0; j < len_s1; ++j) {
        iss >> wid;
        s1.push_back(wid);
      }
	  s1_data_set.push_back(s1);
      for (int j = 0; j < len_s2; ++j) {
        iss >> wid;
		s2.push_back(wid);
      }
	  s2_data_set.push_back(s2);
	}
    utils::Printf("Line count in file: %d\n", line_count);
  }

  void MakePair() {
    int list_count = 0;
	int cur_class = 0;
	vector<vector<int> > class_set;

	// for store the last list pair, so i<=line_count
	for (int i = 0; i <= line_count; ++i) {
	  if (i == line_count || label_set[i] > cur_class) {
		for (int c = 0; c < (int)(class_set.size())-1; ++c) {
		  for (int j = 0; j < (int)(class_set[c].size()); ++j) {
			for (int cc = c+1; cc < (int)(class_set.size()); ++cc) {
		      for (int k = 0; k < (int)(class_set[cc].size()); ++k) {
                vector<int> p(2);
			    p[0] = class_set[cc][k];
			    p[1] = class_set[c][j];
			    pair_set.push_back(p);
			  }
			}
		  }
		  list_count += 1;
		}

        cur_class = label_set[i];
		class_set = vector<vector<int> >(cur_class+1);
	  }
	  if (i == line_count) break;
	  cur_class = label_set[i];
	  if (cur_class >= 0)
		class_set[cur_class].push_back(i);
	}
	total_ins_count = pair_set.size();

    utils::Printf("Total instances count: %d\n", total_ins_count);
	utils::Printf("Total list count: %d\n", list_count);
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "TextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "TextDataLayer:top size problem.");
    
    utils::Check(batch_size > 0, "batch_size <= 0");
    utils::Check(max_doc_len > 0, "max_doc_len <= 0");

    top[0]->Resize(batch_size*2, 2, 1, max_doc_len, true);
    top[1]->Resize(batch_size*2, 1, 1, 1, true);
    
    if (show_info) {
        top[0]->PrintShape("top0");
        top[1]->PrintShape("top1");
    }
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
  }

  inline void FillData(mshadow::Tensor<xpu, 3> &top0_data, mshadow::Tensor<xpu, 2> &top0_length, 
                       mshadow::Tensor<xpu, 1> &top1_data, int top_idx, int data_idx) {
    vector<int> s1 = s1_data_set[data_idx];
	vector<int> s2 = s2_data_set[data_idx];
	int label = label_set[data_idx];
	for (int i = 0; i < s1.size(); ++i) {
	  top0_data[top_idx][0][i] = s1[i];
	}
	for (int i = 0; i < s2.size(); ++i) {
      top0_data[top_idx][1][i] = s2[i];
	}
	top0_length[top_idx][0] = s1.size();
	top0_length[top_idx][1] = s2.size();

	top1_data[top_idx] = label;
  } 

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> top0_data = top[0]->data_d3();
    mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    mshadow::Tensor<xpu, 1> top1_data = top[1]->data_d1();

	top0_data = -1;
	top0_length = 0;
	top1_data = -1;

    for (int i = 0; i < batch_size; ++i) {
      if (shuffle) {
        line_ptr = rand() % total_ins_count;
      } 
	  FillData(top0_data, top0_length, top1_data, i*2, pair_set[line_ptr][0]);
	  FillData(top0_data, top0_length, top1_data, i*2+1, pair_set[line_ptr][1]);
	  line_ptr += 1;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    
  }
  
 protected:
  std::string data_file;
  int batch_size;
  int max_doc_len;
  int min_doc_len;
  bool shuffle;
  vector<vector<int> > s1_data_set;
  vector<vector<int> > s2_data_set;
  vector<int> label_set;
  vector<vector<int> > pair_set;
  int line_ptr;
  int line_count;
  int total_ins_count;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_PAIR_TEXTDATA_LAYER_INL_HPP_

