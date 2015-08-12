#ifndef TEXTNET_LAYER_QA_TEXTDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_QA_TEXTDATA_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include "stdlib.h"

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

using namespace std;

namespace textnet {
namespace layer {

bool list_size_cmp(const vector<int> &x1, const vector<int> &x2) {
  return x1.size() < x2.size(); // sort increase
}

template<typename xpu>
class QATextDataLayer : public Layer<xpu>{
 public:
  QATextDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~QATextDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 3; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["batch_size"] = SettingV(0);
    this->defaults["mode"] = SettingV("batch"); // batch, pair, list    
	this->defaults["shuffle"] = SettingV(false);
	this->defaults["speedup_list"] = SettingV(true); // only when list
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["question_data_file"] = SettingV();
    this->defaults["answer_data_file"] = SettingV();
    this->defaults["question_rel_file"] = SettingV();
    this->defaults["answer_rel_file"] = SettingV();
    this->defaults["max_doc_len"] = SettingV();
    this->defaults["candids"] = SettingV();

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "QATextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "QATextDataLayer:top size problem.");

    question_data_file = setting["question_data_file"].sVal();
    answer_data_file = setting["answer_data_file"].sVal();
    question_rel_file = setting["question_rel_file"].sVal();
    answer_rel_file = setting["answer_rel_file"].sVal();
    batch_size = setting["batch_size"].iVal();
    max_doc_len = setting["max_doc_len"].iVal();
    candids = setting["candids"].iVal();
    mode = setting["mode"].sVal();
	shuffle = setting["shuffle"].bVal();
	speedup_list = setting["speedup_list"].bVal();
    
    utils::Check(mode == "batch" || mode == "pair" || mode == "list",
                  "QATextDataLayer: mode is one of batch, pair or list.");

    ReadTextData(question_data_file, question_data_set);
    ReadTextData(answer_data_file, answer_data_set);

    ReadRelData(question_rel_file, question_rel_set);
    ReadRelData(answer_rel_file, answer_rel_set);

    ReadLabel(question_rel_file, label_set);

    if (mode == "pair") {
      MakePairs(label_set, pair_set);
    } else if (mode == "list") {
      MakeLists(label_set, list_set);
    }
    
    line_ptr = 0;
  }
  
  void ReadTextData(string &data_file, unordered_map<string, vector<int> > &data_set) {
    utils::Printf("Open data file: %s\n", data_file.c_str());    

    std::ifstream fin(data_file.c_str());
    std::string s;
    std::string key;
    int s_len;
    int value;
    utils::Check(fin, "Open data file problem.");

    istringstream iss;
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(s);
      iss >> key;
      iss >> s_len;
      data_set[key] = vector<int>();
      while(!iss.eof()) {
        iss >> value;
        data_set[key].push_back(value);
      }
    }
    fin.close();

    utils::Printf("%s\n", key.c_str());
    for (int i = 0; i < data_set[key].size(); ++i) {
      utils::Printf("%d ", data_set[key][i]);
    }
    utils::Printf("\n");

    utils::Printf("Line count in file: %d\n", data_set.size());
  }
  
  void ReadRelData(string &data_file, vector<vector<string> > &data_set) {
    utils::Printf("Open data file: %s\n", data_file.c_str());    

    std::ifstream fin(data_file.c_str());
    std::string s;
    std::string value;
    int label;
    utils::Check(fin, "Open data file problem.");
    line_count = 0;

    istringstream iss;
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(s);
      iss >> label;
      data_set.push_back(vector<string>());
      while(!iss.eof()) {
        iss >> value;
        data_set[line_count].push_back(value);
      }
      line_count += 1;
    }
    fin.close();

    for (int i = 0; i < data_set[0].size(); ++i) {
      utils::Printf("%s ", data_set[0][i].c_str());
    }
    utils::Printf("\n");

    utils::Printf("Line count in file: %d\n", data_set.size());

  }

  void ReadLabel(string &data_file, vector<int> &data_set) {
    utils::Printf("Open data file: %s\n", data_file.c_str());    

    std::ifstream fin(data_file.c_str());
    std::string s;
    int label;
    utils::Check(fin, "Open data file problem.");

    istringstream iss;
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(s);
      iss >> label;
      label_set.push_back(label);
    }
    fin.close();

    for (int i = 0; i < 10; ++i) {
      utils::Printf("%d ", data_set[i]);
    }
    utils::Printf("\n");

    utils::Printf("Line count in file: %d\n", data_set.size());
  }

  void MakePairs(vector<int> &label_set, vector<vector<int> > &pair_set) {
    int pos_idx = -1;
    for (int i = 0; i < label_set.size(); ++i) {
      if (label_set[i] == 1) {
        pos_idx = i;
      } else {
        vector<int> pair;
        pair.push_back(pos_idx);
        pair.push_back(i);
        pair_set.push_back(pair);
      }
    }
  }

  void MakeLists(vector<int> &label_set, vector<vector<int> > &list_set) {
    vector<int> list;
    max_list = 0;
    list.push_back(0);
    for (int i = 1; i < label_set.size(); ++i) {
      if (label_set[i] == 1) {
        list_set.push_back(list);
        max_list = std::max(max_list, (int)list.size());
        list = vector<int>();
        list.push_back(i);
      } else {
        list.push_back(i);
      }
    }
    list_set.push_back(list);
    max_list = std::max(max_list, (int)list.size());

    // for speed up we can sort list by list.size()
	if (speedup_list)
	  sort(list_set.begin(), list_set.end(), list_size_cmp);
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "QATextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "QATextDataLayer:top size problem.");
    
    utils::Check(batch_size > 0, "batch_size <= 0");
    utils::Check(max_doc_len > 0, "max_doc_len <= 0");
    utils::Check(candids > 0, "candids <= 0");

    if (mode == "batch") {
      top[0]->Resize(batch_size * (candids + 1), 1, 1, max_doc_len, true);
      top[1]->Resize(batch_size * (candids + 1), 1, 1, max_doc_len, true);
      top[2]->Resize(batch_size, 1, 1, 1, true);
    } else if (mode == "pair") {
      top[0]->Resize(2 * batch_size * (candids + 1), 1, 1, max_doc_len, true);
      top[1]->Resize(2 * batch_size * (candids + 1), 1, 1, max_doc_len, true);
      top[2]->Resize(2 * batch_size, 1, 1, 1, true);
    } else if (mode == "list") {
      top[0]->Resize(max_list * (candids + 1), 1, 1, max_doc_len, true);
      top[1]->Resize(max_list * (candids + 1), 1, 1, max_doc_len, true);
      top[2]->Resize(max_list, 1, 1, 1, true);
    }
    
    if (show_info) {
        top[0]->PrintShape("top0");
        top[1]->PrintShape("top1");
        top[2]->PrintShape("top2");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (mode == "list") {
      if (max_list != list_set[line_ptr].size()) {
        max_list = list_set[line_ptr].size();
        need_reshape = true;
      }
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }

  inline void FillData(mshadow::Tensor<xpu, 2> &top0_data, mshadow::Tensor<xpu, 1> &top0_length, 
                       mshadow::Tensor<xpu, 2> &top1_data, mshadow::Tensor<xpu, 1> &top1_length,
                       int top_idx, int data_idx) {
    for (int j = 0; j < candids + 1; ++j) {

      vector<int> &q_data = question_data_set[question_rel_set[data_idx][j]];
      for (int k = 0; k < q_data.size(); ++k) {
          top0_data[top_idx*(candids+1)+j][k] = q_data[k];
      }
      top0_length[top_idx*(candids+1)+j] = q_data.size();

      vector<int> &a_data = answer_data_set[answer_rel_set[data_idx][j]];
      for (int k = 0; k < a_data.size(); ++k) {
          top1_data[top_idx*(candids+1)+j][k] = a_data[k];
      }
      top1_length[top_idx*(candids+1)+j] = a_data.size();

    }
  } 
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> top0_data = top[0]->data_d2();
    mshadow::Tensor<xpu, 1> top0_length = top[0]->length_d1();
    mshadow::Tensor<xpu, 2> top1_data = top[1]->data_d2();
    mshadow::Tensor<xpu, 1> top1_length = top[1]->length_d1();
    mshadow::Tensor<xpu, 1> top2_data = top[2]->data_d1();

    top0_data = -1;
    top1_data = -1;
    top0_length = 0;
    top1_length = 0;
    top2_data = -1;

    if (mode == "batch") {
      for (int i = 0; i < batch_size; ++i) {
        if (shuffle) {
          line_ptr = rand() % line_count;
        } 
        FillData(top0_data, top0_length, top1_data, top1_length, i, line_ptr);
        top2_data[i] = label_set[line_ptr];
        line_ptr = (line_ptr + 1) % line_count;
      }
    } else if (mode == "pair") {
      for (int i = 0; i < batch_size; ++i) {
        if (shuffle) {
          line_ptr = rand() % pair_set.size();
        } 

        int pos_idx = pair_set[line_ptr][0];
        int neg_idx = pair_set[line_ptr][1];

        FillData(top0_data, top0_length, top1_data, top1_length, 2*i, pos_idx);
        FillData(top0_data, top0_length, top1_data, top1_length, 2*i+1, neg_idx);

        top2_data[2*i] = 1;
        top2_data[2*i+1] = 0;
        line_ptr = (line_ptr + 1) % pair_set.size();
      }
    } else if (mode == "list") {
      for (int i = 0; i < list_set[line_ptr].size(); ++i) {
        int idx = list_set[line_ptr][i];
        FillData(top0_data, top0_length, top1_data, top1_length, i, idx);
        top2_data[i] = label_set[idx];
      }
      line_ptr = (line_ptr + 1) % list_set.size();
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    
  }
  
 protected:
  string question_data_file;
  string answer_data_file;
  string question_rel_file;
  string answer_rel_file;

  int batch_size;
  int max_doc_len;
  int candids;
  string mode;
  bool shuffle;
  bool speedup_list;
  
  unordered_map<string, vector<int> > question_data_set;
  unordered_map<string, vector<int> > answer_data_set;

  vector<vector<string> > question_rel_set;
  vector<vector<string> > answer_rel_set;
  vector<int> label_set;
  vector<vector<int> > pair_set;
  vector<vector<int> > list_set;

  int line_count;
  int line_ptr;

  int max_list;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_QA_TEXTDATA_LAYER_INL_HPP_

