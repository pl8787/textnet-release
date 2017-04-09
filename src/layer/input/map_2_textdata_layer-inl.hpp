#ifndef TEXTNET_LAYER_MAP_2_TEXTDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_MAP_2_TEXTDATA_LAYER_INL_HPP_

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

template<typename xpu>
class Map2TextDataLayer : public Layer<xpu>{
 public:
  Map2TextDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~Map2TextDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 3; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["batch_size"] = SettingV(1);
    this->defaults["mode"] = SettingV("batch"); // batch, pair, list    
    this->defaults["shuffle"] = SettingV(false);
    this->defaults["speedup_list"] = SettingV(false); // only when list
    this->defaults["min_doc1_len"] = SettingV(1);
    this->defaults["min_doc2_len"] = SettingV(1);
    this->defaults["fix_length"] = SettingV(false);
    this->defaults["bi_direct"] = SettingV(false);
    this->defaults["disturb_label"] = SettingV(-1.0f);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["data1_file"] = SettingV();
    this->defaults["data2_file"] = SettingV();
    this->defaults["rel_file"] = SettingV();
    this->defaults["max_doc1_len"] = SettingV();
    this->defaults["max_doc2_len"] = SettingV();

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "Map2TextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "Map2TextDataLayer:top size problem.");

    if (!Layer<xpu>::global_data.count("data1")) {
        Layer<xpu>::global_data["data1"] = vector<string>();
    }
    if (!Layer<xpu>::global_data.count("data2")) {
        Layer<xpu>::global_data["data2"] = vector<string>();
    }
    if (!Layer<xpu>::global_data.count("data12")) {
        Layer<xpu>::global_data["data12"] = vector<string>();
    }

    data1_file = setting["data1_file"].sVal();
    data2_file = setting["data2_file"].sVal();
    rel_file = setting["rel_file"].sVal();
    batch_size = setting["batch_size"].iVal();
    max_doc1_len = setting["max_doc1_len"].iVal();
    min_doc1_len = setting["min_doc1_len"].iVal();
    max_doc2_len = setting["max_doc2_len"].iVal();
    min_doc2_len = setting["min_doc2_len"].iVal();
    mode = setting["mode"].sVal();
    shuffle = setting["shuffle"].bVal();
    speedup_list = setting["speedup_list"].bVal();
    fix_length = setting["fix_length"].bVal();
    bi_direct = setting["bi_direct"].bVal();
    disturb_label = setting["disturb_label"].fVal();
    
    utils::Check(mode == "batch" || mode == "pair" || mode == "list" || mode == "inner_pair" || mode == "inner_list",
                  "Map2TextDataLayer: mode is one of batch, pair or list.");

    ReadRelData(rel_file, rel_set, label_set, data1_set, data2_set);

    ReadTextData(data1_file, data1_set, max_doc1_len, min_doc1_len);
    ReadTextData(data2_file, data2_set, max_doc2_len, min_doc2_len);

    if (mode == "pair") {
      MakePairs(rel_set, label_set, pair_set);
    } else if (mode == "list") {
      MakeLists(rel_set, label_set, list_set);
    } else if (mode == "inner_pair") {
      MakeInstances(rel_set, label_set, ins_set_1, ins_set_2, ins_map_12, ins_map_21);
    } else if (mode == "inner_list") {
      MakeInstances(rel_set, label_set, ins_set_1, ins_set_2, ins_map_12, ins_map_21);
    }
    
    line_ptr = 0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rnd_generator = std::default_random_engine(seed);
  }
  
  static bool map_list_size_cmp(const vector<int> &x1, const vector<int> &x2) {
    return x1.size() < x2.size(); // sort increase
  }
  
  void ReadTextData(string &data_file, unordered_map<string, vector<int> > &data_set, 
          int max_doc_len, int min_doc_len) {
    utils::Printf("Open data file: %s\n", data_file.c_str());    

    std::ifstream fin(data_file.c_str());
    std::string s;
    std::string key;
    std::string last_key;
    int s_len;
    int value;
    utils::Check(fin.is_open(), "Map2TextDataLayer: Open data file problem.");

    istringstream iss;
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(s);
      iss >> key;
      iss >> s_len;

      // data_set[key] = vector<int>();
      if (!data_set.count(key) || data_set[key].size() != 0)
          continue;

      last_key = key;

      // Check sentence length
      utils::Check(s_len >= min_doc_len, 
              "Map2TextDataLayer: [Length] Doc length %d less than %d, on line %d.", s_len, min_doc_len, data_set.size());

      if (s_len == 0) continue;

      while(!iss.eof()) {
        iss >> value;
        data_set[key].push_back(value);

        // Constrain on sentence length
        if (data_set[key].size() >= max_doc_len) {
            break;
        }
      }

      // Check sentence length
      utils::Check(data_set[key].size() >= min_doc_len, 
              "Map2TextDataLayer: [Read] Doc length %d less than %d, on line %d.", data_set[key].size(), min_doc_len, data_set.size());
    }
    fin.close();

    std::cout << last_key.c_str() << " ";
    for (int i = 0; i < int(data_set[last_key].size()); ++i) {
        std::cout << data_set[last_key][i] << " ";
    }
    std::cout << std::endl;

    utils::Printf("Line count in file: %d\n", data_set.size());
  }
  
  void ReadRelData(string &rel_file, vector<vector<string> > &rel_set, vector<int> &label_set, 
                   unordered_map<string, vector<int> > &data1_set, unordered_map<string, vector<int> > &data2_set) {
    utils::Printf("Open data file: %s\n", rel_file.c_str());    

    max_label = 0;

    std::ifstream fin(rel_file.c_str());
    std::string s;
    std::string value;
    int label;
    utils::Check(fin.is_open(), "MapTextdataLayer: Open data file problem.");
    line_count = 0;

    istringstream iss;
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(s);
      iss >> label;
      label_set.push_back(label);
      max_label = std::max(max_label, label);
      rel_set.push_back(vector<string>());
      while(!iss.eof()) {
        iss >> value;
        rel_set[line_count].push_back(value);
      }
      if (!data1_set.count(rel_set[line_count][0])) {
        data1_set[rel_set[line_count][0]] = vector<int>();
      }
      if (!data2_set.count(rel_set[line_count][1])) {
        data2_set[rel_set[line_count][1]] = vector<int>();
      }
      line_count += 1;
    }
    fin.close();

    for (int i = 0; i < rel_set[0].size(); ++i) {
        std::cout << rel_set[0][i].c_str() << " ";
    }
    std::cout << std::endl;

    max_label += 1;

    utils::Printf("Line count in file: %d\n", rel_set.size());
    utils::Printf("Max label: %d\n", max_label);
  }

  void MakePairs(vector<vector<string> > &rel_set, vector<int> &label_set, vector<vector<int> > &pair_set) {
    int list_count = 0;
    string cur_data1 = "";
    vector<vector<int> > class_set;

    std::cout << line_count << std::endl;
    
    // for store the last list pair, so i<=line_count
    for (int i = 0; i <= line_count; ++i) {
      if (i == line_count || rel_set[i][0] != cur_data1) {
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
        class_set = vector<vector<int> >(max_label);
      }

      // Quit loop 
      if (i == line_count) break;

      cur_data1 = rel_set[i][0];
      class_set[label_set[i]].push_back(i);
    }

    utils::Printf("Instances count: %d\n", pair_set.size());
    utils::Printf("List count: %d\n", list_count);
  }

  void MakeLists(vector<vector<string> > &rel_set, vector<int> &label_set, vector<vector<int> > &list_set) {
    vector<int> list;
    max_list = 0;
    string cur_data1 = "";

    for (int i = 0; i < line_count; ++i) {
      if (rel_set[i][0] != cur_data1 && list.size() != 0) {
        list_set.push_back(list);
        max_list = std::max(max_list, (int)list.size());
        list = vector<int>();
      }
      cur_data1 = rel_set[i][0];
      list.push_back(i);
    }
    list_set.push_back(list);
    max_list = std::max(max_list, (int)list.size());

    // for speed up we can sort list by list.size()
    if (speedup_list)
      sort(list_set.begin(), list_set.end(), map_list_size_cmp);

    utils::Printf("Max list length: %d\n", max_list);
    utils::Printf("List count: %d\n", list_set.size());
  }

  void MakeInstances(vector<vector<string> > &rel_set, vector<int> &label_set, 
                     vector<string> &ins_set_1, vector<string> &ins_set_2, 
                     unordered_map<string, vector<string> > &ins_map_12, 
                     unordered_map<string, vector<string> > &ins_map_21) {
    for (int i = 0; i < line_count; ++i) {
      utils::Check(label_set[i] == 1, "Map2TextDataLayer: only support one label.");
      // store data1
      if ( !ins_map_12.count(rel_set[i][0]) ) {
        ins_set_1.push_back(rel_set[i][0]);
        ins_map_12[rel_set[i][0]] = vector<string>();
      }
      if ( !ins_map_21.count(rel_set[i][1]) ) {
        ins_set_2.push_back(rel_set[i][1]);
        ins_map_21[rel_set[i][1]] = vector<string>();
      }

      // make instance
      ins_map_12[rel_set[i][0]].push_back(rel_set[i][1]);
      ins_map_21[rel_set[i][1]].push_back(rel_set[i][0]);
    }
    
    utils::Printf("Num instance in data1: %d\n", ins_set_1.size());
    utils::Printf("Num instance in data2: %d\n", ins_set_2.size());
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "Map2TextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "Map2TextDataLayer:top size problem.");
    
    utils::Check(batch_size > 0, "Map2TextDataLayer:batch_size <= 0");
    utils::Check(max_doc1_len > 0, "Map2TextDataLayer:max_doc1_len <= 0");
    utils::Check(max_doc2_len > 0, "Map2TextDataLayer:max_doc2_len <= 0");

    if (mode == "batch") {
      top[0]->Resize(batch_size, 1, 1, max_doc1_len, true);
      top[1]->Resize(batch_size, 1, 1, max_doc2_len, true);
      top[2]->Resize(batch_size, 1, 1, 1, true);
    } else if (mode == "pair") {
      top[0]->Resize(2 * batch_size, 1, 1, max_doc1_len, true);
      top[1]->Resize(2 * batch_size, 1, 1, max_doc2_len, true);
      top[2]->Resize(2 * batch_size, 1, 1, 1, true);
    } else if (mode == "list") {
      top[0]->Resize(max_list * batch_size, 1, 1, max_doc1_len, true);
      top[1]->Resize(max_list * batch_size, 1, 1, max_doc2_len, true);
      top[2]->Resize(max_list * batch_size, 1, 1, 1, true);
    } else if (mode == "inner_pair") {
      int factor = bi_direct ? 4 : 2;
      top[0]->Resize(factor * (batch_size-1) * batch_size, 1, 1, max_doc1_len, true);
      top[1]->Resize(factor * (batch_size-1) * batch_size, 1, 1, max_doc2_len, true);
      top[2]->Resize(factor * (batch_size-1) * batch_size, 1, 1, 1, true);
    } else if (mode == "inner_list") {
      int factor = bi_direct ? 2 : 1;
      top[0]->Resize((factor * (batch_size-1) + 1) * batch_size, 1, 1, max_doc1_len, true);
      top[1]->Resize((factor * (batch_size-1) + 1) * batch_size, 1, 1, max_doc2_len, true);
      top[2]->Resize((factor * (batch_size-1) + 1) * batch_size, 1, 1, 1, true);
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

  inline void FillData(mshadow::Tensor<xpu, 4> &top0_data, mshadow::Tensor<xpu, 2> &top0_length,
                       mshadow::Tensor<xpu, 4> &top1_data, mshadow::Tensor<xpu, 2> &top1_length, 
                       int top_idx, int data_idx) {
      utils::Check(data1_set.count(rel_set[data_idx][0]), 
              "MapTextdataLayer: %s not in data1_set.", rel_set[data_idx][0].c_str());
      utils::Check(data2_set.count(rel_set[data_idx][1]), 
              "MapTextdataLayer: %s not in data2_set.", rel_set[data_idx][1].c_str());
      vector<int> &data1 = data1_set[rel_set[data_idx][0]];
      vector<int> &data2 = data2_set[rel_set[data_idx][1]];

      for (int k = 0; k < data1.size(); ++k) {
          top0_data[top_idx][0][0][k] = data1[k];
      }
      if (fix_length) {
          top0_length[top_idx][0] = max_doc1_len;
      } else {
          top0_length[top_idx][0] = data1.size();
      }

      for (int k = 0; k < data2.size(); ++k) {
          top1_data[top_idx][0][0][k] = data2[k];
      }
      if (fix_length) {
          top1_length[top_idx][0] = max_doc2_len;
      } else {
          top1_length[top_idx][0] = data2.size();
      }
  } 
  
  inline void FillData2(mshadow::Tensor<xpu, 4> &top0_data, mshadow::Tensor<xpu, 2> &top0_length,
                        mshadow::Tensor<xpu, 4> &top1_data, mshadow::Tensor<xpu, 2> &top1_length, 
                        int top_idx, string &data_id1, string &data_id2) {
      utils::Check(data1_set.count(data_id1), 
              "MapTextdataLayer: %s not in data1_set.", data_id1.c_str());
      utils::Check(data2_set.count(data_id2), 
              "MapTextdataLayer: %s not in data2_set.", data_id2.c_str());
      vector<int> &data1 = data1_set[data_id1];
      vector<int> &data2 = data2_set[data_id2];

      for (int k = 0; k < data1.size(); ++k) {
          top0_data[top_idx][0][0][k] = data1[k];
      }
      if (fix_length) {
          top0_length[top_idx][0] = max_doc1_len;
      } else {
          top0_length[top_idx][0] = data1.size();
      }

      for (int k = 0; k < data2.size(); ++k) {
          top1_data[top_idx][0][0][k] = data2[k];
      }
      if (fix_length) {
          top1_length[top_idx][0] = max_doc2_len;
      } else {
          top1_length[top_idx][0] = data2.size();
      }
  } 

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top0_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    mshadow::Tensor<xpu, 4> top1_data = top[1]->data;
    mshadow::Tensor<xpu, 2> top1_length = top[1]->length;
    mshadow::Tensor<xpu, 1> top2_data = top[2]->data_d1();

    top0_data = -1;
    if (fix_length) {
      top0_length = max_doc1_len;
    } else {
      top0_length = 0;
    }
    top1_data = -1;
    if (fix_length) {
      top1_length = max_doc2_len;
    } else {
      top1_length = 0;
    }
    top2_data = -1;

    if (mode == "batch") {
      Layer<xpu>::global_data["data1"].clear();
      Layer<xpu>::global_data["data2"].clear();
      Layer<xpu>::global_data["data12"].clear();
      for (int i = 0; i < batch_size; ++i) {
        if (shuffle) {
          line_ptr = rand() % line_count;
        } 
        FillData(top0_data, top0_length, top1_data, top1_length, i, line_ptr);
        top2_data[i] = label_set[line_ptr];
        if (disturb_label > 0.0f && (rand() % 100) < (disturb_label*100)) {
            top2_data[i] = 1.0f - top2_data[i];
        }
        Layer<xpu>::global_data["data1"].push_back(rel_set[line_ptr][0]);
        Layer<xpu>::global_data["data2"].push_back(rel_set[line_ptr][1]);
        Layer<xpu>::global_data["data12"].push_back(rel_set[line_ptr][0]+string("_")+rel_set[line_ptr][1]);

        line_ptr = (line_ptr + 1) % line_count;
      }
    } else if (mode == "pair") {
      Layer<xpu>::global_data["data1"].clear();
      Layer<xpu>::global_data["data2"].clear();
      Layer<xpu>::global_data["data12"].clear();
      for (int i = 0; i < batch_size; ++i) {
        if (shuffle) {
          line_ptr = rand() % pair_set.size();
        } 

        int pos_idx = pair_set[line_ptr][0];
        int neg_idx = pair_set[line_ptr][1];

        Layer<xpu>::global_data["data1"].push_back(rel_set[pos_idx][0]);
        Layer<xpu>::global_data["data2"].push_back(rel_set[pos_idx][1]);
        Layer<xpu>::global_data["data12"].push_back(rel_set[pos_idx][0]+string("_")+rel_set[pos_idx][1]);
        Layer<xpu>::global_data["data1"].push_back(rel_set[neg_idx][0]);
        Layer<xpu>::global_data["data2"].push_back(rel_set[neg_idx][1]);
        Layer<xpu>::global_data["data12"].push_back(rel_set[neg_idx][0]+string("_")+rel_set[neg_idx][1]);

        FillData(top0_data, top0_length, top1_data, top1_length, 2*i, pos_idx);
        FillData(top0_data, top0_length, top1_data, top1_length, 2*i+1, neg_idx);

        top2_data[2*i] = 1;
        top2_data[2*i+1] = 0;
        line_ptr = (line_ptr + 1) % pair_set.size();
      }
    } else if (mode == "list") {
      Layer<xpu>::global_data["data1"].clear();
      Layer<xpu>::global_data["data2"].clear();
      Layer<xpu>::global_data["data12"].clear();
      for (int s = 0; s < batch_size; ++s) {
        for (int i = 0; i < list_set[line_ptr].size(); ++i) {
          int idx = list_set[line_ptr][i];
          int out_idx = s * max_list + i;

          Layer<xpu>::global_data["data1"].push_back(rel_set[idx][0]);
          Layer<xpu>::global_data["data2"].push_back(rel_set[idx][1]);
          Layer<xpu>::global_data["data12"].push_back(rel_set[idx][0]+string("_")+rel_set[idx][1]);

          FillData(top0_data, top0_length, top1_data, top1_length, out_idx, idx);
          top2_data[out_idx] = label_set[idx];
        }
        line_ptr = (line_ptr + 1) % list_set.size();
      }
    } else if (mode == "inner_pair") {
      int out_idx = 0;
      int ins1_size = ins_set_1.size();
      if (shuffle && line_ptr >= ins1_size) {
        line_ptr = 0;
        std::shuffle(ins_set_1.begin(), ins_set_1.end(), rnd_generator);
      }
      // Initial positive position
      vector<int> rnd_pos(batch_size);
      for (int s = 0; s < batch_size; ++s) {
        rnd_pos[s] = rand() % ins_map_12[ ins_set_1[(line_ptr+s) % ins1_size] ].size();
      }
      for (int s1 = 0; s1 < batch_size; ++s1) {
        string s1_id = ins_set_1[(line_ptr+s1) % ins1_size];
        for (int s2 = 0; s2 < batch_size; ++s2) {
          string s2_id = ins_set_1[(line_ptr+s2) % ins1_size];
          if (s1 != s2) {
            FillData2(top0_data, top0_length, top1_data, top1_length, out_idx, s1_id, ins_map_12[s1_id][rnd_pos[s1]]);
            top2_data[out_idx] = 1;
            ++out_idx;
            FillData2(top0_data, top0_length, top1_data, top1_length, out_idx, s1_id, ins_map_12[s2_id][rnd_pos[s2]]);
            top2_data[out_idx] = 0;
            ++out_idx;
          }
          if (bi_direct && s1 != s2) {
            FillData2(top0_data, top0_length, top1_data, top1_length, out_idx, s1_id, ins_map_12[s1_id][rnd_pos[s1]]);
            top2_data[out_idx] = 1;
            ++out_idx;
            FillData2(top0_data, top0_length, top1_data, top1_length, out_idx, s2_id, ins_map_12[s1_id][rnd_pos[s1]]);
            top2_data[out_idx] = 0;
            ++out_idx;
          }
        }
      }
      line_ptr += batch_size;
    } else if (mode == "inner_list") {
      int out_idx = 0;
      int ins1_size = ins_set_1.size();
      if (shuffle && line_ptr >= ins1_size) {
        line_ptr = 0;
        std::shuffle(ins_set_1.begin(), ins_set_1.end(), rnd_generator);
      }
      // Initial positive position
      vector<int> rnd_pos(batch_size);
      for (int s = 0; s < batch_size; ++s) {
        rnd_pos[s] = rand() % ins_map_12[ ins_set_1[(line_ptr+s) % ins1_size] ].size();
      }
      for (int s1 = 0; s1 < batch_size; ++s1) {
        string s1_id = ins_set_1[(line_ptr+s1) % ins1_size];
        FillData2(top0_data, top0_length, top1_data, top1_length, out_idx, s1_id, ins_map_12[s1_id][rnd_pos[s1]]);
        top2_data[out_idx] = 1;
        ++out_idx;
        for (int s2 = 0; s2 < batch_size; ++s2) {
          string s2_id = ins_set_1[(line_ptr+s2) % ins1_size];
          if (s1 != s2) {
            FillData2(top0_data, top0_length, top1_data, top1_length, out_idx, s1_id, ins_map_12[s2_id][rnd_pos[s2]]);
            top2_data[out_idx] = 0;
            ++out_idx;
          }
          if (bi_direct && s1 != s2) {
            FillData2(top0_data, top0_length, top1_data, top1_length, out_idx, s2_id, ins_map_12[s1_id][rnd_pos[s1]]);
            top2_data[out_idx] = 0;
            ++out_idx;
          }
        }
      }
      line_ptr += batch_size;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    
  }
  
  static unordered_map<string, vector<int> > data1_set;
  static unordered_map<string, vector<int> > data2_set;

 protected:
  string data1_file;
  string data2_file;
  string rel_file;

  int batch_size;
  int min_doc1_len;
  int max_doc1_len;
  int min_doc2_len;
  int max_doc2_len;
  string mode;
  bool shuffle;
  bool speedup_list;
  bool fix_length;
  bool bi_direct;
  float disturb_label;
  
  vector<vector<string> > rel_set;
  vector<int> label_set;
  vector<vector<int> > pair_set;
  vector<vector<int> > list_set;

  vector<string> ins_set_1;
  vector<string> ins_set_2;
  unordered_map<string, vector<string> > ins_map_12;
  unordered_map<string, vector<string> > ins_map_21;

  int line_count;
  int line_ptr;
  int max_label;

  int max_list;
  std::default_random_engine rnd_generator;
};
template<typename xpu> unordered_map<string, vector<int> > Map2TextDataLayer<xpu>::data1_set = unordered_map<string, vector<int> >();
template<typename xpu> unordered_map<string, vector<int> > Map2TextDataLayer<xpu>::data2_set = unordered_map<string, vector<int> >();
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MAP_2_TEXTDATA_LAYER_INL_HPP_

