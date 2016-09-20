#ifndef TEXTNET_LAYER_MAP_2_WINDOW_TEXTDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_MAP_2_WINDOW_TEXTDATA_LAYER_INL_HPP_

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
class Map2WindowTextDataLayer : public Layer<xpu>{
 public:
  Map2WindowTextDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~Map2WindowTextDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 4; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["batch_size"] = SettingV(1);
    this->defaults["window_size"] = SettingV(100);
    this->defaults["mode"] = SettingV("batch"); // batch, pair, list    
    this->defaults["shuffle"] = SettingV(false);
    this->defaults["speedup_list"] = SettingV(false); // only when list
    this->defaults["fix_length"] = SettingV(false);
    this->defaults["max_doc_len"] = SettingV(2000);
    this->defaults["data1_doc_len"] = SettingV(18);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["data1_file"] = SettingV();
    this->defaults["data2_file"] = SettingV();
    this->defaults["rel_file"] = SettingV();

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "Map2WindowTextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "Map2WindowTextDataLayer:top size problem.");

    if (!Layer<xpu>::global_data.count("data1")) {
        Layer<xpu>::global_data["data1"] = vector<string>();
    }
    if (!Layer<xpu>::global_data.count("data2")) {
        Layer<xpu>::global_data["data2"] = vector<string>();
    }

    data1_file = setting["data1_file"].sVal();
    data2_file = setting["data2_file"].sVal();
    rel_file = setting["rel_file"].sVal();
    batch_size = setting["batch_size"].iVal();
    window_size = setting["window_size"].iVal();
    max_doc_len = setting["max_doc_len"].iVal();
    data1_doc_len = setting["data1_doc_len"].iVal();
    mode = setting["mode"].sVal();
    shuffle = setting["shuffle"].bVal();
    speedup_list = setting["speedup_list"].bVal();
    fix_length = setting["fix_length"].bVal();
    max_doc1_len = -1;
    max_doc2_len = -1;
    
    utils::Check(mode == "batch" || mode == "pair" || mode == "list" ,
                  "Map2WindowTextDataLayer: mode is one of batch, pair or list.");

    /*
       * rel_set :: vector<vector<string>> ->  vector<pair<query,doc>>
    */
    ReadRelData(rel_file, rel_set, label_set, data1_set, data2_set);

    ReadTextData(data1_file, data1_set, max_doc1_len, min_doc1_len);
    ReadTextData(data2_file, data2_set, max_doc2_len, min_doc2_len);
    //printf("SetUpLayer max_doc1_len:%d, max_doc2_len:%d\n",max_doc1_len,max_doc2_len);

    if (mode == "pair") {
      MakePairs(rel_set, label_set, pair_set);
    } else if (mode == "list") {
      MakeLists(rel_set, label_set, list_set);
    } 
    //else if (mode == "inner_pair") {
      //MakeInstances(rel_set, label_set, ins_set_1, ins_set_2, ins_map_12, ins_map_21);
    //} else if (mode == "inner_list") {
      //MakeInstances(rel_set, label_set, ins_set_1, ins_set_2, ins_map_12, ins_map_21);
    //}
    
    line_ptr = 0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rnd_generator = std::default_random_engine(seed);
  }
  
  static bool map_list_size_cmp(const vector<int> &x1, const vector<int> &x2) {
    return x1.size() < x2.size(); // sort increase
  }
  
  void ReadTextData(string &data_file, unordered_map<string, vector<int> > &data_set, 
          int & cmax_doc_len, int & cmin_doc_len) {
    utils::Printf("Open data file: %s\n", data_file.c_str());    

    std::ifstream fin(data_file.c_str());
    std::string s;
    std::string key;
    std::string last_key;
    int s_len;
    int value;
    utils::Check(fin.is_open(), "Map2WindowTextDataLayer: Open data file problem.");

    istringstream iss;
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(s);
      iss >> key;
      iss >> s_len;
      cmax_doc_len = cmax_doc_len > s_len ? cmax_doc_len : s_len;
      cmax_doc_len = cmax_doc_len > max_doc_len ? max_doc_len : cmax_doc_len;
      cmin_doc_len = cmin_doc_len > s_len ? cmin_doc_len : s_len;

      // data_set[key] = vector<int>();
      if (!data_set.count(key) || data_set[key].size() != 0)
          continue;

      last_key = key;
      while(!iss.eof()) {
        iss >> value;
        data_set[key].push_back(value);
        if(data_set[key].size() >= max_doc_len) break;
      }

      // Check sentence length
      utils::Check(data_set[key].size() > 0 , 
              "Map2WindowTextDataLayer: Doc length %d less than %d, on line %d.", data_set[key].size(), cmin_doc_len, data_set.size());
    }
    fin.close();

    std::cout << last_key.c_str() << " ";
    for (int i = 0; i < data_set[last_key].size(); ++i) {
        std::cout << data_set[last_key][i] << " ";
    }
    std::cout << std::endl;

    utils::Printf("Line count in file: %d\n", data_set.size());
  }
  
  void ReadRelData(string &rel_file, vector<vector<string> > &crel_set, vector<int> &label_set, 
                   unordered_map<string, vector<int> > &data1_set, unordered_map<string, vector<int> > &data2_set) {
    utils::Printf("Open data file: %s\n", rel_file.c_str());    

    max_label = 0;

    std::ifstream fin(rel_file.c_str());
    std::string s;
    std::string value;
    int label;
    utils::Check(fin.is_open(), "Map2WindowTextDataLayer: Open data file problem.");
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
      crel_set.push_back(vector<string>());
      while(!iss.eof()) {
        iss >> value;
        crel_set[line_count].push_back(value);
      }
      if (!data1_set.count(crel_set[line_count][0])) {
        data1_set[crel_set[line_count][0]] = vector<int>();
      }
      if (!data2_set.count(crel_set[line_count][1])) {
        data2_set[crel_set[line_count][1]] = vector<int>();
      }
      line_count += 1;
    }
    fin.close();

    for (int i = 0; i < crel_set[0].size(); ++i) {
        std::cout << crel_set[0][i].c_str() << " ";
    }
    std::cout << std::endl;

    max_label += 1;

    utils::Printf("Line count in file: %d\n", crel_set.size());
    utils::Printf("Max label: %d\n", max_label);
  }

  void MakePairs(vector<vector<string> > &rel_set, vector<int> &label_set,
          vector<vector<int> > &pair_set) {
    int list_count = 0;
    string cur_data1 = ""; // record current qid
    vector<vector<int> > class_set;

    std::cout << line_count << std::endl;
    
    // for store the last list pair, so i<=line_count
    for (int i = 0; i <= line_count; ++i) {
      if (i == line_count || rel_set[i][0] != cur_data1) {
        for (int c = 0; c < (int)(class_set.size())-1; ++c) {
          for (int j = 0; j < (int)(class_set[c].size()); ++j) { // negative
            for (int cc = c+1; cc < (int)(class_set.size()); ++cc) {
              for (int k = 0; k < (int)(class_set[cc].size()); ++k) {  //positive
                vector<int> p(2);
                p[0] = class_set[cc][k]; // positive  line number
                p[1] = class_set[c][j]; // negative line number
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

    utils::Printf("Instances(pairs) count: %d\n", pair_set.size());
    utils::Printf("List count: %d\n", list_count);
  }

  void MakeLists(vector<vector<string> > &rel_set, vector<int> &label_set,
          vector<vector<int> > &list_set) {
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
      utils::Check(label_set[i] == 1, "Map2WindowTextDataLayer: only support one label.");
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
                  "Map2WindowTextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "Map2WindowTextDataLayer:top size problem.");
    
    utils::Check(batch_size > 0, "Map2WindowTextDataLayer:batch_size <= 0");
    utils::Check(max_doc1_len > 0, "Map2WindowTextDataLayer:max_doc1_len <= 0");
    utils::Check(max_doc2_len > 0, "Map2WindowTextDataLayer:max_doc2_len <= 0");

    if (mode == "batch") {
        utils::Check(0,"map_2_window_datatext: reshape, batch not ready.");
    } else if (mode == "pair") {
      Layer<xpu>::global_data["data1"].clear();
      Layer<xpu>::global_data["data2"].clear();
      top[2]->Resize(2 * batch_size, 1, 1, 1, true); //record label for each document
      top[3]->Resize(2 * batch_size, 1, 1, 1, true); // record the window size for each document
      top[2]->data = -1;
      top[3]->data = -1;

      vector<int> vbatches(batch_size);
      wbatch_size = 0;
      int cline_ptr = line_ptr % pair_set.size();
      for (int i = 0; i < batch_size; ++i) {
        if (shuffle) {
          cline_ptr = rand() % pair_set.size();
        } 
        int pos_idx = pair_set[cline_ptr][0];
        int neg_idx = pair_set[cline_ptr][1];
        int pos_doc_len = data2_set[rel_set[pos_idx][1]].size();
        //int pos_doc_window_num = int((pos_doc_len * 2 - 2 + window_size) / window_size);
        int pos_doc_window_num = int((pos_doc_len  - 1 + window_size) / window_size);
        int neg_doc_len = data2_set[rel_set[neg_idx][1]].size();
        int neg_doc_window_num = int((neg_doc_len  - 1 + window_size) / window_size);
        wbatch_size += (pos_doc_window_num + neg_doc_window_num);
        top[2]->data[2*i][0][0][0] = 1;
        top[2]->data[2*i+1][0][0][0] = 0;
        top[3]->data[2*i][0][0][0] = pos_doc_window_num;
        top[3]->data[2*i+1][0][0][0] = neg_doc_window_num;

        vbatches[i] = cline_ptr;
        cline_ptr = (cline_ptr + 1) % pair_set.size();
      }
      top[0]->Resize(wbatch_size, 1, 1, data1_doc_len, true); //query
      top[1]->Resize(wbatch_size, 1, 1, window_size, true); //doc
      top[0]->data = -1;
      top[1]->data = -1;

      index_t idx_top = 0;
      for(int i = 0 ; i < batch_size; ++ i){
        int pos_idx = pair_set[vbatches[i]][0];
        int neg_idx = pair_set[vbatches[i]][1];

        Layer<xpu>::global_data["data1"].push_back(rel_set[pos_idx][0]);
        Layer<xpu>::global_data["data2"].push_back(rel_set[pos_idx][1]);
        Layer<xpu>::global_data["data1"].push_back(rel_set[neg_idx][0]);
        Layer<xpu>::global_data["data2"].push_back(rel_set[neg_idx][1]);

        FillData(top[0]->data, top[0]->length, top[1]->data, top[1]->length, idx_top, top[3]->data[2*i][0][0][0], pos_idx);
        idx_top += top[3]->data[2*i][0][0][0];
        FillData(top[0]->data, top[0]->length, top[1]->data, top[1]->length, idx_top, top[3]->data[2*i+1][0][0][0], neg_idx);
        idx_top += top[3]->data[2*i+1][0][0][0];
      }
      //if(idx_top != wbatch_size)    printf("idx_top:%d\t,wbatch_size:%d\n",idx_top,wbatch_size);
      utils::Check(idx_top == wbatch_size, "idx_top not equal to wbatch_size.");
    } else if (mode == "list") {
      Layer<xpu>::global_data["data1"].clear();
      Layer<xpu>::global_data["data2"].clear();
      int cline_ptr = line_ptr % list_set.size() ;
      if (max_list != list_set[cline_ptr].size()) {
        max_list = list_set[cline_ptr].size();
      }
      //printf("cline_ptr:%d,max_list:%d,batch_size:%d\n",cline_ptr,max_list,batch_size);
      top[2]->Resize(max_list * batch_size, 1, 1, 1, true);
      top[3]->Resize(max_list * batch_size, 1, 1, 1, true);
      top[2]->data = -1;
      top[3]->data = -1;
      
      vector<int> vbatches(batch_size);
      wbatch_size = 0;
      for (int s = 0; s < batch_size; ++s) {
        vbatches[s] = cline_ptr;
        int curr_pos = 0;
        //printf("curr batch :%d, list-size:%d\n",s,list_set[cline_ptr].size());
        for (int i = 0; i < list_set[cline_ptr].size(); ++i) {
          int idx = list_set[cline_ptr][i];
          //printf("i:%d,idx:%d,doc:%s\n",i,idx,rel_set[idx][1].c_str());
          int curr_doc_len = data2_set[rel_set[idx][1]].size();
          //int curr_doc_window_num = int((curr_doc_len * 2 - 2 + window_size) / window_size);
          int curr_doc_window_num = int((curr_doc_len  - 1 + window_size) / window_size);
          wbatch_size += curr_doc_window_num;
          int out_idx = s * max_list + i;
          top[2]->data[out_idx] = label_set[idx];
          if(label_set[idx] > 0) curr_pos += 1;
          //printf("curr q:%s,d:%s,label:%d,curr-doc-len:%d,curr-window-size:%d\n",rel_set[idx][0].c_str(),rel_set[idx][1].c_str(),label_set[idx],curr_doc_len,curr_doc_window_num);
          top[3]->data[out_idx] = curr_doc_window_num;
        }
        //printf("curr q:%s,label-count:%d\n",rel_set[list_set[cline_ptr][0]][0].c_str(),curr_pos);
        cline_ptr = (cline_ptr + 1) % list_set.size();
        //printf("cline_ptr:%d\n",cline_ptr);
      }
      //printf("wbatch_size:%d, max_doc1_size:%d, window_size:%d\n",wbatch_size,max_doc1_len,window_size);
      top[0]->Resize(wbatch_size, 1, 1, data1_doc_len, true);
      top[1]->Resize(wbatch_size, 1, 1, window_size, true);
      top[0]->data = -1;
      top[1]->data = -1;
      index_t idx_top = 0;
      for(int s = 0 ; s < batch_size; ++ s){
        for (int i = 0; i < list_set[vbatches[s]].size(); ++i) {
          int out_idx = s * max_list + i;
          int idx = list_set[vbatches[s]][i];

          Layer<xpu>::global_data["data1"].push_back(rel_set[idx][0]);
          Layer<xpu>::global_data["data2"].push_back(rel_set[idx][1]);

          FillData(top[0]->data, top[0]->length, top[1]->data, top[1]->length, idx_top, top[3]->data[out_idx][0][0][0], idx);
          idx_top += top[3]->data[out_idx][0][0][0];
        }
      }
      //if(idx_top != wbatch_size)    printf("idx_top:%d\t,wbatch_size:%d\n",idx_top,wbatch_size);
      utils::Check(idx_top == wbatch_size, "idx_top not equal to wbatch_size.");
    }
    
    if (show_info) {
        top[0]->PrintShape("top0");
        top[1]->PrintShape("top1");
        top[2]->PrintShape("top2");
        top[3]->PrintShape("top3");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    if (mode == "list") {
      if (max_list != list_set[line_ptr].size()) {
        max_list = list_set[line_ptr].size();
      }
    }
    //printf("line_ptr:%d,max_list:%d\n",line_ptr,max_list);

    // Do reshape 
    this->Reshape(bottom, top);
    line_ptr += batch_size;
  }

  inline void FillData(mshadow::Tensor<xpu, 4> &top0_data, mshadow::Tensor<xpu, 2> &top0_length,
                       mshadow::Tensor<xpu, 4> &top1_data, mshadow::Tensor<xpu, 2> &top1_length, 
                       int top_idx,int step, int data_idx) {
      utils::Check(data1_set.count(rel_set[data_idx][0]), 
              "Map2WindowTextDataLayer: %s not in data1_set.", rel_set[data_idx][0].c_str());
      utils::Check(data2_set.count(rel_set[data_idx][1]), 
              "Map2WindowTextDataLayer: %s not in data2_set.", rel_set[data_idx][1].c_str());
      const vector<int> &data1 = data1_set[rel_set[data_idx][0]]; //query 
      const vector<int> &data2 = data2_set[rel_set[data_idx][1]]; //doc

      for(int i = 0 ; i < step; ++ i){
        for (int k = 0; k < data1.size(); ++k) {
            top0_data[top_idx+i][0][0][k] = data1[k];
        }
        if (fix_length) {
            top0_length[top_idx+i][0] = max_doc1_len;
        } else {
            top0_length[top_idx+i][0] = data1.size();
            utils::Check(data1.size() > 1, "Map2WindowTextDataLayer: data1 size:%d\n",data1.size());
        }

        //int curr_beg = i * window_size / 2;
        int curr_beg = i * window_size;
        int curr_len = window_size;
        if(curr_beg + curr_len >= data2.size()) curr_len = data2.size() - curr_beg;
        utils::Check(curr_len > 0, "FillData Wrong, curr_len:%d.",curr_len);
        for (int k = 0; k < curr_len; ++k) {
            top1_data[top_idx+i][0][0][k] = data2[curr_beg + k];
        }
        if (fix_length) {
            top1_length[top_idx + i][0] = window_size;
        } else {
            top1_length[top_idx + i][0] = curr_len;
        }
      }
      /*
      if(step == 2){
          printf("data1 length:%d\t,",data1.size());
          for(int i = 0 ; i < data1.size(); ++ i)   printf(" %d",data1[i]);
          printf("\ndata2 length:%d\t,",data2.size());
          for(int i = 0 ; i < data2.size(); ++ i)   printf(" %d",data2[i]);
          printf("\ntop0:\n");
          for(int i = 0 ; i < step; ++ i){
              printf("i:");
              for(int j = 0 ; j < top0_length[top_idx+i][0]; ++ j)  printf(" %.2f",top0_data[top_idx+i][0][0][j]);
              printf("\n");
          }
          printf("\ntop1:\n");
          for(int i = 0 ; i < step; ++ i){
              printf("i:");
              for(int j = 0 ; j < top1_length[top_idx+i][0]; ++ j)  printf(" %.2f",top1_data[top_idx+i][0][0][j]);
              printf("\n");
          }
          exit(0);
      }
      */
  } 
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    /*
    mshadow::Tensor<xpu, 4> top0_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    mshadow::Tensor<xpu, 4> top1_data = top[1]->data;
    mshadow::Tensor<xpu, 2> top1_length = top[1]->length;
    mshadow::Tensor<xpu, 1> top2_data = top[2]->data_d1();

    top0_data = -1;
    top1_data = -1;
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
    }
    */
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
  int window_size;
  int wbatch_size; // according to window_num to set new batch size( nbatch_size)
  int min_doc1_len;
  int max_doc1_len;
  int min_doc2_len;
  int max_doc2_len;
  int max_doc_len;
  int data1_doc_len;
  string mode;
  bool shuffle;
  bool speedup_list;
  bool fix_length;
  
  vector<vector<string> > rel_set;
  vector<int> label_set;
  vector<vector<int> > pair_set;
  vector<vector<int> > list_set;

  vector<string> ins_set_1;
  vector<string> ins_set_2;
  unordered_map<string, vector<string> > ins_map_12;
  unordered_map<string, vector<string> > ins_map_21;

  int line_count; /* line_count records the relation files line number: instance number */
  int line_ptr;
  int max_label;

  int max_list;
  std::default_random_engine rnd_generator;
};
template<typename xpu> unordered_map<string, vector<int> > Map2WindowTextDataLayer<xpu>::data1_set = unordered_map<string, vector<int> >();
template<typename xpu> unordered_map<string, vector<int> > Map2WindowTextDataLayer<xpu>::data2_set = unordered_map<string, vector<int> >();
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MAP_2_TEXTDATA_LAYER_INL_HPP_

