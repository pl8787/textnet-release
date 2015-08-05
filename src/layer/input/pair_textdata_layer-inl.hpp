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
    
    line_ptr = 0;
  }

  int ReadLabel(string &line) {
    std::istringstream iss;
    int label = -1;
    iss.clear();
    iss.seekg(0, iss.beg);
    iss.str(line);
    iss >> label;
    return label;
  }
  
  void ReadLine(int idx, string &line) {
    std::istringstream iss;
    int len_s1 = 0;
    int len_s2 = 0;
    iss.clear();
    iss.seekg(0, iss.beg);
    iss.str(line);
    iss >> label_set[idx] >> len_s1 >> len_s2;
    length_set[idx][0] = len_s1;
    length_set[idx][1] = len_s2;
    for (int j = 0; j < len_s1; ++j) {
      iss >> data_set[idx][0][j];
    }
    for (int j = 0; j < len_s2; ++j) {
      iss >> data_set[idx][1][j];
    }
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

    // Calculate total instances count
    total_ins_count = 0;
    for (int i = 0; i < line_count; ++i) {
      int label = ReadLabel(lines[i]);
      if (label == 0) total_ins_count += 1;
    }
    total_ins_count *= 2;
        
    data_set.Resize(mshadow::Shape3(total_ins_count, 2, max_doc_len));
    length_set.Resize(mshadow::Shape2(total_ins_count, 2));
    label_set.Resize(mshadow::Shape1(total_ins_count), 0);
    data_set = -1;
    length_set = 0;
    
    utils::Printf("Line count in file: %d\n", line_count);
    utils::Printf("Total instances count: %d\n", total_ins_count);

    int ins_idx = 0;
    int pos_idx = -1;
    for (int i = 0; i < line_count; ++i) {
      int label = ReadLabel(lines[i]);
      if (label == 1) {
        pos_idx = i;
      }
      else {
        ReadLine(ins_idx, lines[pos_idx]);
        ins_idx += 1;
        ReadLine(ins_idx, lines[i]);
        ins_idx += 1;
      }
    }
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
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> top0_data = top[0]->data_d3();
    mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    mshadow::Tensor<xpu, 1> top1_data = top[1]->data_d1();
    for (int i = 0; i < batch_size; ++i) {
      if (shuffle) {
        line_ptr = (rand() % (total_ins_count/2)) * 2;
      } 
      top0_data[i*2] = F<op::identity>(data_set[line_ptr]);
      top0_length[i*2] = F<op::identity>(length_set[line_ptr]);
      top1_data[i*2] = label_set[line_ptr];
      line_ptr += 1;
      top0_data[i*2+1] = F<op::identity>(data_set[line_ptr]);
      top0_length[i*2+1] = F<op::identity>(length_set[line_ptr]);
      top1_data[i*2+1] = label_set[line_ptr];
      line_ptr = (line_ptr + 1) % total_ins_count;
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
  mshadow::TensorContainer<xpu, 3> data_set;
  mshadow::TensorContainer<xpu, 2> length_set;
  mshadow::TensorContainer<xpu, 1> label_set;
  int line_count;
  int total_ins_count;
  int line_ptr;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_PAIR_TEXTDATA_LAYER_INL_HPP_

