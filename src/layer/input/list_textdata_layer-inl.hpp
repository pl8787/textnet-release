#ifndef TEXTNET_LAYER_LIST_TEXTDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_LIST_TEXTDATA_LAYER_INL_HPP_

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
class ListTextDataLayer : public Layer<xpu>{
 public:
  ListTextDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~ListTextDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["min_doc_len"] = SettingV(1);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["data_file"] = SettingV();
    this->defaults["max_doc_len"] = SettingV();
    
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
    max_doc_len = setting["max_doc_len"].iVal();
    min_doc_len = setting["min_doc_len"].iVal();
    
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

	// Calculate max list length
	max_list_len = 0;
	int list_len = 0;
	for (int i = 0; i < line_count; ++i) {
	  int label = ReadLabel(lines[i]);
	  if (label == 1) {
		max_list_len = max(max_list_len, list_len);
        list_len = 1;
	  } else {
		list_len += 1;
	  }
	}
		
    data_set.Resize(mshadow::Shape3(line_count, 2, max_doc_len));
    length_set.Resize(mshadow::Shape2(line_count, 2));
    label_set.Resize(mshadow::Shape1(line_count), 0);
    data_set = -1;
	length_set = 0;
    
    utils::Printf("Line count in file: %d\n", line_count);
	utils::Printf("Max list length: %d\n", max_list_len);

    for (int i = 0; i < line_count; ++i) {
	  ReadLine(i, lines[i]);
	}
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "TextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "TextDataLayer:top size problem.");
	
	utils::Check(max_doc_len > 0, "max_doc_len <= 0");

    top[0]->Resize(max_list_len, 2, 1, max_doc_len, true);
    top[1]->Resize(max_list_len, 1, 1, 1, true);
	
	top[0]->PrintShape("top0");
	top[1]->PrintShape("top1");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> top0_data = top[0]->data_d3();
    mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    mshadow::Tensor<xpu, 1> top1_data = top[1]->data_d1();
	bool list_begin = true;
	bool list_end = false;
    for (int i = 0; i < max_list_len; ++i) {
	  if ( list_end || (label_set[line_ptr] == 1 && !list_begin) ) {
        top0_data[i] = -1;
		top0_length[i] = 0;
		top1_data[i] = -1;
		list_end = true;
		// utils::Printf(".");
	  } else {
        top0_data[i] = F<op::identity>(data_set[line_ptr]);
	    top0_length[i] = F<op::identity>(length_set[line_ptr]);
        top1_data[i] = label_set[line_ptr];
        line_ptr = (line_ptr + 1) % line_count;
	    if (list_begin) list_begin = false;
		// utils::Printf("x");
	  }
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
  int max_list_len;
  int line_ptr;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LIST_TEXTDATA_LAYER_INL_HPP_

