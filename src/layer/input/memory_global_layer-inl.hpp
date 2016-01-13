#ifndef TEXTNET_LAYER_MEMORY_GLOBAL_LAYER_INL_HPP_
#define TEXTNET_LAYER_MEMORY_GLOBAL_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class MemoryGlobalLayer : public Layer<xpu>{
 public:
  MemoryGlobalLayer(LayerType type) { this->layer_type = type; }
  virtual ~MemoryGlobalLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["min_doc_len"] = SettingV(1);
	this->defaults["random_select"] = SettingV(-1);
	this->defaults["reverse"] = SettingV(false);
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
                  "MemoryGlobalLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MemoryGlobalLayer:top size problem.");

    data_file = setting["data_file"].sVal();
    max_doc_len = setting["max_doc_len"].iVal();
    min_doc_len = setting["min_doc_len"].iVal();
	random_select = setting["random_select"].iVal();
	reverse = setting["reverse"].bVal();
    
    ReadTextData();

	for (int i = 0; i < line_count; ++i) {
		random_map.push_back(i);
	}
  }
  
  void ReadTextData() {
	if (this->initialed)
		return;
	this->initialed = true;

    utils::Printf("Open data file: %s\n", data_file.c_str());    
    std::vector<std::string> lines;
    std::ifstream fin(data_file.c_str());
    std::string s;
    utils::Check(fin.is_open(), "Open data file problem.");
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      lines.push_back(s);
    }
    fin.close();
	line_count = lines.size();
    utils::Printf("Line count in file: %d\n", line_count);

    std::istringstream iss;
	int label = 0;
    int len_s1 = 0;
    int len_s2 = 0;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> label >> len_s1 >> len_s2;

      label_set.push_back(label);

	  vector<int> s1(len_s1);
      for (int j = 0; j < len_s1; ++j) {
        iss >> s1[j];
      }
	  vector<int> s2(len_s2);
      for (int j = 0; j < len_s2; ++j) {
        iss >> s2[j];
      }

	  data1_set.push_back(s1);
	  data2_set.push_back(s2);
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "MemoryGlobalLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "MemoryGlobalLayer:top size problem.");
	
	utils::Check(max_doc_len > 0, "max_doc_len <= 0");

	if (random_select == -1) {
		memory_size = data1_set.size();
	} else {
		memory_size = random_select;
	}

    top[0]->Resize(1, memory_size, 1, max_doc_len, true);
	top[1]->Resize(1, memory_size, 1, max_doc_len, true);

	if (show_info) {
		top[0]->PrintShape("top0");
		top[1]->PrintShape("top1");
	}
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> top0_data = top[0]->data_d3();
    mshadow::Tensor<xpu, 1> top0_length = top[0]->length_d1();
    mshadow::Tensor<xpu, 3> top1_data = top[1]->data_d3();
    mshadow::Tensor<xpu, 1> top1_length = top[1]->length_d1();

	if (random_select != -1) {
	  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	  std::shuffle(random_map.begin(), random_map.end(), std::default_random_engine(seed)); 
	}

    for (int i = 0; i < memory_size; ++i) {
      int idx = random_map[i];
	  int s1_len = data1_set[idx].size();
	  int s2_len = data2_set[idx].size();
	  if (!reverse) {
        for (int j = 0; j < min(max_doc_len, s1_len); ++j) {
          top0_data[0][i][j] = data1_set[idx][j];
		}
        for (int j = 0; j < min(max_doc_len, s2_len); ++j) {
          top1_data[0][i][j] = data2_set[idx][j];
		}
	  } else {
        for (int j = 0; j < min(max_doc_len, s1_len); ++j) {
          top0_data[0][i][j] = data1_set[idx][s1_len - j - 1];
		}
        for (int j = 0; j < min(max_doc_len, s2_len); ++j) {
          top1_data[0][i][j] = data2_set[idx][s2_len - j - 1];
		}
	  }
      top0_length[i] = min(max_doc_len, s1_len);
	  top1_length[i] = min(max_doc_len, s2_len);
	}
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    
  }
  
  static vector<vector<int> > data1_set;
  static vector<vector<int> > data2_set;
  static vector<int> label_set;
  static bool initialed;

 protected:
  std::string data_file;
  int batch_size;
  int max_doc_len;
  int min_doc_len;
  int random_select;
  int memory_size;
  bool reverse;
  int line_count;

  vector<int> random_map;
};

template<typename xpu> bool MemoryGlobalLayer<xpu>::initialed = false;
template<typename xpu> vector<vector<int> > MemoryGlobalLayer<xpu>::data1_set = vector<vector<int> >();
template<typename xpu> vector<vector<int> > MemoryGlobalLayer<xpu>::data2_set = vector<vector<int> >();
template<typename xpu> vector<int> MemoryGlobalLayer<xpu>::label_set = vector<int>();


}  // namespace layer
}  // namespace textnet
#endif  // LAYER_MEMORY_GLOBAL_LAYER_INL_HPP_

