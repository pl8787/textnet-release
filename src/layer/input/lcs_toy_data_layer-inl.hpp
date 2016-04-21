#ifndef TEXTNET_LAYER_LCS_TOY_DATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_LCS_TOY_DATA_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include "stdlib.h"
#include <random>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class LcsToyDataLayer : public Layer<xpu>{
 public:
  LcsToyDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~LcsToyDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["min_doc_len"] = SettingV(1);
    this->defaults["shuffle"] = SettingV(true);
    this->defaults["shuffle_seed"] = SettingV(123);
    this->defaults["is_whole"] = SettingV(true); // whole or last

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["data_file"] = SettingV();
    this->defaults["batch_size"] = SettingV();
    this->defaults["max_doc_len"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "LcsToyDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LcsToyDataLayer:top size problem.");

    data_file = setting["data_file"].sVal();
    is_whole = setting["is_whole"].bVal();
    batch_size = setting["batch_size"].iVal();
    max_doc_len = setting["max_doc_len"].iVal();
    min_doc_len = setting["min_doc_len"].iVal();
    shuffle = setting["shuffle"].bVal();
    shuffle_seed = setting["shuffle_seed"].iVal();
    
    ReadTextData();
    line_ptr = 0;
  }

  void ReadTextData() {
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
    target_value.Resize(mshadow::Shape4(line_count, 1, max_doc_len, max_doc_len), 0);

    std::istringstream iss;
	for (int i = 0; i < line_count; ++i) {
      int len_s1 = 0;
      int len_s2 = 0;
	  int wid = -1;
	  char w;
	  vector<int> s1;
	  vector<int> s2;
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> len_s1 >> len_s2;
      for (int j = 0; j < len_s1; ++j) {
        iss >> w;
        wid = int(w) - 65;
        s1.push_back(wid);
      }
	  s1_data_set.push_back(s1);
      for (int j = 0; j < len_s2; ++j) {
        iss >> w;
        wid = int(w) - 65;
		s2.push_back(wid);
      }
	  s2_data_set.push_back(s2);

      float score = 0.f;
      for (int m = 0; m < len_s1; ++m) {
        for (int n = 0; n < len_s2; ++n) {
          iss >> score;
          target_value[i][0][m][n] = score;
        }
      }
	}
    for (int i = 0; i < line_count; ++i) {
      example_ids.push_back(i);
    }
    utils::Printf("Line count in file: %d\n", line_count);
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "LcsToyDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "LcsToyDataLayer:top size problem.");
    
    utils::Check(batch_size > 0, "LcsToyDataLayer: batch_size <= 0");
    utils::Check(max_doc_len > 0, "LcsToyDataLayer: max_doc_len <= 0");

    top[0]->Resize(batch_size, 2, 1, max_doc_len, true);
    if (is_whole) { 
      top[1]->Resize(batch_size, 1, max_doc_len, max_doc_len, true);
    } else {
      top[1]->Resize(batch_size, 1, 1, 1, true);
    } 
    
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
    mshadow::Tensor<xpu, 4> top0_data   = top[0]->data;
    mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    mshadow::Tensor<xpu, 4> top1_data   = top[1]->data;

	top0_data = -1;
	top0_length = -1;
	top1_data = -1;

    for (int i = 0; i < batch_size; ++i) {
      if (this->phrase_type == kTrain && line_ptr == 0) {
	    std::shuffle(example_ids.begin(), example_ids.end(), std::default_random_engine(shuffle_seed)); 
      }
      int example_id = example_ids[line_ptr];
      top0_length[i][0] = s1_data_set[example_id].size();
      top0_length[i][1] = s2_data_set[example_id].size();

      for (int k = 0; k < s1_data_set[example_id].size(); ++k) {
        top0_data[i][0][0][k] = s1_data_set[example_id][k];
      }
      for (int k = 0; k < s2_data_set[example_id].size(); ++k) {
        top0_data[i][1][0][k] = s2_data_set[example_id][k];
      }
      if (is_whole) {
        top1_data[i] = mshadow::expr::F<op::identity>(target_value[example_id]);
      } else {
        top1_data[i][0][0][0] = target_value[example_id][0][s1_data_set[example_id].size()-1][s2_data_set[example_id].size()-1];
      }

      line_ptr = (line_ptr + 1) % line_count;
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
  bool is_whole;

  vector<vector<int> > s1_data_set;
  vector<vector<int> > s2_data_set;
  mshadow::TensorContainer<xpu, 4, float> target_value;
  int line_ptr;
  int line_count;
  int shuffle_seed;
  std::vector<int> example_ids;
};
}  // namespace layer
}  // namespace textnet
#endif  // 

