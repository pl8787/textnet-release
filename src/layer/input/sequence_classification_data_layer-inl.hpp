#ifndef TEXTNET_LAYER_SEQUENCECLASSIFICATIONDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_SEQUENCECLASSIFICATIONDATA_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class SequenceClassificationDataLayer : public Layer<xpu>{
 public:
  SequenceClassificationDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~SequenceClassificationDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["shuffle_seed"] = SettingV(123);

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
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "SequenceClassificationDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "SequenceClassificationDataLayer:top size problem.");
                  
    data_file = setting["data_file"].s_val;
    batch_size = setting["batch_size"].i_val;
    max_doc_len = setting["max_doc_len"].i_val;

    ReadSequenceClassificationData();
    
    line_ptr = 0;
    sampler.Seed(shuffle_seed);
  }
  
  void ReadSequenceClassificationData() {
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
	utils::Printf("Line count in file: %d\n", line_count);

    data_set.Resize(mshadow::Shape4(line_count, 1, 1, max_doc_len), 0);
    label_set.Resize(mshadow::Shape1(line_count), -1);
    length.Resize(mshadow::Shape1(line_count), -1);

    std::istringstream iss;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
	    iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> label_set[i];
      int j = 0;
      while (!iss.eof()) {
        iss >> data_set[i][0][0][j++];
      }
      length[i] = j;
      utils::Check(j <= max_doc_len, "SequenceClassificationDataLayer: doc length error.");
    }

    // gen example ids
    for (int i = 0; i < line_count; ++i) {
      example_ids.push_back(i);
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "SequenceClassificationDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "SequenceClassificationDataLayer:top size problem.");
    top[0]->Resize(batch_size, 1, 1, 1, true);
    top[1]->Resize(batch_size, 1, 1, max_doc_len, true);
	if (show_info) {
		top[0]->PrintShape("top0");
		top[1]->PrintShape("top1");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 1> top0_data = top[0]->data_d1();
    mshadow::Tensor<xpu, 4> top1_data = top[1]->data;
    mshadow::Tensor<xpu, 2> top1_length = top[1]->length;

    utils::Check(top1_data.size(0) == batch_size, "ORC: error, need reshape.");
    for (int i = 0; i < batch_size; ++i) {
      if (this->phrase_type == kTrain && line_ptr == 0) {
        this->sampler.Shuffle(example_ids);
      }
      int example_id = example_ids[line_ptr];
      top0_data[i] = label_set[example_id];
      top1_data[i] = F<op::identity>(data_set[example_id]);
      top1_length[i][0] = length[example_id];
      line_ptr = (line_ptr + 1) % line_count;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
  }

 public:
  std::string data_file;
  int batch_size;
  int max_doc_len;
  mshadow::TensorContainer<xpu, 4> data_set;
  mshadow::TensorContainer<xpu, 1> label_set;
  mshadow::TensorContainer<xpu, 1, int> length;
  std::vector<int> example_ids;
  int line_count, line_ptr, shuffle_seed;
  utils::RandomSampler sampler;
};
}  // namespace layer
}  // namespace textnet
#endif 

