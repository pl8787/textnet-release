#ifndef TEXTNET_LAYER_LABLE_FEAT_VALUE_LAYER_INL_HPP_
#define TEXTNET_LAYER_LABEL_FEAT_VALUE_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class LabelFeatValueLayer : public Layer<xpu>{
 public:
  LabelFeatValueLayer(LayerType type) { this->layer_type = type; }
  virtual ~LabelFeatValueLayer(void) {}
  
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
    this->defaults["feat_size"] = SettingV();
    
    Layer<xpu>::Require();
  }

  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "LabelFeatValueLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LabelFeatValueLayer:top size problem.");
                  
    data_file = setting["data_file"].s_val;
    batch_size = setting["batch_size"].i_val;
    feat_size = setting["feat_size"].i_val;

    ReadLabelFeatValueData();
    
    line_ptr = 0;
    sampler.Seed(shuffle_seed);
  }
  
  void ReadLabelFeatValueData() {
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

    data_set.Resize(mshadow::Shape4(line_count, 1, 1, feat_size), 0);
    label_set.Resize(mshadow::Shape1(line_count), -1);

    std::istringstream iss;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
	  iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> label_set[i];
      while (!iss.eof()) {
        int feature_idx = -1;
        float feature_value = -10000.f;
        iss >> feature_idx >> feature_value;
        utils::Check(feature_idx >= 0 && feature_idx < feat_size, "LabelFeatValueLayer: input error");
        utils::Check(feature_value != -10000.f, "LabelFeatValueLayer: input error");
        data_set[i][0][0][feature_idx] = feature_value;
      }
    }
    // gen example ids
    for (int i = 0; i < line_count; ++i) {
      example_ids.push_back(i);
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "LabelFeatValueLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LabelFeatValueLayer:top size problem.");
    top[0]->Resize(batch_size, 1, 1, feat_size, true);
    top[1]->Resize(batch_size, 1, 1, 1, true);
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top0_data = top[0]->data;
    mshadow::Tensor<xpu, 1> top1_data = top[1]->data_d1();

    utils::Check(top0_data.size(0) == batch_size, "ORC: error, need reshape.");
    for (int i = 0; i < batch_size; ++i) {
      if (this->phrase_type == kTrain && line_ptr == 0) {
        this->sampler.Shuffle(example_ids);
      }
      int example_id = example_ids[line_ptr];
      top0_data[i] = F<op::identity>(data_set[example_id]);
      top1_data[i] = label_set[example_id];
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
  int feat_size;
  mshadow::TensorContainer<xpu, 4> data_set;
  mshadow::TensorContainer<xpu, 1> label_set;
  std::vector<int> example_ids;
  int line_count, line_ptr, shuffle_seed;
  utils::RandomSampler sampler;
};
}  // namespace layer
}  // namespace textnet
#endif 

