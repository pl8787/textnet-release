#ifndef TEXTNET_LAYER_MATCH_PHRASE_REP_LAYER_INL_HPP_
#define TEXTNET_LAYER_MATCH_PHRASE_REP_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class MatchPhraseRepLayer : public Layer<xpu>{
 public:
  MatchPhraseRepLayer(LayerType type) { this->layer_type = type; }
  virtual ~MatchPhraseRepLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["data_file"] = SettingV();
    this->defaults["feat_size"] = SettingV();
    this->defaults["batch_size"] = SettingV();
    this->defaults["max_doc_len"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "MatchPhraseRepLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchPhraseRepLayer:top size problem.");

    data_file = setting["data_file"].sVal();
    feat_size = setting["feat_size"].iVal();
    batch_size = setting["batch_size"].iVal();
    max_doc_len = setting["max_doc_len"].iVal();
    
    ReadTextData();
    
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
    data_set.Resize(mshadow::Shape4(line_count, 2, max_doc_len, feat_size));
    length_set.Resize(mshadow::Shape2(line_count, 2));
    label_set.Resize(mshadow::Shape1(line_count), 0);
    data_set = -1;
	length_set = 0;
    
    utils::Printf("Line count in file: %d\n", line_count);

    std::istringstream iss;
    int len_s1 = 0;
    int len_s2 = 0;
    int tmp = 0;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
        iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> label_set[i] >> len_s1 >> len_s2 >> tmp;
      utils::Check(tmp == feat_size, "MatchPhraseRepLayer: input error.");
	  length_set[i][0] = len_s1;
	  length_set[i][1] = len_s2;
      for (int j = 0; j < len_s1; ++j) {
        for (int f = 0; f < feat_size; ++f) {
          iss >> data_set[i][0][j][f];
        }
      }
      for (int j = 0; j < len_s2; ++j) {
        for (int f = 0; f < feat_size; ++f) {
          iss >> data_set[i][1][j][f];
        }
      }
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "MatchPhraseRepLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "MatchPhraseRepLayer:top size problem.");
	
    top[0]->Resize(batch_size, 2, max_doc_len, feat_size, true);
    top[1]->Resize(batch_size, 1, 1, 1, true);
	
	top[0]->PrintShape("top0");
	top[1]->PrintShape("top1");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top0_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    mshadow::Tensor<xpu, 1> top1_data = top[1]->data_d1();
    for (int i = 0; i < batch_size; ++i) {
      top0_data[i] = F<op::identity>(data_set[line_ptr]);
	  top0_length[i] = F<op::identity>(length_set[line_ptr]);
      top1_data[i] = label_set[line_ptr];
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
  int feat_size;
  mshadow::TensorContainer<xpu, 4> data_set;
  mshadow::TensorContainer<xpu, 2> length_set;
  mshadow::TensorContainer<xpu, 1> label_set;
  int line_count;
  int line_ptr;
};
}  // namespace layer
}  // namespace textnet
#endif  // 

