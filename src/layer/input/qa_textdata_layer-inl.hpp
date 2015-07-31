#ifndef TEXTNET_LAYER_QA_TEXTDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_QA_TEXTDATA_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

using namespace std;

namespace textnet {
namespace layer {

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

    utils::Check(mode == "batch" || mode == "pair" || mode == "list",
                  "QATextDataLayer: mode is one of batch, pair or list.");

    question_data_file = setting["question_data_file"].sVal();
    answer_data_file = setting["answer_data_file"].sVal();
    question_rel_file = setting["question_rel_file"].sVal();
    answer_rel_file = setting["answer_rel_file"].sVal();
    batch_size = setting["batch_size"].iVal();
    max_doc_len = setting["max_doc_len"].iVal();
    candids = setting["candids"].iVal();
    
    ReadTextData(question_data_file, question_data_set);
    
    line_ptr = 0;
  }
  
  void ReadTextData(string &data_file, unordered_map<string, vector<int> > &data_set) {
    utils::Printf("Open data file: %s\n", data_file.c_str());    

    std::ifstream fin(data_file.c_str());
    std::string s;
    utils::Check(fin, "Open data file problem.");

    istringstream iss;
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(s);
      iss >> key;
      data_set[key] = vector();
      while(!iss.eof()) 
        iss >> data_set[key] ;
    }
    fin.close();
    
    utils::Printf("Line count in file: %d\n", data_set.size());
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "QATextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "QATextDataLayer:top size problem.");
	
    utils::Check(batch_size > 0, "batch_size <= 0");
    utils::Check(max_doc_len > 0, "max_doc_len <= 0");

    top[0]->Resize(batch_size, 2, 1, max_doc_len, true);
    top[1]->Resize(batch_size, 1, 1, 1, true);
	
    top[0]->PrintShape("top0");
    top[1]->PrintShape("top1");
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    // mshadow::Tensor<xpu, 3> top0_data = top[0]->data_d3();
    // mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    // mshadow::Tensor<xpu, 1> top1_data = top[1]->data_d1();
    // for (int i = 0; i < batch_size; ++i) {
    //   top0_data[i] = F<op::identity>(data_set[line_ptr]);
	  // top0_length[i] = F<op::identity>(length_set[line_ptr]);
    //   top1_data[i] = label_set[line_ptr];
    //   line_ptr = (line_ptr + 1) % line_count;
    // }
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
  
  unordered_map<string, vector<int> > question_data_set;
  unordered_map<string, vector<int> > answer_data_set;

  vector<vector<string> > question_instance_set;
  vector<vector<string> > answer_instance_set;
  vector<int> label_set;

  int line_count;
  int line_ptr;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_QA_TEXTDATA_LAYER_INL_HPP_

