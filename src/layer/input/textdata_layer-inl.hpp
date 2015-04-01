#ifndef TEXTNET_LAYER_TEXTDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_TEXTDATA_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class TextDataLayer : public Layer<xpu>{
 public:
  TextDataLayer(LayerType type) { this->layer_type = type; }
  virtual ~TextDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "TextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "TextDataLayer:top size problem.");
                  
    data_file = setting["data_file"].s_val;
    batch_size = setting["batch_size"].i_val;
    max_doc_len = setting["max_doc_len"].i_val;
    min_doc_len = setting["min_doc_len"].i_val;
    
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
    data_set.Resize(mshadow::Shape3(line_count, 2, max_doc_len));
    label_set.Resize(mshadow::Shape1(line_count), 0);
    data_set = -1;
    
    utils::Printf("Line count in file: %d\n", line_count);

    std::istringstream iss;
    int len_s1 = 0;
    int len_s2 = 0;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
        iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> label_set[i] >> len_s1 >> len_s2;
      for (int j = 0; j < len_s1; ++j) {
        iss >> data_set[i][0][j];
      }
      for (int j = 0; j < len_s2; ++j) {
        iss >> data_set[i][1][j];
      }
    }
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "TextDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "TextDataLayer:top size problem.");
	
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
    mshadow::Tensor<xpu, 3> top0_data = top[0]->data_d3();
    mshadow::Tensor<xpu, 1> top1_data = top[1]->data_d1();
    for (int i = 0; i < batch_size; ++i) {
      top0_data[i] = F<op::identity>(data_set[line_ptr]);
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
  mshadow::TensorContainer<xpu, 3> data_set;
  mshadow::TensorContainer<xpu, 1> label_set;
  int line_count;
  int line_ptr;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_TEXTDATA_LAYER_INL_HPP_

