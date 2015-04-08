#ifndef TEXTNET_LAYER_NEXTBASKETDATA_LAYER_INL_HPP_
#define TEXTNET_LAYER_NEXTBASKETDATA_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/random.h"

namespace textnet {
namespace layer {

template<typename xpu>
class NextBasketDataLayer : public Layer<xpu>{
 public:
  NextBasketDataLayer(LayerType type) { this->layer_type = type; mul = 1000;}
  virtual ~NextBasketDataLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return top_node_num; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    utils::Check(setting.count("context_window"), "NextBasketDataLayer:setting problem."); 
    utils::Check(setting.count("max_session_len"), "NextBasketDataLayer:setting problem."); 
    utils::Check(setting.count("batch_size"), "NextBasketDataLayer:setting problem."); 
    utils::Check(setting.count("data_file"), "NextBasketDataLayer:setting problem."); 

    context_window = setting["context_window"].i_val;
    max_session_len = setting["max_session_len"].i_val;
    shuffle_seed = 113;
    if (setting.count("shuffle_seed")) shuffle_seed = setting["shuffle_seed"].i_val;
    top_node_num = context_window + 3; // label, label set and user
    data_file = setting["data_file"].s_val;
    batch_size = setting["batch_size"].i_val;
    
    utils::Check(bottom.size() == BottomNodeNum(), "NextBasketDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "NextBasketDataLayer:top size problem.");
                  
    ReadNextBasketData();
    example_ptr = 0;
    test_example_ptr = 0;
    sampler.Seed(shuffle_seed);
  }
  void splitByChar(const std::string &s, char c, std::vector<std::string> &vsResult)
  {
      using std::string;
      vsResult.clear();
      size_t uPos = 0, uPrePos = 0;
      string sTmp;
      for (; uPos < s.size(); uPos++)
      {
          if (s[uPos] == c)
          {
              sTmp = s.substr(uPrePos, uPos-uPrePos);
              if (!sTmp.empty())
                  vsResult.push_back(sTmp);
              uPrePos = uPos + 1;
          }
      }
      if (uPrePos < s.size())
      {
          sTmp = s.substr(uPrePos);
          vsResult.push_back(sTmp);
      }
  }
  
  void ReadNextBasketData() {
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

    std::istringstream iss;
    for (int i = 0; i < line_count; ++i) {
        std::vector<std::string> vsTab, vsComma;
        Example e;
        splitByChar(lines[i], ' ', vsTab); 
        assert(vsTab.size() == top_node_num-1);
        e.user = atoi(vsTab[1].c_str());
        splitByChar(vsTab[0], ',', vsComma); 
        for (int j = 0; j < vsComma.size(); ++j) {
            e.next_items.push_back(atoi(vsComma[j].c_str()));
        }
        utils::Check(vsTab.size() == context_window + 2, "NextBasketDataLayer: input data error.");
        for (int k = 2; k < vsTab.size(); ++k) {
            std::vector<int> basket;
            splitByChar(vsTab[k], ',', vsComma); 
            for (int j = 0; j < vsComma.size(); ++j) {
                basket.push_back(atoi(vsComma[j].c_str()));
            }
            e.context.push_back(basket);
        }
        dataset.push_back(e);
    }
	utils::Printf("Example count in file: %d\n", dataset.size());
    // gen example ids
    for (int i = 0; i < dataset.size(); ++i) {
      for (int j = 0; j < dataset[i].next_items.size(); ++j) {
        example_ids.push_back(i * mul + j);
      }
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                 "NextBasketDataLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                 "NextBasketDataLayer:top size problem.");
    top[0]->Resize(batch_size, 1, 1, 1, true);
    top[1]->Resize(batch_size, 1, 1, max_session_len, true);
    top[2]->Resize(batch_size, 1, 1, 1, true);
    for (int i = 0; i < context_window; ++i) {
      top[3+i]->Resize(batch_size, 1, 1, max_session_len, true);
    }
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    for (int node_idx = 0; node_idx < top.size(); ++node_idx) {
      top[node_idx]->data = 0;
    }

    for (int i = 0; i < batch_size; ++i) {
      int exampleIdx = 0, labelIdx = 0;
      if (this->phrase_type == kTrain) {
          if (example_ptr == 0) {
              this->sampler.Shuffle(example_ids);
          }
          exampleIdx = example_ids[example_ptr] / mul;
          labelIdx = example_ids[example_ptr] % mul;
          example_ptr = (example_ptr + 1) % example_ids.size();
      } else {
          exampleIdx = test_example_ptr;
          test_example_ptr = (test_example_ptr+1) % dataset.size();
      }
          
      top[0]->data[i][0][0][0] = dataset[exampleIdx].next_items[labelIdx]; // label
      for (int k = 0; k < dataset[exampleIdx].next_items.size(); ++k) {
        top[1]->data[i][0][0][k] = dataset[exampleIdx].next_items[k]; // label set
      }
      top[1]->length[i] = dataset[exampleIdx].next_items.size();
      top[2]->data[i][0][0][0] = dataset[exampleIdx].user;
      for (int w = 0; w < context_window; ++w) {
        for (int k = 0; k < dataset[exampleIdx].context[w].size(); ++k) {
          top[w+3]->data[i][0][0][k] = dataset[exampleIdx].context[w][k];
        }
        top[w+3]->length[i] = dataset[exampleIdx].context[w].size();
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    // nothing to do
  }

  struct Example {
      int user;
      std::vector<int> next_items;
      std::vector<std::vector<int> > context;
  };


 protected:
  std::string data_file;
  int batch_size, top_node_num, context_window;
  int max_session_len;
  int example_ptr, test_example_ptr;
  
  int mul;

  std::vector<Example> dataset;
  
  std::vector<int> example_ids;
  // mshadow::TensorContainer<xpu, 4> data_set;
  // mshadow::TensorContainer<xpu, 1> label_set;
  int line_count;
  int shuffle_seed;
  utils::RandomSampler sampler;
};
}  // namespace layer
}  // namespace textnet
#endif 

