#ifndef TEXTNET_LAYER_LETTER_TRIGRAM_LAYER_INL_HPP_
#define TEXTNET_LAYER_LETTER_TRIGRAM_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class LetterTrigramLayer : public Layer<xpu>{
 public:
  LetterTrigramLayer(LayerType type) { this->layer_type = type; }
  virtual ~LetterTrigramLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["length_mode"] = SettingV("1d");
    this->defaults["win_size"] = SettingV(0); // 0 for whole document
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["trigram_file"] = SettingV();
    this->defaults["trigram_count"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(),
                  "LetterTrigramLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LetterTrigramLayer:top size problem.");
                  
    trigram_file = setting["trigram_file"].sVal();
    trigram_count = setting["trigram_count"].iVal();
    length_mode = setting["length_mode"].sVal();
    win_size = setting["win_size"].iVal();

    utils::Check(length_mode == "1d" || length_mode == "2d",
                 "LetterTrigramLayer: error value of length_mode");

    ReadInitLetterTrigram();
  }

  void ReadInitLetterTrigram() {
    utils::Printf("Open trigram file: %s\n", trigram_file.c_str());    
    std::vector<std::string> lines;
    std::ifstream fin(trigram_file.c_str());
    std::string s;
    utils::Check(fin.is_open(), "Open trigram file problem.");
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      lines.push_back(s);
    }
    fin.close();
    
    line_count = lines.size();
    utils::Printf("Line count in file: %d\n", line_count);

    std::istringstream iss;
    int w_idx = -1;
    int t_idx = -1;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> w_idx;
      word_trigram_map[w_idx] = vector<int>();
      while (!iss.eof()) {
        iss >> t_idx;
        utils::Check(t_idx < trigram_count, "LetterTrigramLayer: trigram index greater than trigram_count.");
        word_trigram_map[w_idx].push_back(t_idx);
      }
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "LetterTrigramLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "LetterTrigramLayer:top size problem.");
    
    max_doc_len = bottom[0]->data.size(3);
    doc_count = bottom[0]->data.size(1);
    nbatch = bottom[0]->data.size(0);
                  
    if (win_size == 0) {
        max_doc_len = 1;
        feat_size = trigram_count;
    } else {
        max_doc_len -= (win_size - 1);
        feat_size = trigram_count * win_size;
    }

    if (length_mode == "1d") {
      top[0]->Resize(nbatch, doc_count, max_doc_len, feat_size, nbatch, 1, true);
    } else if (length_mode == "2d") {
      top[0]->Resize(nbatch, doc_count, max_doc_len, feat_size, nbatch, 2, true);
    }

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (nbatch != bottom[0]->data.size(0)) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top_len  = top[0]->length;
    
    top_data = 0;
    if (win_size == 0) {
      top_len = 1;
    } else {
      if (length_mode == "1d") {
        top_len = F<op::identity>(bottom_len);
        top_len -= (win_size - 1);
      } else if (length_mode == "2d") {
        top_len = 1;
        for (int i = 0; i < bottom_len.size(0); ++i) {
          top_len[i][1] = bottom_len[i][0] - win_size + 1;
        }
      }
    }

    int w_idx = -1;
    int t_idx = -1;
    for (int i = 0; i < nbatch; ++i) {
      for (int j = 0; j < doc_count; ++j) {
        int doc_len = bottom_len[i][j];
        utils::Check(doc_len >= 0, "LetterTrigram layer: length must be inited.");
        
        if (win_size == 0) {
          for (int k = 0; k < doc_len; ++k) {
            w_idx = (int)bottom_data[i][j][0][k];
            if (w_idx != -1) {
              for (int p = 0; p < word_trigram_map[w_idx].size(); ++p) {
                t_idx = word_trigram_map[w_idx][p];
                top_data[i][j][0][t_idx] += 1;
              }
            }
          }
        } else {
          for (int k = 0; k < doc_len - win_size + 1; ++k) {
            for (int c = 0; c < win_size; ++c) {
              w_idx = (int)bottom_data[i][j][0][k+c];
              if (w_idx != -1) {
                for (int p = 0; p < word_trigram_map[w_idx].size(); ++p) {
                  t_idx = word_trigram_map[w_idx][p];
                  top_data[i][j][k][c * trigram_count + t_idx] += 1;
                }
              }
            }
          }
        }

      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
  }

 protected:
  std::string trigram_file;
  int trigram_count;
  int nbatch;
  int max_doc_len;
  int doc_count;
  int line_count;
  int feat_size;
  int win_size;
  string length_mode;
  unordered_map<int, vector<int> > word_trigram_map;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LETTER_TRIGRAM_LAYER_INL_HPP_

