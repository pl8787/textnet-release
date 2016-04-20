#ifndef TEXTNET_LAYER_EMBEDDING_LAYER_INL_HPP_
#define TEXTNET_LAYER_EMBEDDING_LAYER_INL_HPP_

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
class EmbeddingLayer : public Layer<xpu>{
 public:
  EmbeddingLayer(LayerType type) { this->layer_type = type; read_embed_done = false; }
  virtual ~EmbeddingLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 1; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["pad_value"] = SettingV(0.f);
    this->defaults["embedding_file"] = SettingV("");
    this->defaults["update_indication_file"] = SettingV(""); // id (0 or 1), 1 is for update, 0 is for un update
    this->defaults["length_mode"] = SettingV("embedding"); // embedding or kernel or featmap
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["feat_size"] = SettingV();
    this->defaults["word_count"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(),
                  "EmbeddingLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "EmbeddingLayer:top size problem.");
                  
    embedding_file = setting["embedding_file"].sVal();
    update_indication_file = setting["update_indication_file"].sVal();
    feat_size = setting["feat_size"].iVal();
    word_count = setting["word_count"].iVal();
    pad_value = setting["pad_value"].fVal();
    length_mode = setting["length_mode"].sVal();

    utils::Check(length_mode == "embedding" || length_mode == "kernel" || length_mode == "featmap",
                 "EmbeddingLayer: error value of length_mode");

    // shared layer only read one embedding files
    // saving running time 
    if (!read_embed_done) {
      this->params.resize(1);
      // No need allocate diff memory
      this->params[0].need_diff = false;
      this->params[0].is_sparse = true;
      this->params[0].Resize(word_count, feat_size, 1, 1);
    
      std::map<std::string, SettingV> w_setting = *setting["w_filler"].mVal();
      this->params[0].initializer_ = 
          initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
            w_setting, this->prnd_);
      this->params[0].Init();   
  
      std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
      this->params[0].updater_ = 
          updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
            w_updater, this->prnd_);

      // Check if embedding file is empty
      if(!embedding_file.empty()) {
        read_embed_done = true;
        ReadInitEmbedding();
      }
    } else {
      utils::Printf("EmbeddingLayer: Read Embeddings done, skip.");
    }

    if(!update_indication_file.empty()) {
      ReadUpdateIndicationFile();
    }

  }

  void ReadUpdateIndicationFile() {
    utils::Printf("EmbeddingLayer: Open indication file: %s\n", update_indication_file.c_str());
    std::ifstream ifs(update_indication_file.c_str());
    utils::Check(ifs.is_open(), "EmbeddingLayer: Open indication file problem.");
    int word_idx, indication;
    while (!ifs.eof()) {
      ifs >> word_idx >> indication;
      if (indication == 0) {
        unupdate_words.insert(word_idx);
      }
    }
    utils::Printf("EmbeddingLayer: # of un update words: %d\n", unupdate_words.size());
  }

  void ReadInitEmbedding() {
    utils::Printf("Open embedding file: %s\n", embedding_file.c_str());    
    std::vector<std::string> lines;
    std::ifstream fin(embedding_file.c_str());
    std::string s;
    utils::Check(fin.is_open(), "Open embedding file problem.");
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      lines.push_back(s);
    }
    fin.close();
    
    line_count = lines.size();
    utils::Printf("Line count in file: %d\n", line_count);

    std::istringstream iss;
    int w_idx = 0;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> w_idx;
      int j = 0;
      while (!iss.eof()) {
	utils::Check(j < feat_size, "EmbeddingLayer: init embedding error. More %d. %d.", i, j);
        iss >> this->params[0].data[w_idx][j++][0][0];
      }
      utils::Check(j == feat_size, "EmbeddingLayer: init embedding error. Less %d. %d.", i, j);
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "EmbeddingLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "EmbeddingLayer:top size problem.");
    
    max_doc_len = bottom[0]->data.size(3);
    doc_count = bottom[0]->data.size(1);
    nbatch = bottom[0]->data.size(0);
                  
    // For one dimention length, such as sentence length
    if (length_mode == "embedding") {
      top[0]->Resize(nbatch, doc_count, max_doc_len, feat_size, true);
    // For three dimention length, (channel_in=1, kernel_y=sentence length, kernel_x=1)
    } else if (length_mode == "kernel") {
      top[0]->Resize(nbatch, doc_count, max_doc_len, feat_size, nbatch, 3, true);
    // For two dimention length, (kernel_y=sentence length, kernel_x=feat size)
    } else if (length_mode == "featmap") {
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
    mshadow::Tensor<xpu, 2> weight_data = this->params[0].data_d2();
    
    // fill all top data to pad_value
    top_data = pad_value;
    if (length_mode == "embedding") {
      top_len = F<op::identity>(bottom_len);
    } else if (length_mode == "kernel") {
      top_len = 1;
      for (int i = 0; i < bottom_len.size(0); ++i) {
        top_len[i][1] = bottom_len[i][0];
      }
    } else if (length_mode == "featmap") {
      for (int i = 0; i < bottom_len.size(0); ++i) {
        top_len[i][0] = bottom_len[i][0];
        top_len[i][1] = feat_size;
      }
    }

    int w_idx = -1;
    for (int i = 0; i < nbatch; ++i) {
      for (int j = 0; j < doc_count; ++j) {
        int doc_len = bottom_len[i][j];
        utils::Check(doc_len >= 0, "Embedding layer: length must be inited.");
        for (int k = 0; k < doc_len; ++k) {
          w_idx = (int)bottom_data[i][j][0][k];
          if (w_idx != -1) {
            top_data[i][j][k] = F<op::identity>(weight_data[w_idx]);
          }
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    
    if (this->prop_grad[0]) {
      std::map<int, int> idx_map;
      int w_idx = -1;
      int inc = 0;
      for (int i = 0; i < nbatch; ++i) {
        for (int j = 0; j < doc_count; ++j) {
          int doc_len = bottom_len[i][j];
          utils::Check(doc_len >= 0, "Embedding layer: length must be inited.");
          for (int k = 0; k < doc_len; ++k) {
            w_idx = (int)bottom_data[i][j][0][k];
            if (unupdate_words.find(w_idx) != unupdate_words.end()) {
              continue;
            }
            if (w_idx != -1 && !idx_map.count(w_idx)) {
              idx_map[w_idx] = inc;
              inc++;
            }
          }
        }
      }
      
      this->params[0].diff.Resize(mshadow::Shape4(inc, feat_size, 1, 1), 0);
      this->params[0].idx.Resize(mshadow::Shape1(inc), 0);
      mshadow::Tensor<xpu, 2> weight_diff = this->params[0].diff_d2();
      mshadow::Tensor<xpu, 1> weight_idx = this->params[0].idx;
      
      for (std::map<int,int>::iterator it=idx_map.begin(); 
            it!=idx_map.end(); ++it) {
        weight_idx[it->second] = it->first;      
      }
      
      for (int i = 0; i < nbatch; ++i) {
        for (int j = 0; j < doc_count; ++j) {
          int doc_len = bottom_len[i][j];
          utils::Check(doc_len >= 0, "Embedding layer: length must be inited.");
          for (int k = 0; k < doc_len; ++k) {
            w_idx = (int)bottom_data[i][j][0][k];
            if (unupdate_words.find(w_idx) != unupdate_words.end()) {
              continue;
            }
            if (w_idx != -1) {
              weight_diff[idx_map[w_idx]] += top_diff[i][j][k];
            }
          }
        }
      }
    }
  }
 protected:
  std::string embedding_file, update_indication_file;
  int feat_size;
  int word_count;
  int max_doc_len;
  int doc_count;
  int nbatch;
  int line_count;
  float pad_value;
  bool read_embed_done;
  std::set<int> unupdate_words;
  string length_mode;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_EMBEDDING_LAYER_INL_HPP_

