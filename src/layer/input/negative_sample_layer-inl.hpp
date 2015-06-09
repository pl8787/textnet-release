#ifndef TEXTNET_LAYER_NEGATIVE_SAMPLE_LAYER_INL_HPP_
#define TEXTNET_LAYER_NEGATIVE_SAMPLE_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class NegativeSampleLayer : public Layer<xpu>{
 public:
  NegativeSampleLayer(LayerType type) { this->layer_type = type; }
  virtual ~NegativeSampleLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 4; } // x, pos, sample, y
  virtual int ParamNodeNum() { return 0; }
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["shuffle_seed"] = SettingV(123);

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["data_file"] = SettingV();
    this->defaults["batch_size"] = SettingV();
    this->defaults["max_doc_len"] = SettingV();
    this->defaults["negative_num"] = SettingV();
    this->defaults["position_num"] = SettingV();
    this->defaults["vocab_size"] = SettingV();
    this->defaults["word_freq_file"] = SettingV();
    this->defaults["sample_exp_factor"] = SettingV();
    
    Layer<xpu>::Require();
  }

  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "NegativeSampleLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "NegativeSampleLayer:top size problem.");
                  
    data_file = setting["data_file"].s_val;
    batch_size = setting["batch_size"].i_val;
    max_doc_len = setting["max_doc_len"].i_val;
    negative_num = setting["negative_num"].i_val;
    position_num = setting["position_num"].i_val;
    vocab_size = setting["vocab_size"].i_val;
    word_freq_file = setting["word_freq_file"].s_val;
    sample_exp_factor = setting["sample_exp_factor"].f_val;

    ReadSequenceData();
    construct_sample_pool();
    
    line_ptr = 0;
    sampler.Seed(shuffle_seed);
  }

  void construct_sample_pool() {
    sample_vector.clear();
    if (sample_exp_factor == 0.f) { // uniform sample
      for (int i = 0; i < vocab_size; ++i) {
        sample_vector.push_back(i);
      }
      return;
    }

    utils::Check(sample_exp_factor > 0.f && sample_exp_factor < 3, "NegativeSampleLayer: sample error.");
    std::ifstream ifs(word_freq_file.c_str());
    std::set<int> wids;
    while (!ifs.eof()) {
      int wid = -1, freq = -1;
      ifs >> wid >> freq;
      if (wid < 0) break;
      wids.insert(wid);
      // cout << wid << endl;
      utils::Check(wid >= 0 && wid < vocab_size, "NegativeSampleLayer: sample error.");
      utils::Check(freq >= 1, "NegativeSampleLayer: sample error.");
      float num = pow(freq, sample_exp_factor);
      if (num < 1.f) num = 1.1f;
      int i_num = floor(num);
      for (int i = 0; i < i_num; ++i) {
        sample_vector.push_back(wid);
      }
    }
    ifs.close();
    utils::Check(sample_vector.size() < 2^30, "NegativeSampleLayer: sample error.");
    utils::Check(wids.size() == vocab_size, "NegativeSampleLayer: sample error.");
    utils::Printf("NegativeSampleLayer: sample_vector.size():%d\n", sample_vector.size());
  }

  // return a list of negative samples
  void negative_sampler(vector<int> &negative_sample) {
    negative_sample.clear();
    for (int i = 0; i < negative_num; ++i) {
      int sample = this->sampler.NextUInt32(sample_vector.size());
      sample = sample_vector[sample];
      utils::Assert(sample >= 0 && sample < vocab_size, "NegativeSampleLayer: sampler error");
      negative_sample.push_back(sample);
    }
  }
  // return a list of prediction positions
  void position_sampler(int length, vector<int> &position_sample) {
    position_sample.clear();
    vector<int> shuffle_pos;
    for (int i = 0; i < length; ++i) {
      shuffle_pos.push_back(i);
    } 
    this->sampler.Shuffle(shuffle_pos);

    int end = length < position_num ? length : position_num;
    position_sample = vector<int>(shuffle_pos.begin(), shuffle_pos.begin() + end);
    sort(position_sample.begin(), position_sample.end());
    utils::Check(position_sample.size() == position_num, "NegativeSampleLayer: sampler error.");
    return;
    for (int i = length; i < position_num; ++i) { // this code has not been used
      position_sample.push_back(-1);
    }
    utils::Check(position_sample.size() == position_num, "NegativeSampleLayer: sampler error.");
  }
  
  void ReadSequenceData() {
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
    length.Resize(mshadow::Shape1(line_count), -1);

    std::istringstream iss;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
	  iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      // int label = -1;
      // iss >> label; // not used
      int j = 0;
      while (!iss.eof()) {
        iss >> data_set[i][0][0][j++];
      }
      length[i] = j;
      utils::Check(j <= max_doc_len, "NegativeSampleLayer: doc length error.");
    }

    // gen example ids
    for (int i = 0; i < line_count; ++i) {
      example_ids.push_back(i);
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "NegativeSampleLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "NegativeSampleLayer:top size problem.");
    top[0]->Resize(batch_size, 1, 1, max_doc_len, true);                // x
    top[1]->Resize(batch_size, position_num, 1, 1, true);               // pos
    top[2]->Resize(batch_size, position_num, 1, negative_num+1, true);  // sample
    top[3]->Resize(batch_size, position_num, negative_num+1, 1, true);  // y
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> x        = top[0]->data;
    mshadow::Tensor<xpu, 4> pos      = top[1]->data;
    mshadow::Tensor<xpu, 4> sample   = top[2]->data;
    mshadow::Tensor<xpu, 4> y        = top[3]->data;
    mshadow::Tensor<xpu, 2> x_length = top[0]->length;
    mshadow::Tensor<xpu, 2> sample_length = top[2]->length;

    // sample_length = negative_num + 1;

    utils::Check(x.size(0) == batch_size, "ORC: error, need reshape.");
    x = -1.f, pos = -1.f, sample = -1.f, y = -1.f, x_length = -1, sample_length = -1;
    
    vector<int> position_sample, negative_sample;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      if (this->phrase_type == kTrain && line_ptr == 0) {
        this->sampler.Shuffle(example_ids);
      }
      int example_id = example_ids[line_ptr];
      x[batch_idx] = F<op::identity>(data_set[example_id]);
      x_length[batch_idx][0] = length[example_id];
      
      position_sampler(length[example_id], position_sample);
      for (int pos_idx = 0; pos_idx < position_num; ++pos_idx) {
        pos[batch_idx][pos_idx][0][0] = position_sample[pos_idx];
        if (position_sample[pos_idx] == -1) {
          utils::Check(false, "NegativeSampleLayer: position must >= 0");
          continue;
        }
        int word_idx = x[batch_idx][0][0][position_sample[pos_idx]];
        sample[batch_idx][pos_idx][0][0] = word_idx;
        y[batch_idx][pos_idx][0][0] = 1;
        sample_length[batch_idx][pos_idx] = negative_num + 1;
        negative_sampler(negative_sample);
        for (int sample_idx = 0; sample_idx < negative_num; ++sample_idx) {
          sample[batch_idx][pos_idx][0][sample_idx+1] = negative_sample[sample_idx];
          y[batch_idx][pos_idx][sample_idx+1][0] = 0;
        }
      }
      
      line_ptr = (line_ptr + 1) % line_count;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
  }

 public:
  std::string data_file, word_freq_file;
  int batch_size, max_doc_len, negative_num, position_num, vocab_size;
  mshadow::TensorContainer<xpu, 4> data_set;
  mshadow::TensorContainer<xpu, 1, int> length;
  std::vector<int> example_ids;
  int line_count, line_ptr, shuffle_seed;
  float sample_exp_factor;
  utils::RandomSampler sampler;
  vector<int> sample_vector; 
};
}  // namespace layer
}
#endif 

