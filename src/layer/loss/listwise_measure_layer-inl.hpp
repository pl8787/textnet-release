#ifndef TEXTNET_LAYER_LISTWISE_MEASURE_LAYER_INL_HPP_
#define TEXTNET_LAYER_LISTWISE_MEASURE_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

using namespace std;

namespace textnet {
namespace layer {

bool list_cmp(const pair<float, float> &x1, const pair<float, float> &x2) {
  return x1.first > x2.first; // sort decrease
}


template<typename xpu>
class ListwiseMeasureLayer : public Layer<xpu>{
 public:
  ListwiseMeasureLayer(LayerType type) { this->layer_type = type; }
  virtual ~ListwiseMeasureLayer(void) {}
  
  virtual int BottomNodeNum() { return 2; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["k"] = SettingV(1.0f);
    this->defaults["col"] = SettingV(0);
	this->defaults["batch_size"] = SettingV(1);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["method"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
                            
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ListwiseMeasureLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ListwiseMeasureLayer:top size problem.");
    k = setting["k"].iVal();
    method = setting["method"].sVal();
    col = setting["col"].iVal();
	batch_size = setting["batch_size"].iVal();
    
    utils::Check(method == "MRR" || method == "P@k" || method == "nDCG@k", 
                  "Parameter [method] must be MRR or P@k or nDCG@k.");
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ListwiseMeasureLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ListwiseMeasureLayer:top size problem.");
    nbatch = bottom[0]->data.size(0);    
    top[0]->Resize(1, 1, 1, 1, true);

	utils::Check(nbatch % batch_size == 0,
					"ListwiseMeasureLayer:nbatch %% batch_size != 0.");
	list_size = nbatch / batch_size;

    if (show_info) {
		bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    nbatch = bottom[0]->data.size(0);
	list_size = nbatch / batch_size;
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> bottom0_data = bottom[0]->data_d2();
    mshadow::Tensor<xpu, 1> bottom1_data = bottom[1]->data_d1();
    mshadow::Tensor<xpu, 1> top_data = top[0]->data_d1();
    
	top_data = 0.0;

	for (int s = 0; s < batch_size; ++s) {
      vector< pair<float, float> > score_list;
      float score = 0.0;
      float check = 0.0;

      for (int i = 0; i < list_size; ++i) {
	    int idx = s * list_size + i;
        if (bottom1_data[idx] == -1)
          break;
        score_list.push_back( make_pair(bottom0_data[idx][col], bottom1_data[idx]) );
      }

	  // shuffle before sort!
	  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	  std::shuffle(score_list.begin(), score_list.end(), std::default_random_engine(seed)); 

      sort(score_list.begin(), score_list.end(), list_cmp);

      int score_list_len = score_list.size();

      if (method == "MRR") {
        for (int i = 0; i < score_list_len; ++i) {
          if (score_list[i].second == 1)
            score += 1.0 / (i+1);
          check += score_list[i].second;
        }
        utils::Check(check==1, "Not a valid list.");
      } else if (method == "P@k") {
        for (int i = 0; i < min(k, score_list_len); ++i) {
          if (score_list[i].second == 1)
            score = 1.0;
        }
      } else if (method == "nDCG@k") {
        for (int i = 0; i < min(k, score_list_len); ++i) {
          if (score_list[i].second == 1)
            score += 1.0 / log(i+2);
        }
      }
      top_data[0] += score;
	}

	top_data[0] /= batch_size;
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

  }
  
 protected:
  int nbatch;
  int k;
  string method;
  int col;
  int batch_size;
  int list_size;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_LISTWISE_MEASURE_LAYER_INL_HPP_

