#ifndef TEXTNET_SGDSTEP_UPDATER_INL_HPP_
#define TEXTNET_SGDSTEP_UPDATER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <mshadow/tensor.h>
#include "./updater.h"

namespace textnet {
namespace updater {

template<typename xpu, int dim>
class SGDStepUpdater : public Updater<xpu, dim>{
 public:
  SGDStepUpdater(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupUpdater(setting);
  }
  virtual ~SGDStepUpdater(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    this->defaults["decay"] = SettingV(0.1f);
    this->defaults["momentum"] = SettingV(0.9f);
    this->defaults["l2"] = SettingV(0.0f);
    this->defaults["steps"] = SettingV("");
	this->defaults["steps_lr"] = SettingV("");

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["lr"] = SettingV();
    
    Updater<xpu, dim>::Require(setting);
  }
  
  virtual void SetupUpdater(std::map<std::string, SettingV> &setting) {
    Updater<xpu, dim>::SetupUpdater(setting);
    
    this->updater_type = setting["updater_type"].iVal();
    base_lr = setting["lr"].fVal();
    decay = setting["decay"].fVal();
    momentum = setting["momentum"].fVal();
    l2 = setting["l2"].fVal();
	steps_str = setting["steps"].sVal();
	steps_lr_str = setting["steps_lr"].sVal();

    int s = 0;
	istringstream iss;
	iss.str(steps_str);
	while (iss >> s) {
	  steps.push_back(s);
	}

	float f = 0;
	istringstream iss_lr;
	iss_lr.str(steps_lr_str);
	while (iss_lr >> f) {
	  steps_lr.push_back(f);
	}

	cur_step = 0;

    iteration = 0;
	lr = base_lr;
	
	// for debug
	cout << "base_lr: " << base_lr << endl;
	cout << "decay: " << decay << endl;
	cout << "momentum: " << momentum << endl;
	cout << "l2: " << l2 << endl;
	cout << steps[0] << steps[1] <<endl;
  }
  
  virtual void Update(mshadow::Tensor<xpu, dim> data, 
                      mshadow::Tensor<xpu, dim> diff) {
    if (momentum != 0.0 && iteration == 0) {
      history.Resize(data.shape_, 0);
    }
                        
    AdaptLearningRate();
    iteration++;
    
    if (momentum == 0.0) {
      data -= lr * (diff + l2 * data);
    } else {
      history = lr * (diff + l2 * data) + momentum * history;
      data -= history;
    }
  }
  
  virtual void UpdateSparse(mshadow::Tensor<xpu, dim> data, 
                            mshadow::Tensor<xpu, dim> diff, 
                            mshadow::Tensor<xpu, 1> idx) {
    if (momentum != 0.0 && iteration == 0) {
      history.Resize(data.shape_, 0);
    }
    
    AdaptLearningRate();
    iteration++;

    if (momentum == 0.0) {
      int w_idx = -1;
      for (int i = 0; i < idx.size(0); ++i) {
        w_idx = idx[i];
        data[w_idx] -= lr * (diff[i] + l2 * data[w_idx]);
      }
    } else {
      int w_idx = -1;
      for (int i = 0; i < idx.size(0); ++i) {
        w_idx = idx[i];
        history[w_idx] = lr * (diff[i] + l2 * data[w_idx]) + momentum * history[w_idx];
        data[w_idx] -= history[w_idx];
      }
    }
  }
  
  virtual void AdaptLearningRate() {
	if (cur_step >= steps.size()) return;
	if (iteration == steps[cur_step]) {
	  if (steps_lr.size() == 0) {
	    lr *= decay;
	    ++cur_step;
	  } else {
		lr = steps_lr[cur_step];
		++cur_step;
	  }
	}
  }
  
 protected: 
  float momentum;
  mshadow::TensorContainer<xpu, dim> history;
  int iteration;
  float lr;
  float base_lr;
  float decay;
  float l2;
  vector<int> steps;
  vector<float> steps_lr;
  int cur_step;
  string steps_str;
  string steps_lr_str;
};
}  // namespace updater
}  // namespace textnet
#endif  // TEXTNET_CONSTANT_INIT_INL_HPP_

