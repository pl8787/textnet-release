#ifndef TEXTNET_SGD_UPDATER_INL_HPP_
#define TEXTNET_SGD_UPDATER_INL_HPP_

#include <iostream>
#include <mshadow/tensor.h>
#include "./updater.h"

namespace textnet {
namespace updater {

template<typename xpu, int dim>
class SGDUpdater : public Updater<xpu, dim>{
 public:
  SGDUpdater(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
	this->is_sparse = false;
    SetupUpdater(setting);
  }
  virtual ~SGDUpdater(void) {}
  
  virtual void SetupUpdater(std::map<std::string, SettingV> &setting) {
    this->updater_type = setting["updater_type"].i_val;
    base_lr = setting["lr"].f_val;
    decay = setting["decay"].f_val;
    momentum = setting["momentum"].f_val;
    iteration = 0;
	wd = 0.04;
	lr = base_lr;
  }
  
  virtual void Update(mshadow::Tensor<xpu, dim> data, 
                      mshadow::Tensor<xpu, dim> diff) {
    if (momentum != 0.0 && iteration == 0) {
      history.Resize(data.shape_, 0);
    }
                        
    AdaptLearningRate();
    iteration++;
    
    if (momentum == 0.0) {
      data -= lr * (diff + wd * data);
    } else {
      history = lr * (diff + wd * data) + momentum * history;
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
        data[w_idx] -= lr * (diff[i] + wd * data[w_idx]);
      }
    } else {
      int w_idx = -1;
      for (int i = 0; i < idx.size(0); ++i) {
        w_idx = idx[i];
        history[w_idx] = lr * (diff[i] + wd * data[w_idx]) + momentum * history[w_idx];
        data[w_idx] -= history[w_idx];
      }
    }
  }
  
  virtual void AdaptLearningRate() {
	if (lr < 0.1 * base_lr) return;
    lr = base_lr * (1.0 - decay * iteration);
  }
  
 protected: 
  float momentum;
  mshadow::TensorContainer<xpu, dim> history;
  int iteration;
  float lr;
  float base_lr;
  float decay;
  float wd;
};
}  // namespace updater
}  // namespace textnet
#endif  // TEXTNET_CONSTANT_INIT_INL_HPP_

