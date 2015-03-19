#ifndef TEXTNET_SGD_UPDATER_INL_HPP_
#define TEXTNET_SGD_UPDATER_INL_HPP_

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
    SetupUpdater(setting);
  }
  virtual ~SGDUpdater(void) {}
  
  virtual void SetupUpdater(std::map<std::string, SettingV> &setting) {
    this->updater_type = setting["updater_type"].i_val;
    lr = setting["lr"].f_val;
    decay = setting["decay"].f_val;
    momentum = setting["momentum"].f_val;
    iteration = 0;
  }
  
  virtual void Update(mshadow::Tensor<xpu, dim> data, 
                      mshadow::Tensor<xpu, dim> diff) {
    if (momentum != 0.0 && iteration == 0) {
      history.Resize(data.shape_, 0);
    }
                        
    AdaptLearningRate();
    iteration++;
    
    if (momentum == 0.0) {
      data -= lr * diff;
    } else {
      history = diff + momentum * history;
      data -= lr * history;
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
        data[w_idx] -= lr * diff[i];
      }
    } else {
      int w_idx = -1;
      for (int i = 0; i < idx.size(0); ++i) {
        w_idx = idx[i];
        history[w_idx] = diff[i] + momentum * history[w_idx];
        data[w_idx] -= lr * history[w_idx];
      }
    }
  }
  
  virtual void AdaptLearningRate() {
    lr = lr - decay * iteration;
  }
  
 protected: 
  float momentum;
  mshadow::TensorContainer<xpu, dim> history;
  int iteration;
  float lr;
  float decay;
};
}  // namespace updater
}  // namespace textnet
#endif  // TEXTNET_CONSTANT_INIT_INL_HPP_

