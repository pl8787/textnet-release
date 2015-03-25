#ifndef TEXTNET_ADAGRAD_UPDATER_INL_HPP_
#define TEXTNET_ADAGRAD_UPDATER_INL_HPP_

#include <iostream>
#include <mshadow/tensor.h>
#include "./updater.h"

namespace textnet {
namespace updater {

template<typename xpu, int dim>
class AdagradUpdater : public Updater<xpu, dim>{
 public:
  AdagradUpdater(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
	this->is_sparse = false;
    SetupUpdater(setting);
  }
  virtual ~AdagradUpdater(void) {}
  
  virtual void SetupUpdater(std::map<std::string, SettingV> &setting) {
    this->updater_type = setting["updater_type"].i_val;
    eps = setting["eps"].f_val;
    lr = setting["lr"].f_val;
    maxIteration = setting["max_iter"].f_val; 
    // base_lr = setting["lr"].f_val;
    // decay = setting["decay"].f_val;
    // momentum = setting["momentum"].f_val;
    iteration = 0;
	// wd = 0.0;
	// lr = base_lr;
  }
  struct square_root {
    MSHADOW_XINLINE static real_t Map(real_t a) {
        return sqrt(a);
    }
  };
  virtual void Update(mshadow::Tensor<xpu, dim> data, 
                      mshadow::Tensor<xpu, dim> diff) {

    iteration %= maxIteration;
    if (iteration == 0) {
      sumGradSquare.Resize(data.shape_, 0);
      // adaGrad.Resize(data.shape_, 0);
    }
    ++iteration;
                        
    // AdaptLearningRate();
    sumGradSquare += diff * diff;

    // mshadow::Tensor<xpu, dim> ada();
    // adaGrad = diff / (F<square_root>(sumGradSquare) + eps);
    data -= lr * (diff / (mshadow::expr::F<square_root>(sumGradSquare) + eps));
    // adaGrad += eps;
    // data -= lr * adaGrad;
    
    // if (momentum == 0.0) {
    //   data -= lr * (diff + wd * data);
    // } else {
    //   history = lr * (diff + wd * data) + momentum * history;
    //   data -= history;
    // }
  }
  
  virtual void UpdateSparse(mshadow::Tensor<xpu, dim> data, 
                            mshadow::Tensor<xpu, dim> diff, 
                            mshadow::Tensor<xpu, 1> idx) {

    iteration %= maxIteration;
    if (iteration == 0) {
      sumGradSquare.Resize(data.shape_, 0);
      // adaGrad.Resize(data.shape_, 0);
    }
    ++iteration;

    int w_idx = -1;
    for (int i = 0; i < idx.size(0); ++i) {
      w_idx = idx[i];
      sumGradSquare[w_idx] += diff[i] * diff[i];

      data.Slice(w_idx, w_idx+1) -= (lr * diff.Slice(i, i+1)) /  \
          (mshadow::expr::F<square_root>(sumGradSquare.Slice(w_idx, w_idx+1)) + eps);
      // adaGrad[w_idx] = diff[i] / (F<square_root>(sumGradSquare[w_idx]) + eps);
      // data[w_idx] -= lr * adaGrad[w_idx]; 
    }
  }
 protected: 
  // float momentum;
  // mshadow::TensorContainer<xpu, dim> history;
  int iteration, maxIteration;
  // float lr;
  // float base_lr;
  // float decay;
  // float wd;

  mshadow::TensorContainer<xpu, dim> sumGradSquare;
  // mshadow::TensorContainer<xpu, dim> adaGrad;
  float eps, lr;
};
}  // namespace updater
}  // namespace textnet
#endif  // TEXTNET_CONSTANT_INIT_INL_HPP_

