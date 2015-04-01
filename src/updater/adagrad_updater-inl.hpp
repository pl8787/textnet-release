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
    utils::Check(setting.count("updater_type"), "AdaGradUpdater: parameter error.");
    utils::Check(setting.count("eps"), "AdaGradUpdater: parameter error.");
    utils::Check(setting.count("lr"), "AdaGradUpdater: parameter error.");

    this->updater_type = setting["updater_type"].i_val;
    eps = setting["eps"].f_val;
    lr = setting["lr"].f_val;

    max_iter = -1; 
    wd = 0.;
    if (setting.count("max_iter")) max_iter = setting["max_iter"].i_val; 
    if (setting.count("l2")) wd = setting["l2"].f_val; 
    
    iter = 0;
  }

  struct square_root {
    MSHADOW_XINLINE static real_t Map(real_t a) {
        return sqrt(a);
    }
  };

  virtual void Update(mshadow::Tensor<xpu, dim> data, 
                      mshadow::Tensor<xpu, dim> diff) {

    if (iter == 0) {
      sumGradSquare.Resize(data.shape_, 0.);
    }
    ++iter;
    if (max_iter > 0) {
      iter %= max_iter;
    } 

                        
    sumGradSquare += diff * diff;
    data -= lr * (diff / (mshadow::expr::F<square_root>(sumGradSquare) + eps));
    if (wd > 0.) {
      data -= (wd*lr) * data;
    }
  }
  
  virtual void UpdateSparse(mshadow::Tensor<xpu, dim> data, 
                            mshadow::Tensor<xpu, dim> diff, 
                            mshadow::Tensor<xpu, 1> idx) {

    if (iter == 0) {
      sumGradSquare.Resize(data.shape_, 0.0f);
    }
    ++iter;
    if (max_iter > 0) {
      iter %= max_iter;
    } 

    int w_idx = -1;
    for (int i = 0; i < idx.size(0); ++i) {
      w_idx = idx[i];
      sumGradSquare[w_idx] += diff[i] * diff[i];

      data.Slice(w_idx, w_idx+1) -= (lr * diff.Slice(i, i+1)) /  \
          (mshadow::expr::F<square_root>(sumGradSquare.Slice(w_idx, w_idx+1)) + eps);
      if (wd > 0.) {
        data.Slice(w_idx, w_idx+1) -= (wd*lr) * data.Slice(w_idx, w_idx+1);
      }
    }
  }
 protected: 
  int iter, max_iter;
  mshadow::TensorContainer<xpu, dim> sumGradSquare;
  float eps, lr, wd;

};
}  // namespace updater
}  // namespace textnet
#endif  // TEXTNET_CONSTANT_INIT_INL_HPP_

