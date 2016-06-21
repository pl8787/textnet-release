#ifndef TEXTNET_ADAM_UPDATER_INL_HPP_
#define TEXTNET_ADAM_UPDATER_INL_HPP_

#include <iostream>
#include <mshadow/tensor.h>
#include "./updater.h"

namespace textnet {
namespace updater {

template<typename xpu, int dim>
class AdamUpdater : public Updater<xpu, dim>{
 public:
  AdamUpdater(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupUpdater(setting);
  }
  virtual ~AdamUpdater(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    this->defaults["b1"] = SettingV(0.1f);
    this->defaults["b2"] = SettingV(0.001f);
    this->defaults["bias_correct"] = SettingV(false);
    this->defaults["eps"] = SettingV(1e-8f);
    this->defaults["l2"] = SettingV(0.0f);
    this->defaults["max_iter"] = SettingV(-1);
    this->defaults["lr_decay_factor"] = SettingV(1.f);
    this->defaults["lr_decay_interval"] = SettingV(0);
    this->defaults["batch_size"] = SettingV(1);

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["lr"] = SettingV();
    
    Updater<xpu, dim>::Require(setting);
  }
  
  virtual void SetupUpdater(std::map<std::string, SettingV> &setting) {
    Updater<xpu, dim>::SetupUpdater(setting);
	
    this->updater_type = setting["updater_type"].iVal();
    eps = setting["eps"].fVal();
    b1 = setting["b1"].fVal();
    b2 = setting["b2"].fVal();
    bias_correct = setting["bias_correct"].bVal();
    lr = setting["lr"].fVal();
    max_iter = setting["max_iter"].iVal(); 
    batch_size = setting["batch_size"].iVal(); 
    wd = setting["l2"].fVal(); 
    lr_decay_interval = setting["lr_decay_interval"].iVal(); 
    lr_decay_factor   = setting["lr_decay_factor"].fVal(); 
    
    iter = 0;
  }

  struct square_root {
    MSHADOW_XINLINE static real_t Map(real_t a) {
        return sqrt(a);
    }
  };

  virtual void Update(mshadow::Tensor<xpu, dim> data, 
                      mshadow::Tensor<xpu, dim> diff) {

    if (iter == 0 || ((max_iter > 0) && (iter % max_iter == 0))) {
      b1pt = 1.0f;
      b2pt = 1.0f;
      adam_mt.Resize(data.shape_, 0.);
      adam_vt.Resize(data.shape_, 0.);
    }
    if ((iter > 0) && (lr_decay_interval > 0) && (iter % lr_decay_interval == 0)) {
      lr *= lr_decay_factor;
    }

    ++iter;
    
    if (batch_size > 1) {
        diff /= float(batch_size);
    }
    if (wd > 0.) {
        diff += wd * data;
    }
                        
    adam_mt = (1.0 - b1) * adam_mt + b1 * diff;
    adam_vt = (1.0 - b2) * adam_vt + b2 * diff * diff;
    if (bias_correct) {
        b1pt *= (1.0 - b1);
        b2pt *= (1.0 - b2);
        data -= lr * (adam_mt / (mshadow::expr::F<square_root>(adam_vt) / (1.0 - b2pt) + eps)) / (1.0 - b1pt);
    } else {
        data -= lr * (adam_mt / (mshadow::expr::F<square_root>(adam_vt) + eps));
    }
  }
  
  virtual void UpdateSparse(mshadow::Tensor<xpu, dim> data, 
                            mshadow::Tensor<xpu, dim> diff, 
                            mshadow::Tensor<xpu, 1> idx) {

    if (iter == 0 || ((max_iter > 0) && (iter % max_iter == 0))) {
      b1pt = 1.0f;
      b2pt = 1.0f;
      adam_mt.Resize(data.shape_, 0.);
      adam_vt.Resize(data.shape_, 0.);
    }
    if ((iter > 0) && (lr_decay_interval > 0) && (iter % lr_decay_interval == 0)) {
      lr *= lr_decay_factor;
    }

    ++iter;

    if (batch_size > 1) {
        diff /= float(batch_size);
    }

    int w_idx = -1;
    for (int i = 0; i < idx.size(0); ++i) {
      w_idx = idx[i];
      utils::Assert(w_idx >= 0 && w_idx < data.size(0), "Adagrad Sparse Update index error.");

      mshadow::Tensor<xpu, dim> adam_mtRow = adam_mt.Slice(w_idx, w_idx+1);
      mshadow::Tensor<xpu, dim> adam_vtRow = adam_vt.Slice(w_idx, w_idx+1);
      mshadow::Tensor<xpu, dim> diffRow = diff.Slice(i, i+1);
      mshadow::Tensor<xpu, dim> dataRow = data.Slice(w_idx, w_idx+1);

      if (wd > 0.) {
        diffRow += wd * dataRow;
      }
      adam_mtRow = (1.0 - b1) * adam_mtRow + b1 * diffRow;
      adam_vtRow = (1.0 - b2) * adam_vtRow + b2 * diffRow * diffRow;
      if (bias_correct) {
          b1pt *= (1.0 - b1);
          b2pt *= (1.0 - b2);
          dataRow -= lr * (adam_mtRow / (mshadow::expr::F<square_root>(adam_vtRow) / (1.0 - b2pt) + eps)) / (1.0 - b1pt);
      } else {
          dataRow -= lr * (adam_mtRow / (mshadow::expr::F<square_root>(adam_vtRow) + eps));
      }
    }
  }
 protected: 
  int iter, max_iter, batch_size, lr_decay_interval;
  mshadow::TensorContainer<xpu, dim> adam_mt;
  mshadow::TensorContainer<xpu, dim> adam_vt;
  float eps, lr, wd, lr_decay_factor;
  float b1, b2;
  float b1pt, b2pt;
  bool bias_correct;

};
}  // namespace updater
}  // namespace textnet
#endif  // TEXTNET_ADAM_UPDATER_INL_HPP_

