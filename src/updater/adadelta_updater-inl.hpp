#ifndef TEXTNET_ADADELTA_UPDATER_INL_HPP_
#define TEXTNET_ADADELTA_UPDATER_INL_HPP_

#include <iostream>
#include <mshadow/tensor.h>
#include "./updater.h"

namespace textnet {
namespace updater {

template<int dim>
void norm2ByRow(mshadow::Tensor<cpu, dim> t, mshadow::Tensor<cpu, 1> &norm2row) {
    utils::Check(norm2row.size(0) == t.size(0), "norm2ByRow: size error.");
    utils::Check(t.size(1) >  1, "norm2ByRow: size error.");
    utils::Check(t.size(2) == 1, "norm2ByRow: size error.");
    utils::Check(t.size(3) == 1, "norm2ByRow: size error.");
    
    int num_row = t.size(0);
    int num_col = t.size(1);
    norm2row = 0.f;
    for (int i = 0; i < num_row; ++i) {
        for (int j = 0; j < num_col; ++j) {
            norm2row[i] += t.dptr_[i * num_col + j] * t.dptr_[i * num_col + j];
        }
        norm2row[i] = sqrt(norm2row[i]);
    }
    return;
}

template<typename xpu, int dim>
class AdaDeltaUpdater : public Updater<xpu, dim>{
 public:
  AdaDeltaUpdater(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupUpdater(setting);
  }
  virtual ~AdaDeltaUpdater(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    this->defaults["eps"] = SettingV(0.000001f);
    this->defaults["rho"] = SettingV(0.95f);
    this->defaults["l2"] = SettingV(0.0f);
    this->defaults["norm2"] = SettingV(0.0f);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["batch_size"] = SettingV();
    
    Updater<xpu, dim>::Require(setting);
  }
  
  virtual void SetupUpdater(std::map<std::string, SettingV> &setting) {
    Updater<xpu, dim>::SetupUpdater(setting);
	
    this->updater_type = setting["updater_type"].iVal();
    batch_size = setting["batch_size"].iVal(); 
    eps = setting["eps"].fVal();
    rho = setting["rho"].fVal();
    wd = setting["l2"].fVal(); 
    norm2 = setting["norm2"].fVal(); 
    iter = 0;
  }

  struct square_root {
    MSHADOW_XINLINE static real_t Map(real_t a) {
        return sqrt(a);
    }
  };

  virtual void Update(mshadow::Tensor<xpu, dim> data, 
                      mshadow::Tensor<xpu, dim> diff) {
    using namespace mshadow::expr;

    if (wd > 0.f) {
      diff += wd * data;
    }

    if (iter++ == 0) {
        sumGradSquare.Resize(data.shape_, 0.);
        sumDeltaSquare.Resize(data.shape_, 0.);
        delta.Resize(data.shape_, 0.);
    }
    sumGradSquare = rho * sumGradSquare + (1-rho) * (diff * diff);
    delta = diff * (1.f/(F<square_root>(sumGradSquare + eps))) * (F<square_root>(sumDeltaSquare + eps));
    sumDeltaSquare = rho * sumDeltaSquare + (1-rho) * (delta * delta);

    data -= delta;
    // if (wd > 0.f) {
    //   data -= (wd) * data;
    // }
    if (norm2 > 0.f) {
      float sqrt_norm2 = sqrt(norm2);
      mshadow::Shape<1> shape = mshadow::Shape1(data.size(0));
      mshadow::TensorContainer<cpu, 1> norm2row(shape);
      norm2ByRow(data, norm2row);      
      for (int i = 0; i < norm2row.size(0); ++i) {
          float n2 = norm2row[i];
          if (n2 > sqrt_norm2) {
            float scale = sqrt_norm2/n2;
            for (int j = 0; j < data.size(1); ++j) {
                data.dptr_[i*data.size(1)+j] *= scale;
            }
          }
      }
    }
  }
  
  virtual void UpdateSparse(mshadow::Tensor<xpu, dim> data, 
                            mshadow::Tensor<xpu, dim> diff, 
                            mshadow::Tensor<xpu, 1> idx) {
    using namespace mshadow::expr;


    if (iter++ == 0) {
        sumGradSquare.Resize(data.shape_, 0.);
        sumDeltaSquare.Resize(data.shape_, 0.);
        delta.Resize(data.shape_, 0.);
    }

    int w_idx = -1;
    for (int i = 0; i < idx.size(0); ++i) {
      w_idx = idx[i];
      utils::Assert(w_idx >= 0 && w_idx < data.size(0), "");
      
      mshadow::Tensor<xpu, dim> dataRow = data.Slice(w_idx, w_idx+1);
      mshadow::Tensor<xpu, dim> diffRow = diff.Slice(i, i+1);
      mshadow::Tensor<xpu, dim> deltaRow = delta.Slice(w_idx, w_idx+1);
      mshadow::Tensor<xpu, dim> sumGradSquareRow = sumGradSquare.Slice(w_idx, w_idx+1);
      mshadow::Tensor<xpu, dim> sumDeltaSquareRow = sumDeltaSquare.Slice(w_idx, w_idx+1);

      if (wd > 0.f) {
        diffRow += wd * dataRow;
      }
      sumGradSquareRow = rho * sumGradSquareRow + (1-rho) * (diffRow * diffRow);
      deltaRow = diffRow * (1.f/(F<square_root>(sumGradSquareRow + eps))) * (F<square_root>(sumDeltaSquareRow + eps));
      sumDeltaSquareRow = rho * sumDeltaSquareRow + (1-rho) * (deltaRow * deltaRow);

      dataRow -= deltaRow;
      // if (wd > 0.) {
      //   dataRow -= wd * dataRow;
      // }
    }
  }
 protected: 
  int iter, batch_size;
  mshadow::TensorContainer<xpu, dim> sumGradSquare, sumDeltaSquare, delta;
  float eps, rho, wd, norm2;

};
}  // namespace updater
}  // namespace textnet
#endif 

