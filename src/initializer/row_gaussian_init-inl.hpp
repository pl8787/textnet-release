#ifndef TEXTNET_ROW_GAUSSIAN_INIT_INL_HPP_
#define TEXTNET_ROW_GAUSSIAN_INIT_INL_HPP_

#include <mshadow/tensor.h>
#include "./initializer.h"

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class RowGaussianInitializer : public Initializer<xpu, dim>{
 public:
  RowGaussianInitializer(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupInitializer(setting);
  }
  virtual ~RowGaussianInitializer(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    this->defaults["mu"] = SettingV(0.0f);
    this->defaults["sigma"] = SettingV(1.0f);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    Initializer<xpu, dim>::Require(setting);
  }
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    Initializer<xpu, dim>::SetupInitializer(setting);
    
    this->init_type = setting["init_type"].iVal();
    this->mu = setting["mu"].fVal();
    this->sigma = setting["sigma"].fVal();
    this->count = setting["count"].iVal();
  }
  
  virtual void DoInitialize(mshadow::Tensor<cpu, dim> data) {
    using namespace mshadow::expr;
    using namespace mshadow;
    TensorContainer<xpu, 1> row;
    row.Resize(mshadow::Shape1(count));
    this->prnd_->SampleGaussian(&row, mu, sigma);
    for (int i = 0; i < data.shape_.Size(); i += count) {
        for (int j = 0; j < count; ++j) {
            data.dptr_[i+j] = row[j];
        }
    }
  }
  
  float mu;
  float sigma;
  int count;
};
}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_ROW_GAUSSIAN_INIT_INL_HPP_

