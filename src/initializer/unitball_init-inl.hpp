#ifndef TEXTNET_UNITBALL_INIT_INL_HPP_
#define TEXTNET_UNITBALL_INIT_INL_HPP_

#include <mshadow/tensor.h>
#include "./initializer.h"

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class UnitballInitializer : public Initializer<xpu, dim>{
 public:
  UnitballInitializer(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupInitializer(setting);
  }
  virtual ~UnitballInitializer(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    this->defaults["mu"] = SettingV(0.0f);
    this->defaults["sigma"] = SettingV(1.0f);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
	this->defaults["vec_len"] = SettingV();
    
    Initializer<xpu, dim>::Require(setting);
  }
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    Initializer<xpu, dim>::SetupInitializer(setting);
    
    this->init_type = setting["init_type"].iVal();
    this->mu = setting["mu"].fVal();
    this->sigma = setting["sigma"].fVal();
	this->vec_len = setting["vec_len"].iVal();
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {
    this->prnd_->SampleGaussian(&data, mu, sigma);
    for (int i = 0; i < data.shape_.Size(); i += vec_len) {
	  float norm = 0.0f;
      for (int j = 0; j < vec_len; ++j) {
        norm += data.dptr_[i+j] * data.dptr_[i+j];
	  }
	  norm = sqrt(norm);
	  for (int j = 0; j < vec_len; ++j) {
        data.dptr_[i+j] /= norm;
	  }
	}
  }
  
  float mu;
  float sigma;
  int vec_len;
};
}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_UNITBALL_INIT_INL_HPP_

