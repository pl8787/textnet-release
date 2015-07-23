#ifndef TEXTNET_UNIFORM_RANGE_INIT_INL_HPP_
#define TEXTNET_UNIFORM_RANGE_INIT_INL_HPP_

#include <iostream>
#include <mshadow/tensor.h>
#include "./initializer.h"

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class UniformRangeInitializer : public Initializer<xpu, dim>{
 public:
  UniformRangeInitializer(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupInitializer(setting);
  }
  virtual ~UniformRangeInitializer(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["upper"] = SettingV();
    this->defaults["lower"] = SettingV();
    
    Initializer<xpu, dim>::Require(setting);
  }
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    Initializer<xpu, dim>::SetupInitializer(setting);
    
    this->init_type = setting["init_type"].iVal();
    if (this->init_type == kUniformRange) {
      this->lower = setting["lower"].fVal();
      this->upper = setting["upper"].fVal();
      utils::Check(lower < upper, "UniformRangeInitializer: range error.");
    } else if (this->init_type == kXavier) {
      // Todo
    }
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {
    this->prnd_->SampleUniform(&data, lower, upper);
  }
  
  float lower, upper;
};
}  // namespace initializer
}  // namespace textnet
#endif 

