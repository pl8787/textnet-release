#ifndef TEXTNET_CONSTANT_INIT_INL_HPP_
#define TEXTNET_CONSTANT_INIT_INL_HPP_

#include <mshadow/tensor.h>
#include "./initializer.h"

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class ConstantInitializer : public Initializer<xpu, dim>{
 public:
  ConstantInitializer(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupInitializer(setting);
  }
  virtual ~ConstantInitializer(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    this->defaults["value"] = SettingV(0.0f);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    Initializer<xpu, dim>::Require(setting);
  }
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    Initializer<xpu, dim>::SetupInitializer(setting);
    
    this->init_type = setting["init_type"].i_val;
    if (this->init_type == kZero) {
      this->value = 0.0f;
    } else if (this->init_type == kConstant) {
      this->value = setting["value"].f_val;
    }
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {
    data = this->value;
  }
  
  float value;
};
}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_CONSTANT_INIT_INL_HPP_

