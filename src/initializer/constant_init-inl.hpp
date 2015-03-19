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
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    this->init_type = setting["init_type"].i_val;
    if (this->init_type == kZero) {
      this->value = 0.0;
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

