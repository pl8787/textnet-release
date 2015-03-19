#ifndef TEXTNET_UNIFORM_INIT_INL_HPP_
#define TEXTNET_UNIFORM_INIT_INL_HPP_

#include <mshadow/tensor.h>
#include "./initializer.h"

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class UniformInitializer : public Initializer<xpu, dim>{
 public:
  UniformInitializer(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupInitializer(setting);
  }
  virtual ~UniformInitializer(void) {}
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    this->init_type = setting["init_type"].i_val;
    if (this->init_type == kUniform) {
      this->range = setting["range"].f_val;
    } else if (this->init_type == kXavier) {
      // Todo
    }
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {
    this->prnd_->SampleUniform(&data, -range, range);
  }
  
  float range;
};
}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_UNIFORM_INIT_INL_HPP_

