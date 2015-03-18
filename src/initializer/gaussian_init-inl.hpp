#ifndef TEXTNET_GAUSSIAN_INIT_INL_HPP_
#define TEXTNET_GAUSSIAN_INIT_INL_HPP_

#include <mshadow/tensor.h>
#include "./initializer.h"

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class GaussianInitializer : public Initializer<xpu, dim>{
 public:
  GaussianInitializer(std::map<std::string, SettingV> &setting, int seed = 37) : rnd(seed) {
    SetupInitializer(setting);
  }
  virtual ~GaussianInitializer(void) {}
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    this->init_type = setting["init_type"].i_val;
    if (this->init_type == kGaussian) {
      this->mu = setting["mu"].f_val;
      this->sigma = setting["sigma"].f_val;
    } else if (this->init_type == kKaiming) {
      // Todo
    }
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {
    this->rnd.SampleGaussian(&data, mu, sigma);
  }
  
  float mu;
  float sigma;
  mshadow::Random<xpu> rnd;
};
}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_GAUSSIAN_INIT_INL_HPP_

