#ifndef TEXTNET_GAUSSIAN_INIT_INL_HPP_
#define TEXTNET_GAUSSIAN_INIT_INL_HPP_

#include <mshadow/tensor.h>
#include "./initializer.h"

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class GaussianInitializer : public Initializer<xpu, dim>{
 public:
  GaussianInitializer(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupInitializer(setting);
  }
  virtual ~GaussianInitializer(void) {}
  
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
    if (this->init_type == kGaussian) {
      this->mu = setting["mu"].fVal();
      this->sigma = setting["sigma"].fVal();
    } else if (this->init_type == kKaiming) {
      this->mu = 0.0;
	  this->sigma = 1.0;
    }
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {
	if (this->init_type == kKaiming) {
		float node_size = 1.0;
		for (int i = 1; i < dim; ++i)
			node_size *= data.shape_[i];
		this->sigma = sqrt(2.0 / node_size);
		cout << "MSRA INITIAL: " << this->sigma << endl;
	}
    this->prnd_->SampleGaussian(&data, mu, sigma);
  }
  
  float mu;
  float sigma;
};
}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_GAUSSIAN_INIT_INL_HPP_

