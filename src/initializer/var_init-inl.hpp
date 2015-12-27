#ifndef TEXTNET_VAR_INIT_INL_HPP_
#define TEXTNET_VAR_INIT_INL_HPP_

#include <mshadow/tensor.h>
#include "./initializer.h"

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class VarInitializer : public Initializer<xpu, dim>{
 public:
  VarInitializer(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupInitializer(setting);
  }
  virtual ~VarInitializer(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    this->defaults["value"] = SettingV(0.0f);
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    
    Initializer<xpu, dim>::Require(setting);
  }
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    Initializer<xpu, dim>::SetupInitializer(setting);
    this->init_type = setting["init_type"].iVal();
	value = setting["value"].fVal();
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {
	int init_size = data.shape_.Size();
	utils::Check(init_size % 3 == 0, "In VarInit: init_size %% 3 != 0");
	float * data_ptr = data.dptr_;
	for (int i = 0; i < init_size; i++) {
	  if (i % 3 == 0) {
	    data_ptr[i] = this->value;
	  } else if (i % 3 == 1) {
		data_ptr[i] = 0.0f;
	  } else {
		data_ptr[i] = -this->value;
	  }
	}
  }
  
  float value;
};
}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_VAR_INIT_INL_HPP_

