#ifndef TEXTNET_INITIALIZER_INITIALIZER_H_
#define TEXTNET_INITIALIZER_INITIALIZER_H_

#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/settingv.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of layer defintiion */
namespace initializer {

/*! \brief use integer to encode layer types */
typedef int InitType;

/*! \brief these are enumeration */
const int kZero = 0;
const int kConstant = 1;
const int kUniform = 2;
const int kGaussian = 3;
const int kXavier = 4;
const int kKaiming = 5;
const int kUnitball = 6;
const int kUniformRange = 7;
const int kFileInit = 8;

template<typename xpu, int dim>
class Initializer {
 public:
  Initializer(void) {}
  virtual ~Initializer(void) {}
  
  // To implement this function you need call base function in the end
  virtual void Require(std::map<std::string, SettingV> &setting) {
    defaults["init_type"] = SettingV(kZero);
    for (std::map<std::string, SettingV>::iterator it = defaults.begin();
          it != defaults.end(); ++it) {
      std::string name = it->first;
      if (defaults[name].value_type == SET_NONE) {
        utils::Check(setting.count(name), 
            "\tSetting [%s] needed for this layer.\n", name.c_str());
      } else {
        if (!setting.count(name)) {
          setting[name] = defaults[name];
          utils::Printf("\tSetting [%s] set to default value.\n", name.c_str());
        }
      }
    }
  }
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    init_type = 0;
    this->Require(setting);
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {}
  
  virtual InitType GetInitType() { return init_type; }
  
 protected:
  InitType init_type;
  mshadow::Random<xpu>* prnd_;
  // required setting
  std::map<std::string, SettingV> defaults;
  
};

template<typename xpu, int dim>
Initializer<xpu, dim>* CreateInitializer(InitType type, 
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<xpu>* prnd);

}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_INITIALIZER_INITIALIZER_H_

