#ifndef TEXTNET_INITIALIZER_INITIALIZER_H_
#define TEXTNET_INITIALIZER_INITIALIZER_H_

#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"
#include "../utils/utils.h"
#include "../utils/io.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of layer defintiion */
namespace initializer {

/*! \brief use integer to encode layer types */
typedef int InitType;

template<typename xpu, int dim>
class Initializer {
 public:
  Initializer(void) {}
  virtual ~Initializer(void) {}
  
  virtual void SetupInitializer() {
    init_type = 0;
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {}
  
  virtual InitType GetInitType() { return init_type; }
  
 protected:
  InitType init_type;
  mshadow::Random<xpu>* prnd_;
  
};

/*! \brief these are enumeration */
const int kZero = 0;
const int kConstant = 1;
const int kUniform = 2;
const int kGaussian = 3;
const int kXavier = 4;
const int kKaiming = 5;

template<typename xpu, int dim>
Initializer<xpu, dim>* CreateInitializer(InitType type, 
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<xpu>* prnd);

}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_INITIALIZER_INITIALIZER_H_
