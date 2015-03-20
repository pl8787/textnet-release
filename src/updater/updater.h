#ifndef TEXTNET_UPDATER_UPDATER_H_
#define TEXTNET_UPDATER_UPDATER_H_

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
namespace updater {

/*! \brief use integer to encode layer types */
typedef int UpdaterType;

template<typename xpu, int dim>
class Updater {
 public:
  Updater(void) {}
  virtual ~Updater(void) {}
  
  virtual void SetupUpdater() {
    updater_type = 0;
    is_sparse = false;
  }
  
  virtual void Update(mshadow::Tensor<xpu, dim> data, 
                      mshadow::Tensor<xpu, dim> diff) = 0;
  
  virtual void UpdateSparse(mshadow::Tensor<xpu, dim> data, 
                            mshadow::Tensor<xpu, dim> diff, 
                            mshadow::Tensor<xpu, 1> idx) = 0;
                            
  
  virtual UpdaterType GetUpdaterType() { return updater_type; }
  
  // whether use sparse update
  bool is_sparse;
  
 protected:
  UpdaterType updater_type;
  mshadow::Random<xpu>* prnd_;
  
};

/*! \brief these are enumeration */
const int kSGD = 0;
const int kAdagrad = 1;
const int kAdam = 2;
const int kSGDSparse = 3;

template<typename xpu, int dim>
Updater<xpu, dim>* CreateUpdater(UpdaterType type, std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd);

}  // namespace updater
}  // namespace textnet
#endif  // TEXTNET_UPDATER_UPDATER_H_
