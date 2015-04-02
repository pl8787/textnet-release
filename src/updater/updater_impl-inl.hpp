#ifndef TEXTNET_UPDATER_IMPL_INL_HPP_
#define TEXTNET_UPDATER_IMPL_INL_HPP_

#include "./updater.h"
#include "./sgd_updater-inl.hpp"
// #include "./sgdsparse_updater-inl.hpp"
#include "./adagrad_updater-inl.hpp"
// #include "./adam_updater-inl.hpp"

namespace textnet {
namespace updater {
template<typename xpu, int dim>
Updater<xpu, dim>* CreateUpdater_(UpdaterType type, std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
  switch(type) {
    case kSGD: return new SGDUpdater<xpu, dim>(setting, prnd);
    case kAdagrad: return new AdagradUpdater<xpu, dim>(setting, prnd);
    // case kAdam: return new AdamInitializer<xpu, dim>(setting);
    // case kSGDSparse: return new SGDSparseInitializer<xpu, dim>(setting);
    default: utils::Error("unknown updater type id : \"%d\"", type); return NULL;
  }
}

}  // namespace updater
}  // namespace textnet
#endif
