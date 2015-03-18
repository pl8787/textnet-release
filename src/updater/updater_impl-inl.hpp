#ifndef TEXTNET_UPDATER_IMPL_INL_HPP_
#define TEXTNET_UPDATER_IMPL_INL_HPP_

#include "./updater.h"
// #include "./sgd_updater-inl.hpp"
// #include "./sgdsparse_updater-inl.hpp"
// #include "./adagrad_updater-inl.hpp"
// #include "./adam_updater-inl.hpp"

namespace textnet {
namespace updater {
template<typename xpu, int dim>
Updater<xpu, dim>* CreateUpdater_(UpdaterType type, std::map<std::string, SettingV> &setting) {
  switch(type) {
    // case kSGD: return new SGDInitializer<xpu, dim>(setting);
    // case kAdagrad: return new AdagradInitializer<xpu, dim>(setting);
    // case kAdam: return new AdamInitializer<xpu, dim>(setting);
    // case kSGDSparse: return new SGDSparseInitializer<xpu, dim>(setting);
    default: utils::Error("unknown layer type id : \"%d\"", type); return NULL;
  }
}

}  // namespace updater
}  // namespace textnet
#endif
