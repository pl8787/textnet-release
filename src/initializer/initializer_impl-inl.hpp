#ifndef TEXTNET_INITIALIZER_IMPL_INL_HPP_
#define TEXTNET_INITIALIZER_IMPL_INL_HPP_

#include "./initializer.h"
#include "./constant_init-inl.hpp"
#include "./uniform_init-inl.hpp"
#include "./gaussian_init-inl.hpp"

namespace textnet {
namespace initializer {
template<typename xpu, int dim>
Initializer<xpu, dim>* CreateInitializer_(InitType type, std::map<std::string, SettingV> &setting) {
  switch(type) {
    case kZero: return new ConstantInitializer<xpu, dim>(setting);
    case kConstant: return new ConstantInitializer<xpu, dim>(setting);
    case kUniform: return new UniformInitializer<xpu, dim>(setting);
    case kGaussian: return new GaussianInitializer<xpu, dim>(setting);
    case kXavier: return new UniformInitializer<xpu, dim>(setting);
    case kKaiming: return new GaussianInitializer<xpu, dim>(setting);
    default: utils::Error("unknown layer type id : \"%d\"", type); return NULL;
  }
}

}  // namespace layer
}  // namespace textnet
#endif
