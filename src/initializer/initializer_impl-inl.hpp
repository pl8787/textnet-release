#ifndef TEXTNET_INITIALIZER_IMPL_INL_HPP_
#define TEXTNET_INITIALIZER_IMPL_INL_HPP_

#include "./initializer.h"
#include "./constant_init-inl.hpp"
#include "./uniform_init-inl.hpp"
#include "./uniform_range_init-inl.hpp"
#include "./gaussian_init-inl.hpp"

namespace textnet {
namespace initializer {
template<typename xpu, int dim>
Initializer<xpu, dim>* CreateInitializer_(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<xpu>* prnd) {
  switch(type) {
    case kZero: return new ConstantInitializer<xpu, dim>(setting, prnd);
    case kConstant: return new ConstantInitializer<xpu, dim>(setting, prnd);
    case kUniform: return new UniformInitializer<xpu, dim>(setting, prnd);
    case kUniformRange: return new UniformRangeInitializer<xpu, dim>(setting, prnd);
    case kGaussian: return new GaussianInitializer<xpu, dim>(setting, prnd);
    case kXavier: return new UniformInitializer<xpu, dim>(setting, prnd);
    case kKaiming: return new GaussianInitializer<xpu, dim>(setting, prnd);
    default: utils::Error("unknown initializer type id : \"%d\"", type); return NULL;
  }
}

}  // namespace layer
}  // namespace textnet
#endif
