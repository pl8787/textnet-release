#ifndef TEXTNET_CHECKER_IMPL_INL_HPP_
#define TEXTNET_CHECKER_IMPL_INL_HPP_

#include "./checker.h"

namespace textnet {
namespace checker {

template<typename xpu>
Checker<xpu>* CreateChecker_() {
  return new Checker<xpu>();
}

}  // namespace checker
}  // namespace textnet
#endif
