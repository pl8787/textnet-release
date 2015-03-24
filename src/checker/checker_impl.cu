#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "checker_impl-inl.hpp"
// specialize the gpu implementation here
namespace textnet {
namespace checker {
template<>
Checker<gpu>* CreateChecker<gpu>() {
  return CreateChecker_<gpu>(); 
}

}  // namespace checker
}  // namespace textnet
