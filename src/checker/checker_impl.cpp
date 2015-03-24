#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "checker_impl-inl.hpp"
// specialize the cpu implementation here
namespace textnet {
namespace checker {
template<>
Checker<cpu>* CreateChecker<cpu>() {
  return CreateChecker_<cpu>(); 
}

}  // namespace checker
}  // namespace textnet
