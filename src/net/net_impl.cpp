#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "net_impl-inl.hpp"
// specialize the cpu implementation here
namespace textnet {
namespace net {
template<>
Layer<cpu>* CreateLayer<cpu>(LayerType type) {
  return CreateLayer_<cpu>(type); 
}
}  // namespace net
}  // namespace textnet
