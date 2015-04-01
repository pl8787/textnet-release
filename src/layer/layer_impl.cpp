#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "layer_impl-inl.hpp"
// specialize the cpu implementation here
namespace textnet {
namespace layer {
template<>
Layer<cpu>* CreateLayer<cpu>(LayerType type) {
  return CreateLayer_<cpu>(type); 
}
}  // namespace layer
}
