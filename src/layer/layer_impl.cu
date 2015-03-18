#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are
#include "layer_impl-inl.hpp"
// specialize the gpu implementation here
namespace textnet {
namespace layer {
template<>
Layer<gpu>* CreateLayer<gpu>(LayerType type) {
  return CreateLayer_<gpu>(type); 
}
}  // namespace layer
}  // namespace textnet

