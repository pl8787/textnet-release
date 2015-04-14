#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "net_impl-inl.hpp"
// specialize the gpu implementation here
namespace textnet {
namespace net {
INet* CreateNetGPU(NetType type) {
  return CreateNet_<gpu>(type); 
}
}  // namespace net
}  // namespace textnet

