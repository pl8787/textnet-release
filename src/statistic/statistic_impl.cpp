#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "net_impl-inl.hpp"
// specialize the cpu implementation here
namespace textnet {
namespace net {
INet* CreateNetCPU(NetType type) {
  return CreateNet_<cpu>(type); 
}
#if MSHADOW_USE_CUDA == 0
INet* CreateNetGPU(NetType type) {
  utils::Error("Set CPU_ONLY to 1, so no gpu models.");
  return NULL;
}
#endif
}  // namespace net
}  // namespace textnet
