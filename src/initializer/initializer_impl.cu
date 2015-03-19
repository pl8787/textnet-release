#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "initializer_impl-inl.hpp"
// specialize the cpu implementation here
namespace textnet {
namespace initializer {
template<>
Initializer<gpu, 1>* CreateInitializer<gpu, 1>(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<xpu>* prnd) {
  return CreateInitializer_<gpu, 1>(type, setting, prnd); 
}
template<>
Initializer<gpu, 2>* CreateInitializer<gpu, 2>(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<xpu>* prnd) {
  return CreateInitializer_<gpu, 2>(type, setting, prnd); 
}
template<>
Initializer<gpu, 3>* CreateInitializer<gpu, 3>(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<xpu>* prnd) {
  return CreateInitializer_<gpu, 3>(type, setting, prnd); 
}
template<>
Initializer<gpu, 4>* CreateInitializer<gpu, 4>(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<xpu>* prnd) {
  return CreateInitializer_<gpu, 4>(type, setting, prnd); 
}
}  // namespace initializer
}  // namespace textnet
