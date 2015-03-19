#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "initializer_impl-inl.hpp"
// specialize the cpu implementation here
namespace textnet {
namespace initializer {
template<>
Initializer<cpu, 1>* CreateInitializer<cpu, 1>(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<cpu>* prnd) {
  return CreateInitializer_<cpu, 1>(type, setting, prnd); 
}
template<>
Initializer<cpu, 2>* CreateInitializer<cpu, 2>(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<cpu>* prnd) {
  return CreateInitializer_<cpu, 2>(type, setting, prnd); 
}
template<>
Initializer<cpu, 3>* CreateInitializer<cpu, 3>(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<cpu>* prnd) {
  return CreateInitializer_<cpu, 3>(type, setting, prnd); 
}
template<>
Initializer<cpu, 4>* CreateInitializer<cpu, 4>(
                             InitType type,
                             std::map<std::string, SettingV> &setting,
                             mshadow::Random<cpu>* prnd) {
  return CreateInitializer_<cpu, 4>(type, setting, prnd); 
}
}  // namespace initializer
}  // namespace textnet
