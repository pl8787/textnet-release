#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "updater_impl-inl.hpp"
// specialize the cpu implementation here
namespace textnet {
namespace updater {
template<>
Updater<cpu, 1>* CreateUpdater<cpu, 1>(UpdaterType type, std::map<std::string, SettingV> &setting) {
  return CreateUpdater_<cpu, 1>(type, setting); 
}
template<>
Updater<cpu, 2>* CreateUpdater<cpu, 2>(UpdaterType type, std::map<std::string, SettingV> &setting) {
  return CreateUpdater_<cpu, 2>(type, setting); 
}
template<>
Updater<cpu, 3>* CreateUpdater<cpu, 3>(UpdaterType type, std::map<std::string, SettingV> &setting) {
  return CreateUpdater_<cpu, 3>(type, setting); 
}
template<>
Updater<cpu, 4>* CreateUpdater<cpu, 4>(UpdaterType type, std::map<std::string, SettingV> &setting) {
  return CreateUpdater_<cpu, 4>(type, setting); 
}
}  // namespace updater
}  // namespace textnet
