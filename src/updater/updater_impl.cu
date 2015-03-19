#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "updater_impl-inl.hpp"
// specialize the gpu implementation here
namespace textnet {
namespace updater {
template<>
Updater<gpu, 1>* CreateUpdater<gpu, 1>(UpdaterType type, std::map<std::string, SettingV> &setting, 
                      mshadow::Random<gpu>* prnd) {
  return CreateUpdater_<gpu, 1>(type, setting, prnd); 
}
template<>
Updater<gpu, 2>* CreateUpdater<gpu, 2>(UpdaterType type, std::map<std::string, SettingV> &setting, 
                      mshadow::Random<gpu>* prnd) {
  return CreateUpdater_<gpu, 2>(type, setting, prnd); 
}
template<>
Updater<gpu, 3>* CreateUpdater<gpu, 3>(UpdaterType type, std::map<std::string, SettingV> &setting, 
                      mshadow::Random<gpu>* prnd) {
  return CreateUpdater_<gpu, 3>(type, setting, prnd); 
}
template<>
Updater<gpu, 4>* CreateUpdater<gpu, 4>(UpdaterType type, std::map<std::string, SettingV> &setting, 
                      mshadow::Random<gpu>* prnd) {
  return CreateUpdater_<gpu, 4>(type, setting, prnd); 
}
}  // namespace updater
}  // namespace textnet