#include <iostream>
#include <string>
#include <map>
#include "settingv.h"

namespace textnet {
  std::map<std::string, int> SettingV::SettingIntMap = std::map<std::string, int>();
  std::map<std::string, bool> SettingV::SettingBoolMap = std::map<std::string, bool>();
    
  // Access data interface
  bool SettingV::bVal() {
    switch(value_type) {
      case SET_NONE:
        utils::Error("\tCan not convert [none] to [bool].\n");
        return b_val;
      case SET_INT:
        utils::Error("\tCan not convert [int] to [bool].\n");
        return b_val;
      case SET_FLOAT:
        utils::Error("\tCan not convert [float] to [bool].\n");
        return b_val;
      case SET_BOOL:
        return b_val;
      case SET_STRING:
        utils::Printf("\tConvert [string] to [bool].\n");
        return SettingBoolMap[s_val];
      case SET_MAP:
        utils::Error("\tCan not convert [map] to [bool].\n");
        return b_val;
    }
    return b_val;
  }
  int SettingV::iVal() {
    switch(value_type) {
      case SET_NONE:
        utils::Error("\tCan not convert [none] to [int].\n");
        return i_val;
      case SET_INT:
        return i_val;
      case SET_FLOAT:
        utils::Printf("\tConvert [float] to [int].\n");
        return static_cast<int>(f_val);
      case SET_BOOL:
        utils::Printf("\tConvert [bool] to [int].\n");
        return static_cast<int>(b_val);
      case SET_STRING:
        utils::Printf("\tConvert [string] to [int].\n");
        return SettingIntMap[s_val];
      case SET_MAP:
        utils::Error("\tCan not convert [map] to [int].\n");
        return i_val;
    }
    return i_val;
  }
  float SettingV::fVal() {
    switch(value_type) {
      case SET_NONE:
        utils::Error("\tCan not convert [none] to [float].\n");
        return f_val;
      case SET_INT:
        utils::Printf("\tConvert [int] to [float].\n");
        return static_cast<float>(i_val);
      case SET_FLOAT:
        return f_val;
      case SET_BOOL:
        utils::Error("\tCan not convert [bool] to [float].\n");
        return f_val;
      case SET_STRING:
        utils::Error("\tCan not convert [string] to [float].\n");
        break;
      case SET_MAP:
        utils::Error("\tCan not convert [map] to [float].\n");
        return f_val;
    }
    return f_val;
  }
  std::string SettingV::sVal() {
    switch(value_type) {
      case SET_NONE:
        utils::Error("\tCan not convert [none] to [string].\n");
        return s_val;
      case SET_INT:
        utils::Error("\tCan not convert [int] to [string].\n");
        return s_val;
      case SET_FLOAT:
        utils::Error("\tCan not convert [float] to [string].\n");
        return s_val;
      case SET_BOOL:
        utils::Error("\tCan not convert [bool] to [string].\n");
        return s_val;
      case SET_STRING:
        return s_val;
      case SET_MAP:
        utils::Error("\tCan not convert [map] to [string].\n");
        return s_val;
    }
    return s_val;
  }
  std::map<std::string, SettingV>* SettingV::mVal() {
    switch(value_type) {
      case SET_NONE:
        utils::Error("\tCan not convert [none] to [map].\n");
        return m_val;
      case SET_INT:
        utils::Error("\tCan not convert [int] to [map].\n");
        return m_val;
      case SET_FLOAT:
        utils::Error("\tCan not convert [float] to [map].\n");
        return m_val;
      case SET_BOOL:
        utils::Error("\tCan not convert [bool] to [map].\n");
        return m_val;
      case SET_STRING:
        utils::Error("\tCan not convert [string] to [map].\n");
        return m_val;
      case SET_MAP:
        return m_val;
    }
    return m_val;
  }

}
