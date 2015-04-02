#ifndef TEXTNET_SETTINGV_H_
#define TEXTNET_SETTINGV_H_

#include <iostream>
#include <map>
#include <string>
#include "./utils.h"

namespace textnet {

typedef int SetValueType;

const int SET_NONE = 0;
const int SET_INT = 1;
const int SET_FLOAT = 2;
const int SET_BOOL = 3;
const int SET_STRING = 4;
const int SET_MAP = 5;

/*! \brief Setting structure used to store net settings */
struct SettingV {
  bool b_val;
  int i_val;
  float f_val;
  std::string s_val;
  std::map<std::string, SettingV>* m_val;
  
  SetValueType value_type;

  // String to enum dictionary
  static std::map<std::string, int> SettingIntMap;
  // String to bool dictiionary
  static std::map<std::string, bool> SettingBoolMap;

  // Constructor
  SettingV() { value_type = SET_NONE; }
  SettingV(int i) { i_val = i; value_type = SET_INT; }
  SettingV(float f) { f_val = f; value_type = SET_FLOAT; }
  SettingV(bool b) { b_val = b; value_type = SET_BOOL; }
  SettingV(std::string s) { s_val = s; value_type = SET_STRING; }
  SettingV(const char* c) { s_val = c; value_type = SET_STRING; }
  SettingV(std::map<std::string, SettingV>* m) { m_val = m; value_type = SET_MAP; }

  // Access data interface
  bool bVal();
  int iVal();
  float fVal();
  std::string sVal();
  std::map<std::string, SettingV>* mVal();
}; //SettingV
}
#endif //TEXTNET_SETTINGV_H_
