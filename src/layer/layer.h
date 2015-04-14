#ifndef TEXTNET_LAYER_LAYER_H_
#define TEXTNET_LAYER_LAYER_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"
#include "node.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/settingv.h"
#include "../io/json/json.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of layer defintiion */
namespace layer {

/*! \brief use integer to encode layer types */
typedef int LayerType;
typedef int PhraseType;

/*! \brief these are enumeration */
// shared layer is a special type indicating that this connection
// is sharing Layer with an existing connection
const int kUnkonwnLayer = 0;
// Activation Layer 1-10
const int kRectifiedLinear = 1;
const int kSigmoid = 2;
const int kTanh = 3;

// Common Layer 11-50
const int kFullConnect = 11;
const int kFlatten = 12;
const int kDropout = 13;
const int kConv = 14;
const int kMaxPooling = 15;
const int kSumPooling = 16;
const int kAvgPooling = 17;
const int kConcat = 18;
const int kChConcat = 19;
const int kSplit = 20;
const int kEmbedding = 21;
const int kCross = 22;
const int kMatch = 23;
const int kLstm = 24;
const int kWholePooling = 25;
const int kConvolutionalLstm = 26;
const int kRecurrent = 27;
const int kSequenceDimReduction = 28;
const int kConvLstmSplit = 29;
const int kTensorFullConnect = 30;
const int kMaxRecurrent = 31;

// Loss Layer 51-70
const int kSoftmax = 51;
const int kL2Loss = 52;
const int kMultiLogistic = 53;
const int kHingeLoss = 54;
const int kPairHingeLoss = 55;
const int kAccuracy = 56;

// Input Layer 71-
const int kTextData = 71;
const int kSequenceClassificationData = 72;
const int kNextBasketData = 73;


/*! \brief these are enumeration */
const int kTrain = 0;
const int kTest = 1;

template<typename xpu>
class Layer {
 public:
  Layer(void) {
    layer_type = 0;
    phrase_type = -1;
  }
  virtual ~Layer(void) {}
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting, 
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    this->settings = setting;
    this->Require();
    
    phrase_type = this->settings["phrase_type"].iVal();
    prnd_ = prnd;

	setting = this->settings;
  }
  
  virtual void SetupLayer(Json::Value &root,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    LoadModel(root);
    this->SetupLayer(this->settings, bottom, top, prnd);                      
  }
  
  // To implement this function you need call base function in the end
  virtual void Require() {
    defaults["phrase_type"] = SettingV(kTrain);
    for (std::map<std::string, SettingV>::iterator it = defaults.begin();
          it != defaults.end(); ++it) {
      std::string name = it->first;
      if (defaults[name].value_type == SET_NONE) {
        utils::Check(settings.count(name), 
            "\tSetting [%s] needed for this layer.\n", name.c_str());
      } else {
        if (!settings.count(name)) {
          settings[name] = defaults[name];
          utils::Printf("\tSetting [%s] set to default value.\n", name.c_str());
        }
      }
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {}

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) = 0;
  
  // clear all top nodes diff and paramters diff
  virtual void ClearDiff(const std::vector<Node<xpu>*> &bottom,
                         const std::vector<Node<xpu>*> &top) {
    for (int i = 0; i < params.size(); ++i) {
        params[i].ClearDiff();
    }
    for (int i = 0; i < top.size(); ++i) {
        top[i]->ClearDiff();
    }
  }

  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) = 0;
                        
  virtual int BottomNodeNum() = 0;
  virtual int TopNodeNum() = 0;

  virtual int ParamNodeNum() = 0;

  // parameter sharing
  virtual void ShareParameter(int param_idx, Node<xpu> &other) {
	  utils::Check(param_idx < params.size(), 
			  "Share param index extend params size.");
      params[param_idx].Share(other);
  }


  void SaveSetting(std::map<std::string, SettingV> &setting, Json::Value &root) {
    for (std::map<std::string, SettingV>::iterator it = setting.begin(); 
         it != setting.end(); ++it) {
      switch ( it->second.value_type ) {
        case SET_INT:
          {
            root[it->first] = it->second.iVal();
          }
          break;
        case SET_FLOAT:
          {
            root[it->first] = it->second.fVal();
          }
          break;
        case SET_BOOL:
          {
            root[it->first] = it->second.bVal();
          }
          break;
        case SET_STRING:
          {
            root[it->first] = it->second.sVal();
          }
          break;
        case SET_MAP:
          {
            Json::Value sub_root;
            SaveSetting(*(it->second.m_val), sub_root);
            root[it->first] = sub_root;
          }
          break;
        case SET_NONE:
          break;
      }
    }
  }
  
  void LoadSetting(std::map<std::string, SettingV> &setting, Json::Value &root) {
    Json::Value::Members member = root.getMemberNames();
    for (Json::Value::Members::iterator it = member.begin();
         it != member.end(); ++it) {
      std::string name = *it;
      Json::Value value = root[name];
      switch (value.type()) {
        case Json::intValue:
          {
            setting[name] = SettingV(value.asInt());
          }
          break;
        case Json::realValue:
          {
            setting[name] = SettingV(value.asFloat());
          }
          break;
        case Json::booleanValue:
          {
            setting[name] = SettingV(value.asBool());
          }
          break;
        case Json::stringValue:
          {
            setting[name] = SettingV(value.asString());
          }
          break;
        case Json::objectValue:
          {
            std::map<std::string, SettingV> *sub_setting = new std::map<std::string, SettingV>();
            LoadSetting(*sub_setting, value);
            setting[name] = SettingV(sub_setting);
          }
          break;
        case Json::arrayValue:
        case Json::nullValue:
          break;
        default:
          break;
      }
    }
  }

  virtual void SaveModel(Json::Value &layer_root, bool need_param = true) {
    // Set layer type
    layer_root["layer_type"] = layer_type;
    layer_root["layer_name"] = layer_name;
    layer_root["layer_idx"] = layer_idx;
    
    // Set bottom / top Nodes
    Json::Value bottoms_root;
    Json::Value tops_root;
    for (int i = 0; i < bottom_nodes.size(); ++i) {
      bottoms_root.append(bottom_nodes[i]);
    }
    for (int i = 0; i < top_nodes.size(); ++i) {
      tops_root.append(top_nodes[i]);
    }
    layer_root["bottom_nodes"] = bottoms_root;
    layer_root["top_nodes"] = tops_root;
    
    // Set layer settings
    Json::Value setting_root;
    SaveSetting(settings, setting_root);
    layer_root["setting"] = setting_root;
    
    if (!need_param) return;
    
    // Set layer weights
    Json::Value params_root;
    for (int i = 0; i < params.size(); ++i) {
      Json::Value param_root;
      Json::Value param_value_root;
      Json::Value param_shape_root;
      
      param_shape_root.append(params[i].data.size(0));
      param_shape_root.append(params[i].data.size(1));
      param_shape_root.append(params[i].data.size(2));
      param_shape_root.append(params[i].data.size(3));
      
      for (int j = 0; j < params[i].data.shape_.Size(); ++j) {
        param_value_root.append(params[i].data.dptr_[j]);
      }
      
      param_root["shape"] = param_shape_root;
      param_root["value"] = param_value_root;
      
      params_root.append(param_root);
    }
    layer_root["param"] = params_root;
  }

  virtual void LoadModel(Json::Value &layer_root) {
    // Set Layer type
	if (layer_root["layer_type"].isInt()) {
		layer_type = layer_root["layer_type"].asInt();
	} else if (layer_root["layer_type"].isString()) {
        layer_type = SettingV::SettingIntMap[layer_root["layer_type"].asString()];
	} else {
		utils::Error("[Error] layer type error.\n");
	}
    layer_name = layer_root["layer_name"].asString();
    layer_idx = layer_root["layer_idx"].asInt();
    
    // Set bottom / top nodes
    Json::Value bottoms_root = layer_root["bottom_nodes"];
    Json::Value tops_root = layer_root["top_nodes"];
    for (int i = 0; i < bottoms_root.size(); ++i) {
      bottom_nodes.push_back(bottoms_root[i].asString());
    }
    for (int i = 0; i < tops_root.size(); ++i) {
      top_nodes.push_back(tops_root[i].asString());
    }
    
    // Set layer settings
    Json::Value setting_root = layer_root["setting"];
    LoadSetting(this->settings, setting_root);
    
    // Set layer weights
    if (!layer_root["param"]) 
      return;
    Json::Value params_root = layer_root["param"];
    params.resize(params_root.size());
    
    for (int i = 0; i < params_root.size(); ++i) {
      Json::Value param_root = params_root[i];
      Json::Value param_value_root = param_root["value"];
      Json::Value param_shape_root = param_root["shape"];
      
      params[i].Resize(param_shape_root[0].asInt(), param_shape_root[1].asInt(), 
                       param_shape_root[2].asInt(), param_shape_root[3].asInt(),
                       true);
                        
      for (int j = 0; j < params[i].data.shape_.Size(); ++j) {
        params[i].data.dptr_[j] = param_value_root[j].asFloat();
      }
    }
    
  }
  
  virtual LayerType GetLayerType() { return layer_type; }
  
  virtual void PropAll() {
    for (int i = 0; i < this->BottomNodeNum(); ++i) {
      prop_error.push_back(true);
    }
    for (int i = 0; i < this->ParamNodeNum(); ++i) {
      prop_grad.push_back(true);
    }
  }
  
  virtual std::vector<Node<xpu> >& GetParams() {
    return params;
  }

  virtual void SetPhrase(PhraseType phrase) {
	phrase_type = phrase;
  }
 
  // For Debug
  // If implement net.hpp move to protected
  std::string layer_name;
  int layer_idx;
  std::vector<std::string> bottom_nodes;
  std::vector<std::string> top_nodes; 
 
 // protected:
 public:
  std::vector<Node<xpu> > params;
  std::map<std::string, SettingV> settings;
  std::vector<bool> prop_error;
  std::vector<bool> prop_grad;
  LayerType layer_type;
  PhraseType phrase_type;
  mshadow::Random<xpu> *prnd_;
  
  // required setting
  std::map<std::string, SettingV> defaults;
  
};


template<typename xpu>
Layer<xpu>* CreateLayer(LayerType type);

}  // namespace layer
}  // namespace textnet
#endif  // CXXNET_LAYER_LAYER_H
