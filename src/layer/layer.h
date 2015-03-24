#ifndef TEXTNET_LAYER_LAYER_H_
#define TEXTNET_LAYER_LAYER_H_

#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"
#include "node.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../io/json/json.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of layer defintiion */
namespace layer {

/*! \brief use integer to encode layer types */
typedef int LayerType;
typedef int PhraseType;

template<typename xpu>
class Layer {
 public:
  Layer(void) {
    layer_type = 0;
    phrase_type = 0;
  }
  virtual ~Layer(void) {}
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting, 
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    this->settings = setting;
    phrase_type = setting["phrase_type"].i_val;
    prnd_ = prnd;
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {}

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) = 0;

  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) = 0;
                        
  virtual int BottomNodeNum() = 0;
  virtual int TopNodeNum() = 0;

  virtual int ParamNodeNum() = 0;

  void SaveSetting(std::map<std::string, SettingV> &setting, Json::Value &root) {
    for (std::map<std::string, SettingV>::iterator it = setting.begin(); 
         it != setting.end(); ++it) {
      switch ( it->second.value_type ) {
		  case SET_INT:
			  {
			    root[it->first] = it->second.i_val;
			  }
			  break;
		  case SET_FLOAT:
			  {
                root[it->first] = it->second.f_val;
			  }
			  break;
		  case SET_BOOL:
			  {
                root[it->first] = it->second.b_val;
			  }
			  break;
		  case SET_STRING:
			  {
                root[it->first] = it->second.s_val;
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

  virtual void SaveModel(Json::Value &layer_root) {
    // Set layer type
    layer_root["layer_type"] = layer_type;
    layer_root["layer_name"] = layer_name;
    layer_root["layer_idx"] = layer_idx;
    
    // Set layer settings
    Json::Value setting_root;
    SaveSetting(settings, setting_root);
    layer_root["setting"] = setting_root;
    
    // Set layer weights
    for (int i = 0; i < params.size(); ++i) {
      Json::Value param_root;
      for (int j = 0; j < params[i].data.shape_.Size(); ++j) {
        param_root.append(params[i].data.dptr_[j]);
      }
      char param_name[100] = "param";
      param_name[5] = i + '0';
      param_name[6] = '\0';
      layer_root[param_name] = param_root;
    }
  }

  virtual void LoadModel(Json::Value &layer_root) {
    
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
 
 protected:
  std::vector<Node<xpu> > params;
  std::map<std::string, SettingV> settings;
  std::vector<bool> prop_error;
  std::vector<bool> prop_grad;
  LayerType layer_type;
  std::string layer_name;
  int layer_idx;
  PhraseType phrase_type;
  mshadow::Random<xpu> *prnd_;
  
};

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

// Loss Layer 51-70
const int kSoftmax = 51;
const int kL2Loss = 52;
const int kMultiLogistic = 53;
const int kHingeLoss = 54;
const int kPairHingeLoss = 55;

// Input Layer 71-
const int kTextData = 71;


/*! \brief these are enumeration */
const int kTrain = 0;
const int kValidation = 1;
const int kTest = 2;


template<typename xpu>
Layer<xpu>* CreateLayer(LayerType type);

}  // namespace layer
}  // namespace textnet
#endif  // CXXNET_LAYER_LAYER_H
