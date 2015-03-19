#ifndef TEXTNET_LAYER_LAYER_H_
#define TEXTNET_LAYER_LAYER_H_

#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"
#include "node.h"
#include "../utils/utils.h"
#include "../utils/io.h"

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

  virtual void SaveModel(utils::IStream &fo) {}

  virtual void LoadModel(utils::IStream &fi) {}
  
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
