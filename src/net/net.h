#ifndef TEXTNET_NET_NET_H_
#define TEXTNET_NET_NET_H_

#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"

#include "../layer/layer.h"
#include "../layer/common/lstm_autoencoder_layer-inl.hpp"
#include "../layer/common/lstm_layer-inl.hpp"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../io/json/json.h"
// #include "../statistic/stat.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of net defintiion */
namespace net {
  
using namespace std;
using namespace layer;
using namespace mshadow;

typedef int NetType;
const int kTrainValid = 0;
const int kTrainValidTest = 1;
const int kCrossValid = 2;
const int kTestOnly = 3;
const int kMultiTrainValidTest = 4;

typedef int DeviceType;
const int CPU_DEVICE = 0;
const int GPU_DEVICE = 1;

class INet {
  public:
    virtual ~INet(void){}
    virtual void InitNet(string config_file) = 0;
    virtual void InitNet(Json::Value &net_root) = 0;
    virtual void PropAll() = 0;
    virtual void SetupReshape(string tag) = 0;
    virtual void Reshape(string tag) = 0;
    virtual void Forward(string tag) = 0;
    virtual void Backprop(string tag) = 0;
    virtual void Update(string tag) = 0;
    virtual void SetupAllNets() = 0;
    virtual void TrainOneStep(string tag, int iter) = 0;
    virtual void TrainDisplay(string tag, int iter) = 0;
    virtual void TestAll(string tag, int iter) = 0;
    virtual void Start() = 0;
    virtual void SaveModelActivation(string tag, vector<string> node_names, int num_iter, string file_name, bool save_diff) = 0;
    virtual void SaveModelActivation(int cur_iter) = 0;
    virtual void LoadModel(Json::Value &net_root) = 0;
    virtual void LoadModel(string model_file) = 0;
    virtual void SaveModel(int cur_iter, bool model_save_last) = 0;
    virtual void SaveModel(string file_name, bool save_diff) = 0;
    virtual void PrintClock(string tag) = 0;

  // For Statistic
  virtual Json::Value StatisticNode(Json::Value &req_root) = 0;
  virtual Json::Value StatisticParam(Json::Value &req_root) = 0;
  virtual Json::Value StatisticState(Json::Value &req_root) = 0;
};

template<typename xpu>
class Net : public INet{
 public:
  Net() {
    need_reshape = false;
    var_batch = false;
    model_save_interval = 0;
    model_save_file_prefix = "";
    model_save_last = false;
    model_save_initial = true;
    model_test_initial = true;
    InitSettingEngine();
  }

  
  virtual ~Net(void) {
    mshadow::ShutdownTensorEngine<xpu>(); 
  }
  
  void InitSettingEngine() {
    utils::Printf("[Process] Initial Setting Engine.\n");

    SettingV::SettingIntMap["UnkonwnLayer"] = kUnkonwnLayer;
    // Activation Layer 1-10
    SettingV::SettingIntMap["RectifiedLinear"] = kRectifiedLinear;
    SettingV::SettingIntMap["Sigmoid"] = kSigmoid;
    SettingV::SettingIntMap["Tanh"] = kTanh;

    // Common Layer 11-50
    SettingV::SettingIntMap["FullConnect"] = kFullConnect;
    SettingV::SettingIntMap["Flatten"] = kFlatten;
    SettingV::SettingIntMap["Dropout"] = kDropout;
    SettingV::SettingIntMap["Conv"] = kConv;
    SettingV::SettingIntMap["MaxPooling"] = kMaxPooling;
    SettingV::SettingIntMap["SumPooling"] = kSumPooling;
    SettingV::SettingIntMap["AvgPooling"] = kAvgPooling;
    SettingV::SettingIntMap["Concat"] = kConcat;
    SettingV::SettingIntMap["ChConcat"] = kChConcat;
    SettingV::SettingIntMap["Split"] = kSplit;
    SettingV::SettingIntMap["Embedding"] = kEmbedding;
    SettingV::SettingIntMap["Cross"] = kCross;
    SettingV::SettingIntMap["Match"] = kMatch;
    SettingV::SettingIntMap["Lstm"] = kLstm;
    SettingV::SettingIntMap["Recurrent"] = kRecurrent;
    SettingV::SettingIntMap["TensorFullConnect"] = kTensorFullConnect;
    SettingV::SettingIntMap["WholePooling"] = kWholePooling;
    SettingV::SettingIntMap["Gating"] = kGating;
    SettingV::SettingIntMap["SwapAxis"] = kSwapAxis;

    // Loss Layer 51-70
    SettingV::SettingIntMap["Softmax"] = kSoftmax;
    SettingV::SettingIntMap["L2Loss"] = kL2Loss;
    SettingV::SettingIntMap["MultiLogistic"] = kMultiLogistic;
    SettingV::SettingIntMap["HingeLoss"] = kHingeLoss;
    SettingV::SettingIntMap["PairHingeLoss"] = kPairHingeLoss;
    SettingV::SettingIntMap["Accuracy"] = kAccuracy;

    // Input Layer 71-
    SettingV::SettingIntMap["TextData"] = kTextData;

    // Phrase Type
    SettingV::SettingIntMap["Train"] = kTrain;
    SettingV::SettingIntMap["Test"] = kTest;
   
    using namespace initializer;
    // Initializer
    SettingV::SettingIntMap["Zero"] = kZero;
    SettingV::SettingIntMap["Constant"] = kConstant;
    SettingV::SettingIntMap["Uniform"] = kUniform;
    SettingV::SettingIntMap["Gaussian"] = kGaussian;
    SettingV::SettingIntMap["Xavier"] = kXavier;
    SettingV::SettingIntMap["Kaiming"] = kKaiming;

    using namespace updater;
    // Updater
    SettingV::SettingIntMap["SGD"] = kSGD;
    SettingV::SettingIntMap["Adagrad"] = kAdagrad;
    SettingV::SettingIntMap["Adam"] = kAdam;
    SettingV::SettingIntMap["SGDSparse"] = kSGDSparse;
    SettingV::SettingIntMap["SGDStep"] = kSGDStep;

  }
  
  // expand global_settings to all layer local settings
  void ExpandConfig(Json::Value &net_root) {
    utils::Printf("[Process] Expand Configurations.\n");

    Json::Value &global_root = net_root["global"];
    Json::Value &layers_root = net_root["layers"];
    Json::Value::Members member = global_root.getMemberNames();
    for (Json::Value::Members::iterator it = member.begin();
           it != member.end(); ++it) {
      std::string name = *it; // the place holder
      Json::Value &value = global_root[name];
      Json::Value::Members sub_member = value.getMemberNames();
      
      for (int i = 0; i < layers_root.size(); ++i) {
        if (layers_root[i]["setting"].isMember(name)) { // user should set this place holder in the config file
          layers_root[i]["setting"].removeMember(name);
          for (Json::Value::Members::iterator it = sub_member.begin();
                 it != sub_member.end(); ++it) {
            std::string sub_name = *it;
            layers_root[i]["setting"][sub_name] = value[sub_name];
          }
        }
      }
    }
  }
  
  virtual void InitNet(string config_file) {
    utils::Printf("[Process] Initial Network from file: %s.\n", config_file.c_str());

    ifstream _if(config_file.c_str());
    _if >> root;
    _if.close();
    InitNet(root);
  }

  // redirect stdout to log file
  void setLogFile() {
    if (!root["log"].isNull()) {
      freopen(root["log"].asString().c_str(), "a", stdout);
      setvbuf(stdout, NULL, _IOLBF, 0);
    }
  }
  
  virtual void InitNet(Json::Value &net_root) {

    utils::ShowMemoryUse();

    root = net_root;

    setLogFile();

    // Write original model file to stand output
    Json::StyledWriter writer;
    string json_file = writer.write(root);
    std::cout << "======== Model File ========" << std::endl;
    std::cout << json_file << std::endl;
    std::cout << "======== Model File ========" << std::endl;
    std::cout << std::endl;

    utils::Printf("[Process] Initial Network.\n");

    ExpandConfig(root);
    net_name = root["net_name"].asString();

    // Initial Tensor Engine
    if (root["device_id"].isNull()) {
        device_id = 0;
    } else {
        device_id = root["device_id"].asInt();
    }
    mshadow::InitTensorEngine<xpu>(device_id);
    prnd = new Random<xpu>(59);

    if (!root["need_reshape"].isNull()) {
        need_reshape = root["need_reshape"].asBool();
        utils::Printf("Set need_reshape to %d\n", need_reshape);
    }

    if (!root["var_batch"].isNull()) {
      var_batch = root["var_batch"].asBool();
      utils::Printf("Set var_batch to %d\n", var_batch);
    }

    if (!root["model_save_last"].isNull()) {
      model_save_last = root["model_save_last"].asBool();
      utils::Printf("Set model_save_last to %d\n", model_save_last);
    }

    if (!root["model_save_initial"].isNull()) {
      model_save_initial = root["model_save_initial"].asBool();
      utils::Printf("Set model_save_initial to %d\n", model_save_initial);
    }

    if (!root["model_test_initial"].isNull()) {
      model_test_initial = root["model_test_initial"].asBool();
      utils::Printf("Set model_test_initial to %d\n", model_test_initial);
    }

    ReadNetConfig();
    ReadLayers();
    ReadNodes();
    ReadConnections();

    SetupAllNets();

    ReadParamShare();
    ReadSave();

    // Set init phrase type
    phrase_type = kInit;
    cur_tag = "";
  }

  void ReadNetConfig() {
    // You must define all task tag in this section
    // if layer has no tag, that means share across all tags
    Json::Value &net_config_root = root["net_config"];

    for (int i = 0; i < net_config_root.size(); ++i) {
      Json::Value &one_net = net_config_root[i];
      string tag = one_net["tag"].asString();

      tags.push_back(tag);
      max_iters[tag] = one_net["max_iters"].asInt();
      display_interval[tag] = one_net["display_interval"].asInt();
      
      out_nodes[tag] = vector<string>();
      for (int i = 0; i < one_net["out_nodes"].size(); ++i) {
        out_nodes[tag].push_back(one_net["out_nodes"][i].asString());
      }
      
      out_nodes_type[tag] = vector<string>();
      for (int i = 0; i < one_net["out_nodes"].size(); ++i) {
        if (!one_net["out_nodes_type"].isNull() && i < one_net["out_nodes_type"].size()) {
          out_nodes_type[tag].push_back(one_net["out_nodes_type"][i].asString());
        } else {
          out_nodes_type[tag].push_back("avg");
        }
      }

      // Initial nets vector
      nets[tag] = vector<Layer<xpu>*>();
      utils::Printf("\tTag: %s", tag.c_str());
    }

    utils::Printf("\tDetect %d nets in this config.\n", tags.size());
    
    utils::Printf("\tInitializing Net: %s\n", net_name.c_str());
  }

  void ReadLayers() {
    // ******** Create Layers ********
    utils::Printf("[Process] Creating Layers.\n");
    Json::Value &layers_root = root["layers"];
    
    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value &layer_root = layers_root[i];

      // Get Layer type
      LayerType layer_type = 0;
      if (layer_root["layer_type"].isInt()) {
        layer_type = layer_root["layer_type"].asInt();
      } else if (layer_root["layer_type"].isString()) {
        layer_type = SettingV::SettingIntMap[layer_root["layer_type"].asString()];
      } else {
        utils::Error("[Error] layer type error.\n");
      }
      
      // The default mode of tag is share
      // Share means there is one layer share by multiple nets,
      //   such as a validation net share param with train net
      // New means create a new layer for this tag net,
      //   in order to implement Cross Validation we 
      //   need this kind of logic.
      string tag_mode = "share";
      if (!layer_root["tag_mode"].isNull()) {
        tag_mode = "new";
      }
      
      string layer_name = layer_root["layer_name"].asString();

      if (tag_mode == "share") { 

        // For a layer mode is share
        Layer<xpu> * new_layer = CreateLayer<xpu>(layer_type);
        new_layer->layer_name = layer_name;

        // Reset layer index
        layer_root["layer_idx"] = i;
        new_layer->layer_idx = i;

        if (layer_root["tag"].isNull()) {
          for (int t = 0; t < tags.size(); ++t) {
            nets[tags[t]].push_back(new_layer);
            name2layer[tags[t]][layer_name] = new_layer;
          }
        } else {
          utils::Check(layer_root["tag"].isArray(), 
              "Tag should be an array.");
          for (int t = 0; t < layer_root["tag"].size(); ++t) {
            string tag = layer_root["tag"][t].asString();
            nets[tag].push_back(new_layer);
            name2layer[tag][layer_name] = new_layer;
          }
        }
        layers.push_back(new_layer);

      } else if (tag_mode == "new") {

        // For a layer mode is new
        if (layer_root["tag"].isNull()) {
          for (int t = 0; t < tags.size(); ++t) {
            Layer<xpu> * new_layer = CreateLayer<xpu>(layer_type);
            new_layer->layer_name = layer_name;

            // Reset layer index
            layer_root["layer_idx"] = i;
            new_layer->layer_idx = i;

            nets[tags[t]].push_back(new_layer);
            name2layer[tags[t]][layer_name] = new_layer;
            layers.push_back(new_layer);
          }
        } else {
          utils::Check(layer_root["tag"].isArray(), 
              "Tag should be an array.");
          for (int t = 0; t < layer_root["tag"].size(); ++t) {
            Layer<xpu> * new_layer = CreateLayer<xpu>(layer_type);
            new_layer->layer_name = layer_name;

            // Reset layer index
            layer_root["layer_idx"] = i;
            new_layer->layer_idx = i;

            string tag = layer_root["tag"][t].asString();
            nets[tag].push_back(new_layer);
            name2layer[tag][layer_name] = new_layer;
            layers.push_back(new_layer);
          }
        }
      }
      
      utils::Printf("\t Layer Type: %d\t Layer Name: %s\n", layer_type, layer_name.c_str());  
    }
    
    for (int i = 0; i < tags.size(); ++i) {
      utils::Printf("\t Net[%s] has %d layers.\n", tags[i].c_str(), nets[tags[i]].size());
    }
    utils::Printf("\t Total number of layers is %d. \n", layers.size());
  }

  void ReadNodes() {
    // ******** Create Nodes ********
    utils::Printf("[Process] Creating Nodes.\n");
    Json::Value &layers_root = root["layers"];

    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value &layer_root = layers_root[i];
      Json::Value &bottoms_root = layer_root["bottom_nodes"];
      Json::Value &tops_root = layer_root["top_nodes"];
      for (int j = 0; j < bottoms_root.size(); ++j) {
        string node_name = bottoms_root[j].asString();
        if (!nodes.count(node_name)) {
          nodes[node_name] = new Node<xpu>();
          nodes[node_name]->node_name = node_name;
          node_list.push_back(nodes[node_name]);
          utils::Printf("\t Node Name: %s\n", node_name.c_str());
        }
      }
      for (int j = 0; j < tops_root.size(); ++j) {
        string node_name = tops_root[j].asString();
        if (!nodes.count(node_name)) {
          nodes[node_name] = new Node<xpu>();
          nodes[node_name]->node_name = node_name;
          node_list.push_back(nodes[node_name]);
          utils::Printf("\t Node Name: %s\n", node_name.c_str());
        }
      }
    }

    // Check outnode exist
    for (int t = 0; t < tags.size(); ++t) {
      for (int i = 0; i < out_nodes[tags[t]].size(); ++i) {
        utils::Check(nodes.count(out_nodes[tags[t]][i]), 
        "out_node [%s] not in nodes.", out_nodes[tags[t]][i].c_str());
      }
    }

    utils::Printf("Nodes count: %d\n", nodes.size());
  }

  void ReadConnections() {
    // ******** Connect layers ********
    utils::Printf("[Process] Connecting Layers.\n");
    Json::Value &layers_root = root["layers"];

    bottom_vecs.resize(layers_root.size());
    top_vecs.resize(layers_root.size());

    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value &layer_root = layers_root[i];
      Json::Value &bottoms_root = layer_root["bottom_nodes"];
      Json::Value &tops_root = layer_root["top_nodes"];

      for (int j = 0; j < bottoms_root.size(); ++j) {
        string node_name = bottoms_root[j].asString();
        bottom_vecs[i].push_back(nodes[node_name]);
      }
      for (int j = 0; j < tops_root.size(); ++j) {
        string node_name = tops_root[j].asString();
        top_vecs[i].push_back(nodes[node_name]);
      }
    }
  }

  void ReadParamShare() {
    // ******** Cope with param sharing ********
    utils::Printf("[Process] Add Params Sharing.\n");
    Json::Value &layers_root = root["layers"];

    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value &layer_root = layers_root[i];
      Json::Value &shares_root = layer_root["setting"]["share"];
      if (!shares_root.isNull()) {
        for (int j = 0; j < shares_root.size(); ++j) {
          Json::Value &share_root = shares_root[j];
          string target_layer_name = layer_root["layer_name"].asString();
          string source_layer_name = share_root["source_layer_name"].asString();
          int target_param_id = share_root["param_id"].asInt();
          int source_param_id = share_root["source_param_id"].asInt();

          // orc: may be a bug, layer may not occur in all tags
          for (int t = 0; t < tags.size(); ++t) {
            name2layer[tags[t]][target_layer_name]->ShareParameter(target_param_id,
               name2layer[tags[t]][source_layer_name]->GetParams()[source_param_id]); 
          }

          utils::Printf("\t%s.param[%d] <=== %s.param[%d]\n", 
            target_layer_name.c_str(),
            target_param_id,
            source_layer_name.c_str(),
            source_param_id);
        }
      }
    }
  }

  void ReadSave() {
    // **** read save model and activation config
    Json::Value save_model_root = root["save_model"];
    if (!save_model_root.isNull()) {
      model_save_interval = save_model_root["save_interval"].asInt();
      model_save_file_prefix = save_model_root["file_prefix"].asString();
      if (!save_model_root["everything"].isNull()) {
        model_save_everything = save_model_root["everything"].asBool();
      } else {
        model_save_everything = false;
      }
      if (!save_model_root["everything_once"].isNull()) {
        model_save_everything_once = save_model_root["everything_once"].asBool();
      } else {
        model_save_everything_once = false;
      }
      if (!save_model_root["save_diff"].isNull()) {
        model_save_diff = save_model_root["save_diff"].asBool();
      } else {
        model_save_diff = false;
      }
    }
    Json::Value save_act_root = root["save_activation"];
    if (!save_act_root.isNull()) {
      for (int i = 0; i < save_act_root.size(); ++i) {
        Json::Value tag_act_root = save_act_root[i];
        string tag = tag_act_root["tag"].asString();
        activation_save_file_prefix[tag] = tag_act_root["file_prefix"].asString();
        activation_save_interval[tag] = tag_act_root["save_interval"].asInt();
        activation_save_iter_num[tag] = tag_act_root["save_iter_num"].asInt();

        if (!tag_act_root["save_diff"].isNull()) {
          activation_save_diff[tag] = tag_act_root["save_diff"].asBool();
        } else {
          activation_save_diff[tag] = false;
        }
        
        Json::Value save_nodes_root = tag_act_root["save_nodes"];
        vector<string> save_nodes;
        if (save_nodes_root.isNull()) {
          for (size_t l = 0; l < nets[tag].size(); ++l) {
            int layer_idx = nets[tag][l]->layer_idx;
            for (size_t node_idx = 0; node_idx < top_vecs[layer_idx].size(); ++node_idx) {
              save_nodes.push_back(top_vecs[layer_idx][node_idx]->node_name);
            }
          }
        } else {
          for (int name_idx = 0; name_idx < save_nodes_root.size(); ++name_idx) {
            utils::Check(nodes.count(save_nodes_root[name_idx].asString()), 
              "save_node [%s] not in nodes.", save_nodes_root[name_idx].asString().c_str());
            save_nodes.push_back(save_nodes_root[name_idx].asString());
          }
        }
        activation_save_nodes[tag] = save_nodes;
      }
    }
  }

  virtual void PropAll() {
    utils::Printf("[Process] PropAll Layers.\n");
    for (int i = 0; i < layers.size(); ++i) {
      layers[i]->PropAll();
    }
  }
  
  virtual void SetupReshape(string tag) {
    utils::Printf("[Process] Setup Layers.\n");
    Json::Value &layers_root = root["layers"];
    
    for (int i = 0; i < nets[tag].size(); ++i) {
      int layer_idx = nets[tag][i]->layer_idx;
      utils::Printf("[layer] set layer %s\n", nets[tag][i]->layer_name.c_str());
      nets[tag][i]->SetupLayer(layers_root[layer_idx], 
          bottom_vecs[layer_idx], top_vecs[layer_idx], prnd);
      nets[tag][i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx], true);
      utils::ShowMemoryUse();
    }
  }

  virtual void Reshape(string tag) {
    utils::Printf("[Process] Reshape network.\n");
    for (int i = 0; i < nets[tag].size(); ++i) {
      int layer_idx = nets[tag][i]->layer_idx;
#if DEBUG
      utils::Printf("[layer] set layer %s\n", nets[tag][i]->layer_name.c_str());
      nets[tag][i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx], true);
#else 
      nets[tag][i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx], false);
#endif
    }
  }
  
  virtual void SetPhrase(string tag, PhraseType phrase) {
    if (phrase_type == phrase && cur_tag == tag) return;

    utils::Printf("[Process] Set Tag to %s.\n", tag.c_str());
    utils::Printf("[Process] Set Phrase to %d.\n", phrase);
    phrase_type = phrase;
    cur_tag = tag;
    for (int i = 0; i < nets[tag].size(); ++i) {
      nets[tag][i]->SetPhrase(phrase);
    }
    if (need_reshape) Reshape(tag);
  }

  virtual void Forward(string tag) {
      for (int i = 0; i < nets[tag].size(); ++i) {
        int layer_idx = nets[tag][i]->layer_idx;

        if (var_batch) {
#if DEBUG
          cout << "Layer " << layer_idx << endl;
          cout << "\tBefore" << endl;
          for (int j = 0; j < top_vecs[layer_idx].size(); ++j) {
            cout << "\tNode " << j << " data :" << " ";
            for (int k = 0; k < 4; ++k) {
              cout << top_vecs[layer_idx][j]->data.size(k) << " x ";
            }
            cout << endl;
            cout << "\tNode " << j << " len :" << " ";
            for (int k = 0; k < 2; ++k) {
              cout << top_vecs[layer_idx][j]->length.size(k) << " x ";
            }
            cout << endl;
          }
#endif
          nets[tag][i]->CheckReshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
#if DEBUG
          cout << "\tAfter" << endl;
          for (int j = 0; j < top_vecs[layer_idx].size(); ++j) {
            cout << "\tNode " << j << " data :" << " ";
            for (int k = 0; k < 4; ++k) {
              cout << top_vecs[layer_idx][j]->data.size(k) << " x ";
            }
            cout << endl;
            cout << "\tNode " << j << " len :" << " ";
            for (int k = 0; k < 2; ++k) {
              cout << top_vecs[layer_idx][j]->length.size(k) << " x ";
            }
            cout << endl;
          }
#endif
        }

#if TIME_DEBUG
        nets[tag][i]->ClockStart(0);
#endif

        nets[tag][i]->Forward(bottom_vecs[layer_idx], top_vecs[layer_idx]);

#if TIME_DEBUG
        nets[tag][i]->ClockStop(0);
#endif

#if LENGTH_DEBUG
        for (int j = 0; j < top_vecs[layer_idx].size(); ++j) {
          cout << top_vecs[layer_idx][j]->node_name << endl;
          for (int d1 = 0; d1 < top_vecs[layer_idx][j]->length.size(0); ++d1) {
            for (int d2 = 0; d2 < top_vecs[layer_idx][j]->length.size(1); ++d2) {
              cout << top_vecs[layer_idx][j]->length[d1][d2] << " ";    
            }
            cout << endl;
          } 
          cout << endl;
        }
#endif
#if DEBUG
        cout << "Feed " ;
        for (int j = 0; j < bottom_vecs[layer_idx].size(); ++j)
            cout << bottom_vecs[layer_idx][j]->node_name << ", ";
        cout << " and ";
        for (int j = 0; j < top_vecs[layer_idx].size(); ++j)
            cout << top_vecs[layer_idx][j]->node_name << ", ";
        cout << " to " << nets[tag][i]->layer_name << endl;
#endif
    }
  }

  virtual void Backprop(string tag) {
    utils::Check(phrase_type == kTrain, 
                  "Only call in Train Phrase.");
    for (int i = nets[tag].size()-1; i >= 0; --i) {
        int layer_idx = nets[tag][i]->layer_idx;
        nets[tag][i]->ClearDiff(bottom_vecs[layer_idx], top_vecs[layer_idx]);
    }
    for (int i = nets[tag].size()-1; i>=0; --i) {
      int layer_idx = nets[tag][i]->layer_idx;

#if TIME_DEBUG
        nets[tag][i]->ClockStart(1);
#endif

      nets[tag][i]->Backprop(bottom_vecs[layer_idx], top_vecs[layer_idx]);

#if TIME_DEBUG
        nets[tag][i]->ClockStop(1);
#endif

#if DEBUG
      cout << "BP " << nets[tag][i]->layer_name << endl;
#endif
    }
    NormLstmGradient(tag);
  }

  float Norm2Square(mshadow::Tensor<xpu, 4, float> t) {
    float norm2_square = 0.f;
    for (index_t i = 0; i < t.size(0); ++i) 
      for (index_t j = 0; j < t.size(1); ++j)
        for (index_t m = 0; m < t.size(2); ++m)
          for (index_t n = 0; n < t.size(3); ++n)
              norm2_square += t[i][j][m][n] * t[i][j][m][n];

    return norm2_square;
  }
  // orc this is for lstm or rnn, if the gradients of parameters are too big, 
  // rescale all layers' gradients
  void NormLstmGradient(string tag) {
    utils::Check(phrase_type == kTrain, "Only call in Train Phrase.");
    float norm2 = 0.f;
    float max_norm2 = 0.f;
    for (int i = 0; i < nets[tag].size(); ++i) {
      int layer_type = nets[tag][i]->layer_type;
      if (layer_type != kRecurrent && layer_type != kLstm && layer_type != kLstmAutoencoder) {
        continue;
      } 
      if (layer_type == kLstmAutoencoder) {
        max_norm2 = ((LstmAutoencoderLayer<xpu> *)(nets[tag][i]))->max_norm2;
      }
      if (layer_type == kLstm) {
        max_norm2 = ((LstmLayer<xpu> *)(nets[tag][i]))->max_norm2;
      }
      if (max_norm2 == 0.f) {
        return;
      }
      for (int param_id = 0; param_id < nets[tag][i]->ParamNodeNum(); ++param_id) {
        norm2 += Norm2Square(nets[tag][i]->params[param_id].diff);
      }
    }
    if (max_norm2 == 0.f) 
      return;
    norm2 = sqrt(norm2);
    if (norm2 <= max_norm2) 
      return;
    float scale = max_norm2/norm2;
    utils::Printf("Rescale Gradient By %f.\n", scale);
    for (int i = 0; i < nets[tag].size(); ++i) {
      for (int param_id = 0; param_id < nets[tag][i]->ParamNodeNum(); ++param_id) {
        nets[tag][i]->params[param_id].diff *= scale;
      }
    }
  }
  
  virtual void Update(string tag) {
    utils::Check(phrase_type == kTrain, 
                  "Only call in Train Phrase.");
    for (int i = 0; i < nets[tag].size(); ++i) {
      for (int j = 0; j < nets[tag][i]->ParamNodeNum(); ++j) {
#if DEBUG
        cout << "Update param in layer " << i << "(" << nets[tag][i]->layer_name.c_str() << ") params " << j << endl;
        cout << "param data" << i << " , " << j << ": " << nets[tag][i]->GetParams()[j].data[0][0][0][0] 
             << "\t" << nets[tag][i]->GetParams()[j].data[0][0][0][1]
             << endl;
        cout << "param data" << i << " , " << j << ": " << nets[tag][i]->GetParams()[j].diff[0][0][0][0]
             << "\t" << nets[tag][i]->GetParams()[j].diff[0][0][0][1]
             << endl;
#endif

#if TIME_DEBUG
        nets[tag][i]->ClockStart(2);
#endif

        nets[tag][i]->GetParams()[j].Update();

#if TIME_DEBUG
        nets[tag][i]->ClockStop(2);
#endif

#if DEBUG
        cout << "param data" << i << " , " << j << ": " << nets[tag][i]->GetParams()[j].data[0][0][0][0]
             << "\t" << nets[tag][i]->GetParams()[j].data[0][0][0][1]
             << endl;
#endif
      }
    }
  }

  virtual void SetupAllNets() {
    // Prepare
    for (int i = 0; i < tags.size(); ++i) {
      SetupReshape(tags[i]);
    }
    //PropAll();
  }
  
  virtual void TrainOneStep(string tag, int iter = 0) {
    SetPhrase(tag, kTrain);

    Forward(tag);
    Backprop(tag);

#if DEBUG
    // For debug
    //for (typename map<string, Node<xpu>*>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    for (int k = 0; k < node_list.size(); ++k) {
      string name = node_list[k]->node_name; //it->first;
      cout << "Snapshot [" << name << "]" << endl;
      cout << "data : ";
    if (utils::checkNan(node_list[k]->data.dptr_, 
          node_list[k]->data.size(0)*node_list[k]->data.size(1)*node_list[k]->data.size(2)*node_list[k]->data.size(3))) {
      cout << "[Error] Contain NAN!" << endl;
    }
      for (int i = 0; i < 5; ++i) {
        cout << node_list[k]->data[0][0][0][i] << "\t";
      }
      cout << endl;
      cout << "diff : ";
    if (utils::checkNan(node_list[k]->diff.dptr_, 
          node_list[k]->diff.size(0)*node_list[k]->diff.size(1)*node_list[k]->diff.size(2)*node_list[k]->diff.size(3))) {
      cout << "[Error] Contain NAN!" << endl;
    }
      for (int i = 0; i < 5; ++i) {
        cout << node_list[k]->diff[0][0][0][i] << "\t";
      }
      cout << endl;
      cout << "length : ";
    if (utils::checkNan(node_list[k]->length.dptr_, 
          node_list[k]->length.size(0)*node_list[k]->length.size(1))) {
      cout << "[Error] Contain NAN!" << endl;
    }
      for (int i = 0; i < 5; ++i) {
        cout << node_list[k]->length[0][i] << "\t";
      }
      cout << endl;
    }
#endif

    Update(tag);
  }

  virtual void TrainDisplay(string tag, int iter = 0) {
    for (int i = 0; i < out_nodes[tag].size(); ++i) {
      utils::Printf("[%s:kTrain]\tIter\t%d:\tOut[%s] =\t%f\n", 
          tag.c_str(), iter, out_nodes[tag][i].c_str(), 
          nodes[out_nodes[tag][i]]->data_d1()[0]);
      // cout << "[" << tag << ":kTrain]\tIter\t" << iter 
      //      << ":\tOut[" << out_nodes[tag][i] << "] =\t" 
      //      << nodes[out_nodes[tag][i]]->data_d1()[0] << endl; 
    }
  }
      
  virtual void PrintClock(string tag) {
      for (int i = 0; i < nets[tag].size(); ++i) {
        nets[tag][i]->ClockPrint();
      }
  }

  virtual void TestAll(string tag, int iter = 0) {
      SetPhrase(tag, kTest);

      // Initial test loss
      vector<float> test_loss;
      vector< vector<float> > test_loss_list;
      for (int i = 0; i < out_nodes[tag].size(); ++i) {
        test_loss.push_back(0.0f);
        test_loss_list.push_back(vector<float>());
      }
#if DEBUG
      cout<<"Start TestALL ..."<<endl;
#endif
      
      for (int test_iter = 0; test_iter < max_iters[tag]; ++test_iter) {
        Forward(tag);
        for (int i = 0; i < out_nodes[tag].size(); ++i) {
          test_loss_list[i].push_back(nodes[out_nodes[tag][i]]->data_d1()[0]);
        }
      }

      // Reduce to one output
      for (int i = 0; i < out_nodes[tag].size(); ++i) {
        if (out_nodes_type[tag][i] == "avg") {
          test_loss[i] = accumulate(test_loss_list[i].begin(), test_loss_list[i].end(), 0.0f);
          test_loss[i] /= test_loss_list[i].size();
        } else if (out_nodes_type[tag][i] == "sum") {
          test_loss[i] = accumulate(test_loss_list[i].begin(), test_loss_list[i].end(), 0.0f);
        } else if (out_nodes_type[tag][i] == "median") {
          int med_idx = test_loss_list[i].size() / 2;
          partial_sort( test_loss_list[i].begin(), 
                        test_loss_list[i].begin() + med_idx + 1, 
                        test_loss_list[i].end() );

          if (med_idx % 2 == 0) {
            test_loss[i] = test_loss_list[i][med_idx];
          } else {
            test_loss[i] = (test_loss_list[i][med_idx-1] + test_loss_list[i][med_idx]) * 0.5;
          }
        }
      }
      
      // Output
      for (int i = 0; i < out_nodes[tag].size(); ++i) {
        utils::Printf("[%s:kTest]\tIter\t%d:\tOut[%s]#%s =\t%f\n",
            tag.c_str(), iter, out_nodes[tag][i].c_str(), out_nodes_type[tag][i].c_str(), test_loss[i]);
        // cout << "[" << tag << ":kTest]\tIter\t" << iter 
        //      << ":\tOut[" << out_nodes[tag][i] << "] =\t" 
        //      << test_loss[i] << endl; 
      }
  }

  virtual void SaveModelActivation(int cur_iter) {
    for (map<string, int>::iterator it = activation_save_interval.begin();
         it != activation_save_interval.end(); ++it) {
      const string &tag = it->first;
      if (cur_iter % it->second != 0) {
          continue;
      }
      string file_name = activation_save_file_prefix[tag] + "." + int2str(cur_iter);
      SaveModelActivation(tag, activation_save_nodes[tag], activation_save_iter_num[tag], file_name, activation_save_diff[tag]);
    }
  }

  virtual void SaveModel(int cur_iter, bool last_iter = false) {
    if (!last_iter && (model_save_interval <= 0 || cur_iter % model_save_interval != 0)) {
        return;
    }
    string file_name = model_save_file_prefix + "." + int2str(cur_iter);
    SaveModel(file_name, model_save_diff);
  }
 
  virtual void SaveModelActivation(string tag, vector<string> node_names, int num_iter, string file_name, bool save_diff = false) {
    utils::Printf("[Save] Save activation to %s.\n", file_name.c_str());
    SetPhrase(tag, kTest);
    Json::Value iters_root;
    for (int iter = 0; iter < num_iter; ++iter) {
      Forward(tag);
      Json::Value nodes_root;
      for (int i = 0; i < node_names.size(); ++i) {
        string name = node_names[i];
        Json::Value node_root;
        nodes[name]->SaveNode(node_root, save_diff);
        nodes_root[name] = node_root;
      }
      iters_root.append(nodes_root);
    }
    ofstream ofs(file_name.c_str());
    // Json::FastWriter writer;
    Json::StyledWriter writer;
    string json_file = writer.write(iters_root);
    ofs << json_file;
    ofs.close();
  }
  
  virtual void SaveModel(string model_file, bool save_diff = false) {
    utils::Printf("[Save] Save model to %s.\n", model_file.c_str());
    ofstream ofs(model_file.c_str());
    Json::StyledWriter writer;
    Json::Value net_root, layers_params_root;
    net_root["config"] = root;
    for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
      Json::Value layer_params_root;
      for (int param_idx = 0; param_idx < layers[layer_idx]->ParamNodeNum(); ++param_idx) {
        if (layers[layer_idx]->params[param_idx].is_share) {
            layer_params_root.append(0);
            continue;
        }
        // Check for embedding saving 
        if (!model_save_everything && !model_save_everything_once && \
             (layers[layer_idx]->layer_type == kEmbedding || \
             layers[layer_idx]->layer_type == kWordClassSoftmaxLoss) ) {
            cout << "\t Without save embedding, in layer " << layers[layer_idx]->layer_name << "." << endl;
            layer_params_root.append(0);
            continue;
        }
        // save the content of the matrix
        Json::Value node_root;
        layers[layer_idx]->params[param_idx].SaveNode(node_root, save_diff);
        layer_params_root.append(node_root);
      }
      layers_params_root.append(layer_params_root);
    }
    net_root["layers_params"] = layers_params_root;
    string json_file = writer.write(net_root);
    ofs << json_file;
    ofs.close();

    // Reset save everything_once
    if (model_save_everything_once) {
        model_save_everything_once = false;
        cout << "\t Turn off save everything." << endl;
    }
    
  }

  void LoadParams(Json::Value &layers_params_root) {
    utils::Printf("[Load] Load Params to Net.\n");
    for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
      for (int param_idx = 0; param_idx < layers[layer_idx]->ParamNodeNum(); ++param_idx) {
        if (layers[layer_idx]->params[param_idx].is_share) {
          continue;
        }

        if (layers_params_root[layer_idx].isNull()) {
          utils::Printf("\tNo Initial Params at layer: %d\n", layer_idx);
          continue;
        }

        if (layers_params_root[layer_idx][param_idx].isNull()) {
          utils::Printf("\tNo Initial Params at layer: %d, param: \n", layer_idx, param_idx);
          continue;
        }

        Json::Value node_root = layers_params_root[layer_idx][param_idx];
        layers[layer_idx]->params[param_idx].LoadNode(node_root, false);
      }
    }
  }

  virtual void LoadModel(string model_file) {
    Json::Value net_root;
    ifstream ifs(model_file.c_str());
    ifs >> net_root;
    ifs.close();
    LoadModel(net_root);
  }

  virtual void LoadModel(Json::Value &net_root) {
  utils::Check(!net_root["config"].isNull(), "No [config] section.");
  utils::Check(!net_root["layers_params"].isNull(), "No [layers_params] section.");
    root = net_root["config"];
    InitNet(root);
    LoadParams(net_root["layers_params"]);
  }

  
  virtual void Start() = 0;

  virtual Json::Value StatisticNode(Json::Value &req_root) {
    bool rtn = utils::Require(!req_root["node_name"].isNull(), "Require node_name!\n") &&
               utils::Require(!req_root["static_node"].isNull(), "Require static_node!(data or diff)\n") &&
               utils::Require(!req_root["static_value"].isNull(), "Require static_value!\n");
    Json::Value rtn_root;
    if (rtn) {
      string node_name = req_root["node_name"].asString();
      rtn_root["request"] = req_root;
      for (int i = 0; i < req_root["static_node"].size(); ++i) {
        if (req_root["static_node"][i] == "data") {
          rtn_root["data"] = nodes[node_name]->data_statistic(req_root["static_value"]);
        } else if (req_root["static_node"][i] == "diff") {
          rtn_root["diff"] = nodes[node_name]->diff_statistic(req_root["static_value"]);
        }
      }
    }
    return rtn_root;
  }

  virtual Json::Value StatisticParam(Json::Value &req_root) {
    bool rtn = utils::Require(!req_root["tag"].isNull(), "Require tag!\n") &&
               utils::Require(!req_root["layer_name"].isNull(), "Require node_name!\n") &&
               utils::Require(!req_root["param_id"].isNull(), "Require param_id!\n") &&
               utils::Require(!req_root["static_node"].isNull(), "Require static_node!(data or diff)\n") &&
               utils::Require(!req_root["static_value"].isNull(), "Require static_value!\n");
    Json::Value rtn_root;
    if (rtn) {
      string tag = req_root["tag"].asString();
      string layer_name = req_root["layer_name"].asString();
      int param_id = req_root["param_id"].asInt();
      rtn_root["request"] = req_root;
      for (int i = 0; i < req_root["static_node"].size(); ++i) {
        if (req_root["static_node"][i] == "data") {
          rtn_root["data"] = name2layer[tag][layer_name]->GetParams()[param_id].data_statistic(req_root["static_value"]);
        } else if (req_root["static_node"][i] == "diff") {
          rtn_root["diff"] = name2layer[tag][layer_name]->GetParams()[param_id].diff_statistic(req_root["static_value"]);
        }
      }
    }
    return rtn_root;
  }

  virtual Json::Value StatisticState(Json::Value &req_root) {
  Json::Value rtn_root;

  return rtn_root;
  }
  
 public:
  // Net name 
  string net_name;
  // Net type
  NetType net_type;
  // Random Machine for all
  mshadow::Random<xpu>* prnd;
  // Tag for different tasks
  vector<string> tags;
  // Nets for muti-tasks 
  map<string, vector<Layer<xpu>*> > nets;
  // max iterations for nets
  map<string, int> max_iters;
  // display interval for nets
  map<string, int> display_interval;
  // save interval for nets
  map<string, int> activation_save_interval;
  // save iter num for nets
  map<string, int> activation_save_iter_num;
  // save file prefix, subfixed by training iter num
  map<string, string> activation_save_file_prefix;
  // save diff for nets
  map<string, bool> activation_save_diff;
  // save nodes
  map<string, vector<string> > activation_save_nodes;
  int model_save_interval;
  string model_save_file_prefix;
  bool model_save_everything;
  bool model_save_everything_once;
  bool model_save_initial;
  bool model_test_initial;
  bool model_save_last;
  bool model_save_diff;
  // is save best or not, output file is file_prefix + ".best" // to do
  // map<string, bool> save_best;
  // nets output nodes
  map<string, vector<string> > out_nodes;
  // nets output nodes reduce type
  map<string, vector<string> > out_nodes_type;
  // All layers
  vector<Layer<xpu>*> layers;
  // Add tags to name search, 
  // there exist same name in different tag net
  map<string, map<string, Layer<xpu>*> > name2layer;
  // Nodes to store datum between layers
  map<string, Node<xpu>*> nodes;
  // bottom vectors
  vector<vector<Node<xpu>*> > bottom_vecs;
  // top vectors
  vector<vector<Node<xpu>*> > top_vecs;
  // phrase type
  PhraseType phrase_type;
  // current tag
  string cur_tag;
  // Config
  Json::Value root;
  // need reshape : when change tag/phrase change shape
  bool need_reshape;
  // var batch : every batch is different, need check
  bool var_batch;
  // node list
  vector<Node<xpu>*> node_list;

  // gpu device id
  int device_id;
};

INet* CreateNetCPU(NetType type);
INet* CreateNetGPU(NetType type);
inline INet* CreateNet(DeviceType device_type, NetType net_type) {
  switch(device_type) {
    case CPU_DEVICE:
        return CreateNetCPU(net_type);
    case GPU_DEVICE:
        return CreateNetGPU(net_type);
    default:
        utils::Error("Invalid device type.");
  }
  return NULL;
}
}  // namespace net
}  // namespace textnet
#endif  // TEXTNET_NET_NET_H_
