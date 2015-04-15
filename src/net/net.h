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
    virtual void SaveModelActivation(string tag, string dir_path, vector<string> node_names, int num_iter) = 0;
    virtual void LoadModel(Json::Value &layer_root) = 0;
    virtual void SaveModel(string model_name) = 0;
};

template<typename xpu>
class Net : public INet{
 public:
  Net() {
    need_reshape = false;
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
  }
  
  void ExpandConfig(Json::Value &net_root) {
    utils::Printf("[Process] Expand Configurations.\n");

    Json::Value &global_root = net_root["global"];
    Json::Value &layers_root = net_root["layers"];
    Json::Value::Members member = global_root.getMemberNames();
    for (Json::Value::Members::iterator it = member.begin();
           it != member.end(); ++it) {
      std::string name = *it;
      Json::Value &value = global_root[name];
      Json::Value::Members sub_member = value.getMemberNames();
      
      for (int i = 0; i < layers_root.size(); ++i) {
        if (layers_root[i]["setting"].isMember(name)) {
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
    string log_file = root["log"].asString();
    if (!log_file.empty()) {
        freopen(log_file.c_str(), "w", stdout);
        setvbuf(stdout, NULL, _IOLBF, 0);
    }
    InitNet(root);
  }
  
  virtual void InitNet(Json::Value &net_root) {
    utils::Printf("[Process] Initial Network.\n");

    root = net_root;
    ExpandConfig(root);
    net_name = net_root["net_name"].asString();

    // Initial Tensor Engine
    if (net_root["device_id"].isNull()) {
        device_id = 0;
    } else {
        device_id = net_root["device_id"].asInt();
    }
    mshadow::InitTensorEngine<xpu>(device_id);
    prnd = new Random<xpu>(59);

    // You must define all task tag in this section
    // if layer has no tag, that means share across all tags
    Json::Value &net_config_root = net_root["net_config"];

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

      // Initial nets vector
      nets[tag] = vector<Layer<xpu>*>();
      utils::Printf("\tTag: %s", tag.c_str());
    }

    utils::Printf("\tDetect %d nets in this config.\n", tags.size());
    
    utils::Printf("\tInitializing Net: %s\n", net_name.c_str());
    
    // ******** Create Layers ********
    utils::Printf("[Process] Creating Layers.\n");
    Json::Value &layers_root = net_root["layers"];
    
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
	  // Share means there is one layer share by muti net,
	  //   such as a validation net share param with train net
	  // New means create a new layer for this tag net,
	  //   in order to implement Cross Validation we 
	  //   need this kind of logic.
      string tag_mode = "share";
      if (!layer_root["tag_mode"].isNull()) {
        tag_mode = "new";
      }
      
      string layer_name = layer_root["layer_name"].asString();

	  // For a layer mode is share
      if (tag_mode == "share") { 
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

    // ******** Create Nodes ********
    utils::Printf("[Process] Creating Nodes.\n");
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

    utils::Printf("Nodes count: %d\n", nodes.size());
    
    // ******** Connect layers ********
    utils::Printf("[Process] Connecting Layers.\n");

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

    // ******** Cope with param sharing ********
    utils::Printf("[Process] Add Params Sharing.\n");

    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value &layer_root = layers_root[i];
      Json::Value &shares_root = layer_root["share"];
      if (!shares_root.isNull()) {
        for (int j = 0; j < shares_root.size(); ++j) {
          Json::Value &share_root = shares_root[j];
          string target_layer_name = layer_root["layer_name"].asString();
          string source_layer_name = share_root["source_layer_name"].asString();
          int target_param_id = share_root["param_id"].asInt();
          int source_param_id = share_root["source_param_id"].asInt();

          for (int t = 0; t < tags.size(); ++t) {
            name2layer[tags[t]][target_layer_name]->ShareParameter(target_param_id,
               name2layer[tags[t]][source_layer_name]->GetParams()[source_param_id]); 
          }

          utils::Printf("\t%s.param[%d] === %s.param[%d]\n", 
                target_layer_name.c_str(),
                target_param_id,
                source_layer_name.c_str(),
                source_param_id);
        }
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
      nets[tag][i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
    }
  }

  virtual void Reshape(string tag) {
    utils::Printf("[Process] Reshape network.\n");
    for (int i = 0; i < nets[tag].size(); ++i) {
      int layer_idx = nets[tag][i]->layer_idx;
      nets[tag][i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
    }
  }
  
  virtual void SetPhrase(string tag, PhraseType phrase) {
    if (phrase_type == phrase) return;

    utils::Printf("[Process] Set Phrase to %d.\n", phrase);
    phrase_type = phrase;
    for (int i = 0; i < nets[tag].size(); ++i) {
      nets[tag][i]->SetPhrase(phrase);
    }
    if (need_reshape) Reshape(tag);
  }

  virtual void Forward(string tag) {
      for (int i = 0; i < nets[tag].size(); ++i) {
        int layer_idx = nets[tag][i]->layer_idx;
        nets[tag][i]->Forward(bottom_vecs[layer_idx], top_vecs[layer_idx]);
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
      nets[tag][i]->Backprop(bottom_vecs[layer_idx], top_vecs[layer_idx]);
    }
  }
  
  virtual void Update(string tag) {
    utils::Check(phrase_type == kTrain, 
                  "Only call in Train Phrase.");
    for (int i = 0; i < nets[tag].size(); ++i) {
      for (int j = 0; j < nets[tag][i]->ParamNodeNum(); ++j) {
#if DEBUG
        cout << "Update param in layer " << i << " params " << j << endl;
        cout << "param data" << i << " , " << j << ": " << nets[tag][i]->GetParams()[j].data[0][0][0][0] 
             << "\t" << nets[tag][i]->GetParams()[j].data[0][0][0][1]
             << endl;
        cout << "param data" << i << " , " << j << ": " << nets[tag][i]->GetParams()[j].diff[0][0][0][0]
             << "\t" << nets[tag][i]->GetParams()[j].diff[0][0][0][1]
             << endl;
#endif
        nets[tag][i]->GetParams()[j].Update();
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
    PropAll();
    for (int i = 0; i < tags.size(); ++i) {
      SetupReshape(tags[i]);
    }
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
      for (int i = 0; i < 5; ++i) {
        cout << node_list[k]->data[0][0][0][i] << "\t";
      }
      cout << endl;
      cout << "diff : ";
      for (int i = 0; i < 5; ++i) {
        cout << node_list[k]->diff[0][0][0][i] << "\t";
      }
      cout << endl;
    }
#endif

    Update(tag);
  }

  virtual void TrainDisplay(string tag, int iter = 0) {
    for (int i = 0; i < out_nodes[tag].size(); ++i) {
      cout << "[Train]\tIter\t" << iter 
           << ":\tOut[" << out_nodes[tag][i] << "] =\t" 
           << nodes[out_nodes[tag][i]]->data_d1()[0] << endl; 
    }
  }
      
  virtual void TestAll(string tag, int iter = 0) {
      SetPhrase(tag, kTest);

      // Initial test loss
      vector<float> test_loss;
      for (int i = 0; i < out_nodes[tag].size(); ++i) {
        test_loss.push_back(0.0f);
      }
      
      for (int test_iter = 0; test_iter < max_iters[tag]; ++test_iter) {
        Forward(tag);
        for (int i = 0; i < out_nodes[tag].size(); ++i) {
          test_loss[i] += nodes[out_nodes[tag][i]]->data_d1()[0];
        }
        // orc_tmp
        // cout << "test loss:" << nodes[test_out[0]]->data_d1()[0] << endl;
      }
      
      for (int i = 0; i < out_nodes[tag].size(); ++i) {
        test_loss[i] /= max_iters[tag];
      }
      
      // Output
      for (int i = 0; i < out_nodes[tag].size(); ++i) {
        cout << "[" << tag << "]\tIter\t" << iter 
             << ":\tOut[" << out_nodes[tag][i] << "] =\t" 
             << test_loss[i] << endl; 
      }
  }
 
  virtual void SaveModelActivation(string tag, string dir_path, vector<string> node_names, int num_iter) {
    SetPhrase(tag, kTrain);
    for (int iter = 0; iter < num_iter; ++iter) {
      Forward(tag);
      cout << "Forward " << iter << " over!" << endl;
      for (int i = 0; i < node_names.size(); ++i) {
        string name = node_names[i];
        Json::Value node_root;

        nodes[name]->SaveNode(node_root);

        // Prepare for output filename
        ostringstream file_name;
        file_name << dir_path << name << "_" << iter << ".json";

        cout << "Save node " << name << " to " << file_name.str() << endl;
        // Write Node to file
        ofstream _of(file_name.str().c_str());
        Json::FastWriter writer;
        string json_file = writer.write(node_root);
        _of << json_file;
        _of.close();

      }
    }
  }
  
  virtual void SaveModel(string model_name) {
    ofstream _of(model_name.c_str());
    Json::StyledWriter writer;
    Json::Value net_root;
    net_root["net_name"] = net_name;
    Json::Value layers_root;
    for (int i = 0; i < layers.size(); ++i) {
        Json::Value layer_root;
        layers[i]->SaveModel(layer_root);
        layers_root.append(layer_root);
    }
    net_root["layers"] = layers_root;
    string json_file = writer.write(net_root);
    _of << json_file;
    _of.close();
  }

  virtual void LoadModel(Json::Value &layer_root) {
    
  }

  virtual void Start() = 0;
  
 // protected:
 // orc: add a friend? 
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
  // nets output nodes
  map<string, vector<string> > out_nodes;
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
  // Config
  Json::Value root;
  // need reshape
  bool need_reshape;
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
