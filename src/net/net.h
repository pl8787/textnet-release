#ifndef TEXTNET_NET_NET_H_
#define TEXTNET_NET_NET_H_

#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"

#include "../layer/layer.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../io/json/json.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of net defintiion */
namespace net {
  
using namespace std;
using namespace layer;
using namespace mshadow;

template<typename xpu>
class Net {
 public:
  Net(void) {

  }
  
  virtual ~Net(void) {
    
  }
  
  virtual void InitNet(string config_file) {
    
  }
  
  virtual void InitNet(Json::Value &net_root) {
    net_name = net_root["net_name"].asString();
    utils::Printf("Initializing Net: %s\n", net_name.c_str());
    
    // ******** Create Layers ********
    utils::Printf("Creating Layers.\n");
    Json::Value layers_root = net_root["layers"];
    
    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value layer_root = layers_root[i];
      // Get Layer type
      LayerType layer_type = layer_root["layer_type"].asInt();
      // Reset layer index
      layer_root["layer_idx"] = i;
      
      if (layer_root["phrase"] == "TRAIN") {
        train_net.push_back(CreateLayer<xpu>(layer_type));
      } else if (layer_root["phrase"] == "TEST") {
        test_net.push_back(CreateLayer<xpu>(layer_type));
      } else {
        train_net.push_back(CreateLayer<xpu>(layer_type));
        test_net.push_back(CreateLayer<xpu>(layer_type));
      }
      
      utils::Printf("\t Layer Type: %d\n", layer_type);
    }
    
    // ******** Create Nodes ********
    utils::Printf("Creating Nodes.\n");
    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value layer_root = layers_root[i];
      Json::Value bottoms_root = layer_root["bottom_nodes"];
      Json::Value tops_root = layer_root["top_nodes"];
      for (int j = 0; j < bottoms_root.size(); ++j) {
        string node_name = bottoms_root[j].asString();
        if (!nodes.count(node_name)) {
          nodes[node_name] = new Node<xpu>();
          utils::Printf("\t Node Name: %s\n", node_name.c_str());
        }
      }
      for (int j = 0; j < tops_root.size(); ++j) {
        string node_name = tops_root[j].asString();
        if (!nodes.count(node_name)) {
          nodes[node_name] = new Node<xpu>();
          utils::Printf("\t Node Name: %s\n", node_name.c_str());
        }
      }
    }
    
    // ******** Connect layers ********
    utils::Printf("Connecting Layers.\n");
    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value layer_root = layers_root[i];
      Json::Value bottoms_root = layer_root["bottom_nodes"];
      Json::Value tops_root = layer_root["top_nodes"];
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
  
  virtual void Setup() {

  }
  
  virtual void Reshape() {
    
  }

  virtual void Forward() {
    
  }

  virtual void Backprop() {
    
  }

  virtual void SaveModel(Json::Value &layer_root) {
    
  }

  virtual void LoadModel(Json::Value &layer_root) {
    
  }
  
 
 protected:
  // Net name 
  string net_name;
  // Random Machine for all
  mshadow::Random<xpu> rnd;
  // Net for train model
  vector<Layer<xpu>*> train_net;
  // Net for test model
  vector<Layer<xpu>*> test_net; 
  // Nodes to store datum between layers
  map<string, Node<xpu>*> nodes;
  // bottom vectors
  vector<vector<Node<xpu>*> > bottom_vecs;
  // top vectors
  vector<vector<Node<xpu>*> > top_vecs;
  
};

}  // namespace net
}  // namespace textnet
#endif  // TEXTNET_NET_NET_H_
