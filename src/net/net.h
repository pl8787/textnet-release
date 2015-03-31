#ifndef TEXTNET_NET_NET_H_
#define TEXTNET_NET_NET_H_

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
  Net(Random<xpu>* prnd_) {
    prnd = prnd_;
  }

  
  virtual ~Net(void) {
    
  }
  
  virtual void InitNet(string config_file) {
    ifstream _if(config_file.c_str());
    _if >> root;
    InitNet(root);
  }
  
  virtual void InitNet(Json::Value &net_root) {
    root = net_root;
    net_name = net_root["net_name"].asString();
    max_iters = net_root["max_iters"].asInt();
    max_test_iters = net_root["max_test_iters"].asInt();
    display_interval = net_root["display_interval"].asInt();
    test_interval = net_root["test_interval"].asInt();
    for (int i = 0; i < net_root["train_out"].size(); ++i) {
      train_out.push_back(net_root["train_out"][i].asString());
    }
    for (int i = 0; i < net_root["test_out"].size(); ++i) {
      test_out.push_back(net_root["test_out"][i].asString());
    }
    
    utils::Printf("Initializing Net: %s\n", net_name.c_str());
    
    // ******** Create Layers ********
    utils::Printf("Creating Layers.\n");
    Json::Value &layers_root = net_root["layers"];
    
    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value &layer_root = layers_root[i];
      // Get Layer type
      LayerType layer_type = layer_root["layer_type"].asInt();
      
      Layer<xpu> * new_layer = CreateLayer<xpu>(layer_type);
      string layer_name = layer_root["layer_name"].asString();
      
      // Reset layer index
      layer_root["layer_idx"] = i;
    new_layer->layer_idx = i;

    if (!layer_root["setting"]["phrase_type"]) 
     layer_root["setting"]["phrase_type"] = 2; 
    
      if (layer_root["setting"]["phrase_type"].asInt() == 0) {
        train_net.push_back(new_layer);
      } else if (layer_root["setting"]["phrase_type"].asInt() == 1) {
        test_net.push_back(new_layer);
      } else {
        train_net.push_back(new_layer);
        test_net.push_back(new_layer);
      }  

      name2layer[layer_name] = new_layer;
      layers.push_back(new_layer);
      
      utils::Printf("\t Layer Type: %d\t Layer Name: %s\n", layer_type, layer_name.c_str());
    }
    
    utils::Printf("Train Layer Deep: %d\n", train_net.size());
    utils::Printf("Test Layer Deep: %d\n", test_net.size());
    
    // ******** Create Nodes ********
    utils::Printf("Creating Nodes.\n");
    for (int i = 0; i < layers_root.size(); ++i) {
      Json::Value &layer_root = layers_root[i];
      Json::Value &bottoms_root = layer_root["bottom_nodes"];
      Json::Value &tops_root = layer_root["top_nodes"];
      for (int j = 0; j < bottoms_root.size(); ++j) {
        string node_name = bottoms_root[j].asString();
        if (!nodes.count(node_name)) {
          nodes[node_name] = new Node<xpu>();
      nodes[node_name]->node_name = node_name;
          utils::Printf("\t Node Name: %s\n", node_name.c_str());
        }
      }
      for (int j = 0; j < tops_root.size(); ++j) {
        string node_name = tops_root[j].asString();
        if (!nodes.count(node_name)) {
          nodes[node_name] = new Node<xpu>();
      nodes[node_name]->node_name = node_name;
          utils::Printf("\t Node Name: %s\n", node_name.c_str());
        }
      }
    }

    utils::Printf("Nodes count: %d\n", nodes.size());
    
    // ******** Connect layers ********
    utils::Printf("Connecting Layers.\n");

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

  virtual void PropAll() {
    utils::Printf("PropAll Layers.\n");
    for (int i = 0; i < layers.size(); ++i) {
      layers[i]->PropAll();
    }
  }
  
  virtual void SetupReshape() {
    utils::Printf("Setup Layers.\n");
    Json::Value &layers_root = root["layers"];
    for (int i = 0; i < test_net.size(); ++i) {
      int layer_idx = test_net[i]->layer_idx;
      test_net[i]->SetupLayer(layers_root[layer_idx], 
      bottom_vecs[layer_idx], top_vecs[layer_idx], prnd);
      test_net[i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
    }
    for (int i = 0; i < train_net.size(); ++i) {
      int layer_idx = train_net[i]->layer_idx;
      train_net[i]->SetupLayer(layers_root[layer_idx], 
      bottom_vecs[layer_idx], top_vecs[layer_idx], prnd);
      train_net[i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
    }
  }

  virtual void Reshape() {
    if (phrase_type == kTrain) {
      for (int i = 0; i < train_net.size(); ++i) {
        int layer_idx = train_net[i]->layer_idx;
        train_net[i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
      }
    } else if (phrase_type == kTest) {
      for (int i = 0; i < test_net.size(); ++i) {
        int layer_idx = test_net[i]->layer_idx;
        test_net[i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
      }
    }
  }
  
  virtual void Forward() {
    if (phrase_type == kTrain) {
      for (int i = 0; i < train_net.size(); ++i) {
        int layer_idx = train_net[i]->layer_idx;
        train_net[i]->Forward(bottom_vecs[layer_idx], top_vecs[layer_idx]);
      }
    } else if (phrase_type == kTest) {
      for (int i = 0; i < test_net.size(); ++i) {
        int layer_idx = test_net[i]->layer_idx;
        test_net[i]->Forward(bottom_vecs[layer_idx], top_vecs[layer_idx]);
      }
    }
  }

  virtual void Backprop() {
    utils::Check(phrase_type == kTrain, 
                  "Only call in Train Phrase.");
    if (phrase_type == kTrain) {
      for (int i = train_net.size()-1; i>=0; --i) {
        int layer_idx = train_net[i]->layer_idx;
        train_net[i]->Backprop(bottom_vecs[layer_idx], top_vecs[layer_idx]);
      }
    }
  }
  
  virtual void Update() {
    utils::Check(phrase_type == kTrain, 
                  "Only call in Train Phrase.");
    if (phrase_type == kTrain) {
      for (int i = 0; i < train_net.size(); ++i) {
        for (int j = 0; j < train_net[i]->ParamNodeNum(); ++j) {
          train_net[i]->GetParams()[j].Update();
        }
      }
    }
  }
  
  virtual void Training() {
    phrase_type = kTrain;
    
    // Prepare
    PropAll();
    SetupReshape();

    for (int iter = 0; iter < max_iters; ++iter) {
      phrase_type = kTrain;
      
      // Do job
      Forward();
      Backprop();
      Update();
      
      // Output 
      if (display_interval > 0 && iter % display_interval == 0) {
        for (int i = 0; i < train_out.size(); ++i) {
          cout << "[Train]\tIter\t" << iter 
               << ":\terror[" << i << "] =\t" 
               << nodes[train_out[i]]->data_d1()[0] << endl; 
        }
      }
      
      if (test_interval > 0 && iter % test_interval == 0) {
        if (need_reshape) Reshape();
        TestOne(iter);
        if (need_reshape) Reshape();
      }
  }
  }

  virtual void TestOne(int iter) {
      phrase_type = kTest;

      // Initial test loss
      vector<float> test_loss;
      for (int i = 0; i < test_out.size(); ++i) {
        test_loss.push_back(0.0f);
      }
      
      for (int test_iter = 0; test_iter < max_test_iters; ++test_iter) {
        Forward();
        for (int i = 0; i < test_out.size(); ++i) {
          test_loss[i] += nodes[test_out[i]]->data_d1()[0];
        }
      }
      
      for (int i = 0; i < test_out.size(); ++i) {
        test_loss[i] /= max_test_iters;
      }
      
      // Output
      for (int i = 0; i < test_out.size(); ++i) {
        cout << "[Test]\tIter\t" << iter 
             << ":\terror[" << i << "] =\t" 
             << test_loss[i] << endl; 
      }
        
      phrase_type = kTrain;
  }
  virtual void SaveModel(string model_name) {
    ofstream _of(model_name.c_str());
    Json::StyledWriter writer;
    Json::Value net_root;
    net_root["net_name"] = net_name;
    net_root["max_iters"] = max_iters;
    net_root["max_test_iters"] = max_test_iters;
    net_root["display_interval"] = display_interval;
    net_root["test_interval"] = test_interval;
    for (int i = 0; i < net_root["train_out"].size(); ++i) {
      net_root["train_out"].append(train_out[i]);
    }
    for (int i = 0; i < net_root["test_out"].size(); ++i) {
      net_root["test_out"].append(test_out[i]);
    }
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
  
 
 protected:
  // Net name 
  string net_name;
  // Random Machine for all
  mshadow::Random<xpu>* prnd;
  // Net for train model
  vector<Layer<xpu>*> train_net;
  // Net for test model
  vector<Layer<xpu>*> test_net;
  // All layers
  vector<Layer<xpu>*> layers;
  map<string, Layer<xpu>*> name2layer;
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
  // max iterations
  int max_iters;
  // max test iterations
  int max_test_iters;
  // train display interval
  int display_interval;
  // test interval
  int test_interval;
  // train output nodes
  vector<string> train_out;
  // test output nodes
  vector<string> test_out;
  // need reshape
  bool need_reshape;
  
};

}  // namespace net
}  // namespace textnet
#endif  // TEXTNET_NET_NET_H_
