#ifndef TEXTNET_NET_NET_H_
#define TEXTNET_NET_NET_H_

// #define DEBUG 1

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
	InitSettingEngine();
  }

  
  virtual ~Net(void) {
    
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
    SettingV::SettingIntMap["Both"] = kBoth;
   
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
    InitNet(root);
  }
  
  virtual void InitNet(Json::Value &net_root) {
	utils::Printf("[Process] Initial Network.\n");

    root = net_root;
    ExpandConfig(root);
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
      
      Layer<xpu> * new_layer = CreateLayer<xpu>(layer_type);
      string layer_name = layer_root["layer_name"].asString();
      new_layer->layer_name = layer_name;

      // Reset layer index
      layer_root["layer_idx"] = i;
      new_layer->layer_idx = i;

      if (!layer_root["setting"]["phrase_type"]) 
        layer_root["setting"]["phrase_type"] = kBoth; 

	  // Get Phrase Type
	  PhraseType layer_phrase = 0;
	  if (layer_root["setting"]["phrase_type"].type() == Json::intValue) {
		layer_phrase = layer_root["setting"]["phrase_type"].asInt();
	  } else if (layer_root["setting"]["phrase_type"].type() == Json::stringValue) {
        layer_phrase = SettingV::SettingIntMap[layer_root["setting"]["phrase_type"].asString()];
	  } else {
		utils::Error("[Error] phrase type error.\n");
	  }
    
      if (layer_phrase == kTrain) {
        train_net.push_back(new_layer);
      } else if (layer_phrase == kTest) {
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

    //      name2layer[target_layer_name]->ShareParameter(target_param_id,
	//			name2layer[source_layer_name]->GetParams()[source_param_id]);

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
  
  virtual void SetupReshape() {
    utils::Printf("[Process] Setup Layers.\n");
    Json::Value &layers_root = root["layers"];
    for (int i = 0; i < test_net.size(); ++i) {
      int layer_idx = test_net[i]->layer_idx;
	  utils::Printf("[layer] set layer %s\n", test_net[i]->layer_name.c_str());
      test_net[i]->SetupLayer(layers_root[layer_idx], 
          bottom_vecs[layer_idx], top_vecs[layer_idx], prnd);
      test_net[i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
    }
    for (int i = 0; i < train_net.size(); ++i) {
      int layer_idx = train_net[i]->layer_idx;
	  utils::Printf("[layer] set layer %s\n", train_net[i]->layer_name.c_str());
      train_net[i]->SetupLayer(layers_root[layer_idx], 
          bottom_vecs[layer_idx], top_vecs[layer_idx], prnd);
      train_net[i]->Reshape(bottom_vecs[layer_idx], top_vecs[layer_idx]);
    }
  }

  virtual void Reshape() {
	utils::Printf("[Process] Reshape network.\n");
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
  
  virtual void SetPhrase(vector<Layer<xpu>*> &net, PhraseType phrase) {
	utils::Printf("[Process] Set Phrase to %d.\n", phrase);
	phrase_type = phrase;
	for (int i = 0; i < net.size(); ++i) {
	  net[i]->SetPhrase(phrase);
	}
    if (need_reshape) Reshape();
  }

  virtual void Forward() {
    if (phrase_type == kTrain) {
      for (int i = 0; i < train_net.size(); ++i) {
        int layer_idx = train_net[i]->layer_idx;
        train_net[i]->Forward(bottom_vecs[layer_idx], top_vecs[layer_idx]);
#if DEBUG
		cout << "Feed " ;
		for (int j = 0; j < bottom_vecs[layer_idx].size(); ++j)
			cout << bottom_vecs[layer_idx][j]->node_name << ", ";
		cout << " and ";
		for (int j = 0; j < top_vecs[layer_idx].size(); ++j)
			cout << top_vecs[layer_idx][j]->node_name << ", ";
		cout << " to " << train_net[i]->layer_name << endl;
#endif
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
      for (int i = train_net.size()-1; i >= 0; --i) {
          int layer_idx = train_net[i]->layer_idx;
          train_net[i]->ClearDiff(bottom_vecs[layer_idx], top_vecs[layer_idx]);
      }
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
#if DEBUG
		  cout << "Update param in layer " << i << " params " << j << endl;
          cout << "param data" << i << " , " << j << ": " << train_net[i]->GetParams()[j].data[0][0][0][0] 
			   << "\t" << train_net[i]->GetParams()[j].data[0][0][0][1]
			   << endl;
          cout << "param data" << i << " , " << j << ": " << train_net[i]->GetParams()[j].diff[0][0][0][0]
			   << "\t" << train_net[i]->GetParams()[j].diff[0][0][0][1]
			   << endl;
#endif
          train_net[i]->GetParams()[j].Update();
#if DEBUG
          cout << "param data" << i << " , " << j << ": " << train_net[i]->GetParams()[j].data[0][0][0][0]
			   << "\t" << train_net[i]->GetParams()[j].data[0][0][0][1]
			   << endl;
#endif
        }
      }
    }
  }
  
  virtual void Training() {
	need_reshape = false;

    // Prepare
    PropAll();
    SetupReshape();

	SetPhrase(train_net, kTrain);

    for (int iter = 0; iter < max_iters; ++iter) {
      
      // Do job
      Forward();
      Backprop();

#if DEBUG
	  // For debug
	  //for (typename map<string, Node<xpu>*>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
	  for (int k = 0; k < node_list.size(); ++k) {
		string name = node_list[k]->node_name; //it->first;
		cout << "Snapshot [" << name << "]" << endl;
		cout << "data : ";
		for (int i = 0; i < 5; ++i) {
			//cout << nodes[name]->data[0][0][0][i] << "\t";
			cout << node_list[k]->data[0][0][0][i] << "\t";
		}
		cout << endl;
		cout << "diff : ";
		for (int i = 0; i < 5; ++i) {
			//cout << nodes[name]->diff[0][0][0][i] << "\t";
			cout << node_list[k]->diff[0][0][0][i] << "\t";
		}
		cout << endl;
	  }
#endif

      Update();
      
      // Output 
      if (display_interval > 0 && iter % display_interval == 0) {
        for (int i = 0; i < train_out.size(); ++i) {
          cout << "[Train]\tIter\t" << iter 
               << ":\tOut[" << train_out[i] << "] =\t" 
               << nodes[train_out[i]]->data_d1()[0] << endl; 
        }
      }
      
      if (test_interval > 0 && iter % test_interval == 0) {
        TestOne(iter);
      }
    }
  }

  virtual void TestOne(int iter) {
	  SetPhrase(test_net, kTest);

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
             << ":\tOut[" << test_out[i] << "] =\t" 
             << test_loss[i] << endl; 
      }
        
	  SetPhrase(train_net, kTrain);
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
  
  void PrintTensor(const char * name, Tensor<cpu, 1> x) {
    Shape<1> s = x.shape_;
    cout << name << " shape " << s[0] << endl;
    for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
      cout << x[d1] << " ";
    }
    cout << endl;
  }

  void PrintTensor(const char * name, Tensor<cpu, 2> x) {
    Shape<2> s = x.shape_;
    cout << name << " shape " << s[0] << "x" << s[1] << endl;
    for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
      for (unsigned int d2 = 0; d2 < s[1]; ++d2) {
        cout << x[d1][d2] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  void PrintTensor(const char * name, Tensor<cpu, 3> x) {
    Shape<3> s = x.shape_;
    cout << name << " shape " << s[0] << "x" << s[1] << "x" << s[2] << endl;
    for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
        for (unsigned int d2 = 0; d2 < s[1]; ++d2) {
            for (unsigned int d3 = 0; d3 < s[2]; ++d3) {
                    cout << x[d1][d2][d3] << " ";
            }
            cout << ";";
        }
        cout << endl;
    }
  }

  void PrintTensor(const char * name, Tensor<cpu, 4> x) {
    Shape<4> s = x.shape_;
    cout << name << " shape " << s[0] << "x" << s[1] << "x" << s[2] << "x" << s[3] << endl;
    for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
        for (unsigned int d2 = 0; d2 < s[1]; ++d2) {
            for (unsigned int d3 = 0; d3 < s[2]; ++d3) {
                for (unsigned int d4 = 0; d4 < s[3]; ++d4) {
                    cout << x[d1][d2][d3][d4] << " ";
                }
                cout << "|";
            }
            cout << ";";
        }
        cout << endl;
    }
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
  // node list
  vector<Node<xpu>*> node_list;

};

}  // namespace net
}  // namespace textnet
#endif  // TEXTNET_NET_NET_H_
