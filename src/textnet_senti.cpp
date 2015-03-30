#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <cstring>
#include <vector>
#include <map>
#include <climits>

#include "./layer/layer.h"
#include "./io/json/json.h"
#include "global.h"

using namespace std;
using namespace textnet;
using namespace textnet::layer;
using namespace mshadow;

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
    cout << endl;
}

int main(int argc, char *argv[]) {
  mshadow::Random<cpu> rnd(37);
  vector<Layer<cpu>*> senti_net;
  vector<Layer<cpu>*> senti_net_test;
  vector<vector<Node<cpu>*> > bottom_vecs;
  vector<vector<Node<cpu>*> > top_vecs;
  vector<Node<cpu>*> nodes;

  int lstm_hidden_dim = 10;
  // int lstm_input_dim = 50;
  int word_rep_dim = 50;
  int max_doc_len = 100;
  int min_doc_len = 4;
  int batch_size = 1;
  
  senti_net.push_back(CreateLayer<cpu>(kTextData));
  senti_net.push_back(CreateLayer<cpu>(kEmbedding));
  senti_net.push_back(CreateLayer<cpu>(kLstm));
  senti_net.push_back(CreateLayer<cpu>(kMaxPooling));
  senti_net.push_back(CreateLayer<cpu>(kSoftmax));
  
  senti_net_test = senti_net;
  senti_net_test[0] = CreateLayer<cpu>(kTextData);
  senti_net_test.push_back(CreateLayer<cpu>(kAccuracy));
  
  for (index_t i = 0; i < senti_net.size(); ++i) {
    vector<Node<cpu>*> bottoms;
    vector<Node<cpu>*> tops;
    bottom_vecs.push_back(bottoms);
    top_vecs.push_back(tops);
    for (index_t j = 0; j < senti_net[i]->TopNodeNum(); ++j) {
      Node<cpu>* node = new Node<cpu>();
      nodes.push_back(node);
    }
  }

  // {
  //   vector<Node<cpu>*> bottoms;
  //   vector<Node<cpu>*> tops;
  //   bottom_vecs.push_back(bottoms);
  //   top_vecs.push_back(tops);
  //   nodes.push_back(new Node<cpu>());
  // }

  // Name the layers
  senti_net[0]->layer_name = "textdata";
  senti_net[1]->layer_name = "embedding";
  senti_net[2]->layer_name = "lstm";
  senti_net[3]->layer_name = "pooling";
  senti_net[4]->layer_name = "softmax";
  
  senti_net_test[0]->layer_name = "textdata_test";
  senti_net_test[0]->layer_idx = 0;
  
  for (index_t i = 0; i < senti_net.size(); ++i) {
    senti_net[i]->layer_idx = i;
  }
  
  cout << "Nodes size:" << nodes.size() << endl;
  // Name the nodes
  nodes[0]->node_name = "text_data";
  nodes[1]->node_name = "label";
  nodes[2]->node_name = "embedding";
  nodes[3]->node_name = "lstm";
  nodes[4]->node_name = "pool";
  nodes[5]->node_name = "loss";
  
  for (index_t i = 0; i < nodes.size(); ++i) {
    nodes[i]->node_idx = i;
  }

  cout << "Total node count: " << nodes.size() << endl;
  
  // Manual connect layers
  top_vecs[0].push_back(nodes[0]);
  top_vecs[0].push_back(nodes[1]);
  bottom_vecs[1].push_back(nodes[0]);
  top_vecs[1].push_back(nodes[2]);
  bottom_vecs[2].push_back(nodes[2]);
  top_vecs[2].push_back(nodes[3]);
  bottom_vecs[3].push_back(nodes[3]);
  top_vecs[3].push_back(nodes[4]);
  bottom_vecs[4].push_back(nodes[4]);
  bottom_vecs[4].push_back(nodes[1]);
  top_vecs[4].push_back(nodes[5]);
  // Manual connect layers node name
  // kTextData
  // senti_net[0]->top_nodes.push_back(nodes[0]->node_name);
  // senti_net[0]->top_nodes.push_back(nodes[1]->node_name);
  // kEmbedding
  // senti_net[1]->bottom_nodes.push_back(nodes[0]->node_name);
  // senti_net[1]->top_nodes.push_back(nodes[2]->node_name);
  // kMaxPooling
  // senti_net[6]->bottom_nodes.push_back(nodes[7]->node_name);
  // senti_net[6]->top_nodes.push_back(nodes[8]->node_name);
  
  // kTextData Test
  // senti_net_test[0]->top_nodes.push_back(nodes[0]->node_name);
  // senti_net_test[0]->top_nodes.push_back(nodes[1]->node_name);
  // senti_net_test[19]->bottom_nodes.push_back(nodes[19]->node_name);
  // senti_net_test[19]->bottom_nodes.push_back(nodes[1]->node_name);
  // senti_net_test[19]->top_nodes.push_back(nodes[21]->node_name);
  
  float base_lr = 0.01;
  float ADA_GRAD_EPS = 0.01;
  float decay = 0.01;
  // Fill Settings vector
  vector<map<string, SettingV> > setting_vec;
  // kTextData
  {
    map<string, SettingV> setting;
    // orc
    // setting["data_file"] = SettingV("/home/pangliang/matching/data/msr_paraphrase_train_wid_dup.txt");
    setting["data_file"] = SettingV("/home/wsx/dl.shengxian/data/mr/all.lstm");
    setting["batch_size"] = SettingV(batch_size);
    setting["max_doc_len"] = SettingV(max_doc_len);
    setting["min_doc_len"] = SettingV(1);
    setting_vec.push_back(setting);
  }
  // kEmbedding
  {
    map<string, SettingV> setting;
    // orc
    // setting["embedding_file"] = SettingV("/home/pangliang/matching/data/wikicorp_50_msr.txt");
    setting["embedding_file"] = SettingV("/home/wsx/repo.other/textnet/data/wikicorp_50_msr.txt");
    setting["word_count"] = SettingV(14727);
    setting["feat_size"] = SettingV(word_rep_dim);
      
    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kUniform);
      w_setting["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_setting);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["eps"] = SettingV(ADA_GRAD_EPS);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);  
    setting["w_updater"] = SettingV(&w_updater);
    setting_vec.push_back(setting);
  }
  // kLstm
  {
    map<string, SettingV> setting;
    setting["d_input"] = SettingV(word_rep_dim);
    setting["d_mem"] = SettingV(lstm_hidden_dim);
    setting["no_bias"] = SettingV(true);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_filler);
    setting["u_filler"] = SettingV(&w_filler);
    // map<string, SettingV> &u_filler = *(new map<string, SettingV>());
    //   u_filler["init_type"] = SettingV(initializer::kUniform);
    //   u_filler["range"] = SettingV(0.01f);
    // setting["u_filler"] = SettingV(&u_filler);
    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["eps"] = SettingV(ADA_GRAD_EPS);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);  
    setting["w_updater"] = SettingV(&w_updater);
    setting["u_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
    // map<string, SettingV> &u_updater = *(new map<string, SettingV>());
    //   u_updater["updater_type"] = SettingV(updater::kAdagrad);
    //   u_updater["momentum"] = SettingV(0.0f);
    //   u_updater["eps"] = SettingV(ADA_GRAD_EPS);
    //   u_updater["lr"] = SettingV(base_lr);
    //   u_updater["decay"] = SettingV(decay);  
    // setting["u_updater"] = SettingV(&u_updater);
    // map<string, SettingV> &b_updater = *(new map<string, SettingV>());
    //   b_updater["updater_type"] = SettingV(updater::kAdagrad);
    //   b_updater["momentum"] = SettingV(0.0f);
    //   b_updater["eps"] = SettingV(ADA_GRAD_EPS);
    //   b_updater["lr"] = SettingV(base_lr);
    //   b_updater["decay"] = SettingV(decay);  
    // setting["b_updater"] = SettingV(&b_updater);
    
    setting_vec.push_back(setting);
  }

  // kMaxPooling
  {
    map<string, SettingV> setting;
    setting["kernel_x"] = SettingV(1);
    setting["kernel_y"] = SettingV(max_doc_len);
    setting["stride"] = SettingV(1);
    setting_vec.push_back(setting);
  }
  // kCrossEntopyLoss
  {
    map<string, SettingV> setting;
    setting["delta"] = SettingV(1.0f);
    setting_vec.push_back(setting);
  }
  
  // kTextData
  {
    map<string, SettingV> setting;
    // orc
    // setting["data_file"] = SettingV("/home/pangliang/matching/data/msr_paraphrase_test_wid.txt");
    setting["data_file"] = SettingV("/home/wsx/repo.other/textnet/data/msr_paraphrase_test_wid.txt");
    setting["batch_size"] = SettingV(batch_size);//1725);
    setting["max_doc_len"] = SettingV(max_doc_len);
    setting["min_doc_len"] = SettingV(5);
    setting_vec.push_back(setting);
  }
  // kAccuracy
  // {
  //   map<string, SettingV> setting;
  //   setting["topk"] = SettingV(1);
  //   setting_vec.push_back(setting);
  // }
  
  cout << "Setting Vector Filled." << endl;

  // Set up Layers
  for (index_t i = 0; i < senti_net.size(); ++i) {
    cout << "Begin set up layer " << i << endl;
    senti_net[i]->PropAll();
    // cout << "\tPropAll" << endl;
    senti_net[i]->SetupLayer(setting_vec[i], bottom_vecs[i], top_vecs[i], &rnd);
    // cout << "\tSetup Layer" << endl;
    senti_net[i]->Reshape(bottom_vecs[i], top_vecs[i]);
    // cout << "\tReshape" << endl;
  }
  // senti_net_test[0]->PropAll();
  // senti_net_test[0]->SetupLayer(setting_vec[setting_vec.size()-2], bottom_vecs[0], top_vecs[0], &rnd);
  // senti_net_test[0]->Reshape(bottom_vecs[0], top_vecs[0]);
  // 
  // senti_net_test[19]->PropAll();
  // senti_net_test[19]->SetupLayer(setting_vec[setting_vec.size()-1], bottom_vecs[19], top_vecs[19], &rnd);
  // senti_net_test[19]->Reshape(bottom_vecs[19], top_vecs[19]);
  
  // Save Initial Model
  // {
  //   ofstream _of("model/matching0.model");
  //   Json::StyledWriter writer;
  //   Json::Value net_root;
  //   net_root["net_name"] = "senti_net";
  //   Json::Value layers_root;
  //   for (index_t i = 0; i < senti_net.size(); ++i) {
  //     Json::Value layer_root;
  //     senti_net[i]->SaveModel(layer_root);
  //     layers_root.append(layer_root);
  //   }
  //   net_root["layers"] = layers_root;
  //   string json_file = writer.write(net_root);
  //   _of << json_file;
  //   _of.close();
  // }

  // Begin Training 
  int max_iters = 4000;
  for (int iter = 0; iter < max_iters; ++iter) {
    cout << "Begin iter " << iter << endl;
    for (index_t i = 0; i < senti_net.size(); ++i) {
      //cout << "Forward layer " << i << endl;
      senti_net[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }
    
#if 0
    for (index_t i = 0; i < nodes.size(); ++i) {
      cout << "# Data " << nodes[i]->node_name << " : ";
      for (index_t j = 0; j < 5; ++j) {
          cout << nodes[i]->data[0][0][0][j] << "\t";
      }
      cout << endl;
      cout << "# Diff " << nodes[i]->node_name << " : ";
      for (index_t j = 0; j < 5; ++j) {
          cout << nodes[i]->diff[0][0][0][j] << "\t";
      }
      cout << endl;
    }
#endif

    for (index_t i = senti_net.size()-1; i >= 0; --i) {
      //cout << "Backprop layer " << i << endl;
      senti_net[i]->Backprop(bottom_vecs[i], top_vecs[i]);
    }
    for (index_t i = 0; i < senti_net.size(); ++i) {
      for (index_t j = 0; j < senti_net[i]->ParamNodeNum(); ++j) {
        // cout << "Update param in layer " << i << " params " << j << endl;
        // cout << "param data" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].data[0][0][0][0] << endl;
        // cout << "param diff" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].diff[0][0][0][0] << endl;
        senti_net[i]->GetParams()[j].Update();
        // cout << "param data" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].data[0][0][0][0] << endl<<endl;
      }
    }
    
    // Output informations
    cout << "###### Iter " << iter << ": error =\t" << nodes[20]->data_d1()[0] << endl;
    
    if (iter % 100 == 0) {
      float loss = 0.0;
      float acc = 0.0;
      int max_test_iter = 34;
      for (index_t test_iter = 0; test_iter < max_test_iter; ++test_iter) {
        for (index_t i = 0; i < senti_net_test.size(); ++i) {
          senti_net_test[i]->Forward(bottom_vecs[i], top_vecs[i]);
        }
        loss += nodes[20]->data_d1()[0];
        acc += nodes[21]->data_d1()[0];
      }
      loss /= max_test_iter;
      acc /= max_test_iter;
      cout << endl;
      cout << "****** Test loss =\t" << loss << endl;
      cout << "****** Test accuracy =\t" << acc << endl;
    }
  }
  
  // Save Initial Model
  // {
  // ofstream _of("model/matching1.model");
  // Json::StyledWriter writer;
  // Json::Value net_root;
  // net_root["net_name"] = "senti_net";
  // Json::Value layers_root;
  // for (index_t i = 0; i < senti_net.size(); ++i) {
  //     Json::Value layer_root;
  //     senti_net[i]->SaveModel(layer_root);
  //     layers_root.append(layer_root);
  // }
  // net_root["layers"] = layers_root;
  // string json_file = writer.write(net_root);
  // _of << json_file;
  // _of.close();
  // }
  return 0;
}

