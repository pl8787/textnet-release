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
#include <cassert>
#include "./checker/checker.h"


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

struct EvalRet {
    float acc, loss;
    EvalRet(): acc(0.), loss(0.) {}
    void clear() { acc = 0.; loss = 0.; }
};

void eval(vector<Layer<cpu> *> senti_net, 
          vector<vector<Node<cpu> *> > &bottoms, 
          vector<vector<Node<cpu> *> > &tops, 
          int nBatch,
          EvalRet &ret) {
  ret.clear();
  for (int i = 0; i < nBatch; ++i) {
      for (int j = 0; j < senti_net.size(); ++j) {
          senti_net[j]->SetPhrase(kTest);
          senti_net[j]->Forward(bottoms[j], tops[j]);
      }
      ret.loss += tops[tops.size()-2][0]->data_d1()[0];
      ret.acc  += tops[tops.size()-1][0]->data_d1()[0];
  }
  ret.loss /= float(nBatch);
  ret.acc /= float(nBatch);
}
    

int main(int argc, char *argv[]) {
  mshadow::Random<cpu> rnd(37);
  vector<Layer<cpu>*> senti_net, senti_valid, senti_test;

  int lstm_hidden_dim = 30;
  int word_rep_dim = 300;
  int max_doc_len = 100;
  int min_doc_len = 1;
  int batch_size = 50;
  int vocab_size = 200000;
  int num_class = 2;
  int ADA_GRAD_MAX_ITER = 1000000;
  float base_lr = 0.1;
  float ADA_GRAD_EPS = 0.01;
  float decay = 0.00;
  float l2 = 0.f;
  float pad_value = (float)(NAN);

  // string train_data_file = "/home/wsx/dl.shengxian/data/simulation/lstm.train";
  // string valid_data_file = "/home/wsx/dl.shengxian/data/simulation/lstm.test";
  // string test_data_file = "/home/wsx/dl.shengxian/data/simulation/lstm.test";
  // string embedding_file = ""; 
  // int nTrain = 6, nValid = 7, nTest = 7;
  
  string train_data_file = "/home/wsx/dl.shengxian/data/mr/lstm.train";
  string valid_data_file = "/home/wsx/dl.shengxian/data/mr/lstm.dev";
  string test_data_file = "/home/wsx/dl.shengxian/data/mr/lstm.test";
  string embedding_file = "/home/wsx/dl.shengxian/data/mr/word_rep_w2v.plpl"; 
  int nTrain = 8528, nValid = 1067, nTest = 1067;

  if (argc == 2) {
    base_lr = atof(argv[1]);
  }
  map<string, SettingV> &g_updater = *(new map<string, SettingV>());
  // g_updater["updater_type"] = SettingV(updater::kAdaDelta);
  g_updater["updater_type"] = SettingV(updater::kAdaDelta);
  // g_updater["batch_size"] = SettingV(batch_size);
  g_updater["l2"] = SettingV(l2);
  g_updater["batch_size"] = SettingV(batch_size);
  // g_updater["max_iter"] = SettingV(ADA_GRAD_MAX_ITER);
  // g_updater["eps"] = SettingV(ADA_GRAD_EPS);
  // g_updater["lr"] = SettingV(base_lr);
  // g_updater["decay"] = SettingV(decay);  
  // g_updater["momentum"] = SettingV(0.0f);


  
  senti_net.push_back(CreateLayer<cpu>(kSequenceClassificationData));
  senti_net[senti_net.size()-1]->layer_name = "text";
  senti_net.push_back(CreateLayer<cpu>(kEmbedding));
  senti_net[senti_net.size()-1]->layer_name = "embedding";
  senti_net.push_back(CreateLayer<cpu>(kLstm));
  senti_net[senti_net.size()-1]->layer_name = "l_lstm";
  senti_net.push_back(CreateLayer<cpu>(kLstm));
  senti_net[senti_net.size()-1]->layer_name = "r_lstm";
  senti_net.push_back(CreateLayer<cpu>(kConvolutionalLstm));
  senti_net[senti_net.size()-1]->layer_name = "conv_lstm";
  senti_net.push_back(CreateLayer<cpu>(kWholePooling));
  senti_net[senti_net.size()-1]->layer_name = "wholepooling";
  senti_net.push_back(CreateLayer<cpu>(kDropout));
  senti_net[senti_net.size()-1]->layer_name = "dropout";
  senti_net.push_back(CreateLayer<cpu>(kFullConnect));
  senti_net[senti_net.size()-1]->layer_name = "fullconnect";
  senti_net.push_back(CreateLayer<cpu>(kSoftmax));
  senti_net[senti_net.size()-1]->layer_name = "softmax";
  senti_net.push_back(CreateLayer<cpu>(kAccuracy));
  senti_net[senti_net.size()-1]->layer_name = "accuracy";

  senti_test = senti_net;
  senti_valid = senti_net;
  senti_test[0] = CreateLayer<cpu>(kSequenceClassificationData);
  senti_valid[0]= CreateLayer<cpu>(kSequenceClassificationData);
  
  vector<vector<Node<cpu>*> > bottom_vecs(senti_net.size(), vector<Node<cpu>*>());
  vector<vector<Node<cpu>*> > top_vecs(senti_net.size(), vector<Node<cpu>*>());
  vector<Node<cpu>*> nodes;

  // Manual connect layers
  int inc_node = 0;
  for (int i = 0; i < senti_net.size(); ++i) { // last layers are softmax layer and accuracy layers
    for (int j = 0; j < senti_net[i]->TopNodeNum(); ++j) {
      Node<cpu>* node = new Node<cpu>();
      stringstream ss;
      ss << nodes.size();
      node->node_name = ss.str();
      nodes.push_back(node);
      top_vecs[i].push_back(node);
    }

    // connect by node name 
    if (i == 0) {
        senti_net[0]->top_nodes.push_back(nodes[0]->node_name);
        senti_net[0]->top_nodes.push_back(nodes[1]->node_name);
        inc_node += senti_net[i]->TopNodeNum();
    } else if (senti_net[i]->layer_name == "r_lstm") {
        senti_net[i]->bottom_nodes.push_back(nodes[inc_node-2]->node_name);
        senti_net[i]->top_nodes.push_back(nodes[inc_node]->node_name);
        inc_node += senti_net[i]->TopNodeNum();
    } else if (senti_net[i]->layer_name == "conv_lstm") {
        senti_net[i]->bottom_nodes.push_back(nodes[inc_node-2]->node_name);
        senti_net[i]->bottom_nodes.push_back(nodes[inc_node-3]->node_name);
        senti_net[i]->bottom_nodes.push_back(nodes[inc_node-1]->node_name);
        senti_net[i]->top_nodes.push_back(nodes[inc_node]->node_name);
        inc_node += senti_net[i]->TopNodeNum();
    } else if (senti_net[i]->layer_name == "accuracy") {
        senti_net[i]->bottom_nodes.push_back(nodes[inc_node-2]->node_name);
        senti_net[i]->bottom_nodes.push_back(nodes[0]->node_name);
        senti_net[i]->top_nodes.push_back(nodes[inc_node]->node_name);
        inc_node += senti_net[i]->TopNodeNum();
    } else if (senti_net[i]->layer_name == "softmax") {
        senti_net[i]->bottom_nodes.push_back(nodes[inc_node-1]->node_name);
        senti_net[i]->bottom_nodes.push_back(nodes[0]->node_name);
        senti_net[i]->top_nodes.push_back(nodes[inc_node]->node_name);
        inc_node += senti_net[i]->TopNodeNum();
    } else {
        senti_net[i]->bottom_nodes.push_back(nodes[inc_node-1]->node_name);
        senti_net[i]->top_nodes.push_back(nodes[inc_node]->node_name);
        inc_node += senti_net[i]->TopNodeNum();
    }
    // connect by node memory
    if (i == 0) {
      continue;
    } else if (senti_net[i]->layer_name == "r_lstm") {
      bottom_vecs[i].push_back(top_vecs[i-2][0]);
    } else if (senti_net[i]->layer_name == "conv_lstm") {
      bottom_vecs[i].push_back(top_vecs[i-2][0]);
      bottom_vecs[i].push_back(top_vecs[i-3][0]);
      bottom_vecs[i].push_back(top_vecs[i-1][0]);
    } else if (senti_net[i]->layer_name == "accuracy") {
      bottom_vecs[i].push_back(top_vecs[i-2][0]);
      bottom_vecs[i].push_back(nodes[0]); // accuracy
    } else if (senti_net[i]->layer_name == "softmax") {
      bottom_vecs[i].push_back(top_vecs[i-1][0]);
      bottom_vecs[i].push_back(nodes[0]); 
    } else if (senti_net[i]->layer_name == "embedding") {
      bottom_vecs[i].push_back(top_vecs[i-1][1]);
    } else {
      bottom_vecs[i].push_back(top_vecs[i-1][0]);
    }
  }
  cout << "Total node count: " << nodes.size() << endl;
  
  // Fill Settings vector
  vector<map<string, SettingV> > setting_vec;
  // kSequenceClassificationData
  {
    map<string, SettingV> setting;
    // orc
    setting["data_file"] = SettingV(train_data_file);
    setting["batch_size"] = SettingV(batch_size);
    setting["max_doc_len"] = SettingV(max_doc_len);
    setting_vec.push_back(setting);
    
    // valid
    // setting["batch_size"] = SettingV(1);
    // setting["data_file"] = SettingV("/home/wsx/dl.shengxian/data/treebank/trees/dev.seq.pad.binary");
    setting["data_file"] = SettingV(valid_data_file);
    senti_valid[0]->PropAll();
    senti_valid[0]->SetupLayer(setting, bottom_vecs[0], top_vecs[0], &rnd);
    senti_valid[0]->Reshape(bottom_vecs[0], top_vecs[0]);
    // test 
    // setting["batch_size"] = SettingV(1);
    setting["data_file"] = SettingV(test_data_file);
    senti_test[0]->PropAll();
    senti_test[0]->SetupLayer(setting, bottom_vecs[0], top_vecs[0], &rnd);
    senti_test[0]->Reshape(bottom_vecs[0], top_vecs[0]);
  }
  // kEmbedding
  {
    map<string, SettingV> setting;
    // orc
    setting["embedding_file"] = SettingV(embedding_file);
    setting["word_count"] = SettingV(vocab_size);
    setting["feat_size"] = SettingV(word_rep_dim);
    setting["pad_value"] = SettingV(pad_value);
      
    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kUniform);
      w_setting["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_setting);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    setting["w_updater"] = SettingV(&w_updater);
    setting_vec.push_back(setting);
  }
  // kLstm
  {
    map<string, SettingV> setting;
    setting["d_input"] = SettingV(word_rep_dim);
    setting["d_mem"] = SettingV(lstm_hidden_dim);
    setting["no_bias"] = SettingV(true);
    setting["pad_value"] = SettingV(pad_value);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.0001f);
    setting["w_filler"] = SettingV(&w_filler);
    setting["u_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    setting["w_updater"] = SettingV(&w_updater);
    setting["u_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
    
    setting_vec.push_back(setting);
  }
  {
    map<string, SettingV> setting;
    setting["d_input"] = SettingV(word_rep_dim);
    setting["d_mem"] = SettingV(lstm_hidden_dim);
    setting["no_bias"] = SettingV(true);
    setting["pad_value"] = SettingV(pad_value);
    setting["reverse"] = SettingV(true);

    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.0001f);
    setting["w_filler"] = SettingV(&w_filler);
    setting["u_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    setting["w_updater"] = SettingV(&w_updater);
    setting["u_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
    
    setting_vec.push_back(setting);
  }
  // kConvLstm
  {
    map<string, SettingV> setting;
    setting["num_hidden"] = SettingV(lstm_hidden_dim);
    setting["no_bias"] = SettingV(true);
    setting["pad_value"] = SettingV(pad_value);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.0001f);
    setting["w_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
    setting_vec.push_back(setting);

  }
  // kWholeMaxPooling
  {
    map<string, SettingV> setting;
    setting["pool_type"] = "max";
    setting_vec.push_back(setting);
  }
  //  kDropout
  {
    map<string, SettingV> setting;
    setting["rate"] = SettingV(0.0f);
    setting_vec.push_back(setting);
  }
  // kFullConnect
  {
    map<string, SettingV> setting;
    setting["num_hidden"] = SettingV(num_class);
    setting["no_bias"] = SettingV(false);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      // w_setting["init_type"] = SettingV(initializer::kZero);
      w_setting["init_type"] = SettingV(initializer::kUniform);
      w_setting["range"] = SettingV(0.0001f);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
    setting_vec.push_back(setting);
  }
  // kSoftmax
  {
    map<string, SettingV> setting;
    setting_vec.push_back(setting);
  }
  // kAccuracy
  {
    map<string, SettingV> setting;
    setting["topk"] = SettingV(1);
    setting_vec.push_back(setting);
  }

  cout << "Setting Vector Filled." << endl;

  // Set up Layers
  for (index_t i = 0; i < senti_net.size(); ++i) {
    cout << "Begin set up layer " << i << endl;
    // senti_net[i]->Require();
    senti_net[i]->PropAll();
    // cout << "\tPropAll" << endl;
    senti_net[i]->SetupLayer(setting_vec[i], bottom_vecs[i], top_vecs[i], &rnd);
    // cout << "\tSetup Layer" << endl;
    senti_net[i]->Reshape(bottom_vecs[i], top_vecs[i]);
    // cout << "\tReshape" << endl;
  }

  // Save Initial Model
  {
  ofstream _of("./senti_lstm.manudataset.model");
  Json::StyledWriter writer;
  Json::Value net_root;
  net_root["net_name"] = "senti_lstm";
  Json::Value layers_root;
  for (index_t i = 0; i < senti_net.size(); ++i) {
      Json::Value layer_root;
      senti_net[i]->SaveModel(layer_root, false);
      layers_root.append(layer_root);
  }
  net_root["layers"] = layers_root;
  string json_file = writer.write(net_root);
  _of << json_file;
  _of.close();
  }

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.1f);
  setting_checker["range_max"] = SettingV(0.1f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, &rnd);
  // check gradient, uncheck last two layers (softmax and accuracy)
  // for (index_t i = 0; i < senti_net.size()-2; ++i) {
  //   cout << "Check gradient of layer " << i << endl;
  //   senti_net[i]->Forward(bottom_vecs[i], top_vecs[i]);
  //   cker->CheckError(senti_net[i], bottom_vecs[i], top_vecs[i]);
  // }

  // Begin Training 
  int max_iters = 3000;
  float maxValidAcc = 0., testAccByValid = 0.;
  for (int iter = 0; iter < max_iters; ++iter) {
    // if (iter % 100 == 0) { cout << "iter:" << iter << endl; }
    if (iter+1 % (nTrain/batch_size) == 0) {
      EvalRet train_ret, valid_ret, test_ret;
      eval(senti_net, bottom_vecs, top_vecs, (nTrain/batch_size), train_ret);
      eval(senti_valid, bottom_vecs, top_vecs, (nValid/batch_size), valid_ret);
      eval(senti_test,  bottom_vecs, top_vecs, (nTest/batch_size), test_ret);
      fprintf(stdout, "****%d,%f,%f,%f,%f,%f,%f", iter, train_ret.loss, valid_ret.loss, test_ret.loss, 
                                                        train_ret.acc,  valid_ret.acc,  test_ret.acc);
      if (valid_ret.acc > maxValidAcc) {
          maxValidAcc = valid_ret.acc;
          testAccByValid = test_ret.acc;
          fprintf(stdout, "*^_^*");
      }
      fprintf(stdout, "\n");
    }

    // cout << "Begin iter " << iter << endl;
    for (index_t i = 0; i < senti_net.size(); ++i) {
      senti_net[i]->SetPhrase(kTrain);
      // cout << "Forward layer " << i << endl;
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
    
    for (int i = 0; i < senti_net.size(); ++i) {
        senti_net[i]->ClearDiff(bottom_vecs[i], top_vecs[i]);
    }
    for (int i = senti_net.size()-2; i >= 0; --i) {
      // cout << "Backprop layer " << i << endl;
      senti_net[i]->Backprop(bottom_vecs[i], top_vecs[i]);
    }
    for (index_t i = 0; i < senti_net.size(); ++i) {
      for (index_t j = 0; j < senti_net[i]->ParamNodeNum(); ++j) {
        // cout << "Update param in layer " << i << " params " << j << endl;
        // cout << "param data" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].data[0][0][0][0] << endl;
        // cout << "param diff" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].diff[0][0][0][0] << endl;
        if (senti_net[i]->layer_name == "softmax" && j == 1) continue; // bias is shit
        if (senti_net[i]->layer_name == "lstm" && j == 2) continue; // bias is shit
        senti_net[i]->GetParams()[j].Update();
        // cout << "param data" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].data[0][0][0][0] << endl<<endl;
      }
    }
    // Output informations
    // cout << "###### Iter " << iter << ": error =\t" << nodes[nodes.size()-2]->data_d1()[0] << endl;
  }
  
    return 0;
}

