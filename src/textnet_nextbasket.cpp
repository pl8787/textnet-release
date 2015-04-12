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
      ret.acc  += tops[tops.size()-1][0]->data_d1()[0] * tops[0][0]->data.size(0);
  }
  ret.loss /= float(nBatch);
  // ret.acc /= float(nBatch);
}
    

int main(int argc, char *argv[]) {
  mshadow::Random<cpu> rnd(37);
  vector<Layer<cpu>*> senti_net, senti_valid, senti_test, senti_train;
  if (argc >= 3) {
    freopen(argv[2], "w", stdout);
    setvbuf(stdout, NULL, _IOLBF, 0);
  }

  bool no_bias = true;
  float l2 = 0.f;
  int max_session_len = 300;
  int context_window = 1;
  int min_doc_len = 1;
  int batch_size = 1;
  int word_rep_dim = 50;
  int num_hidden = (context_window+1) * word_rep_dim;
  int num_item = 7973;
  int num_user = 2265;
  int num_class = num_item;
  float base_lr = 0.1;
  float ADA_GRAD_EPS = 0.1;
  float decay = 0.00;
  int nTrain = 168933;// nValid = 7, nTest = 7;
  int nTrainPred = num_user;// nValid = 7, nTest = 7;
  int nValid = num_user;// nValid = 7, nTest = 7;
  int nTest = num_user; // nValid = 7, nTest = 7;
  int max_iter = (20*nTrain)/(batch_size);
  int iter_eval = (nTrain/batch_size)/20;
  int ADA_GRAD_MAX_ITER = 1000000;
  // int ADA_GRAD_MAX_ITER = nTrain/batch_size;

  string train_data_file = "/home/wsx/dl.shengxian/data/pengfei/tafeng_sub.textnet.train.1";
  string train_pred_data_file = "/home/wsx/dl.shengxian/data/pengfei/tafeng_sub.textnet.train_pred.1";
  string valid_data_file = "/home/wsx/dl.shengxian/data/pengfei/tafeng_sub.textnet.valid.1";
  string test_data_file  = "/home/wsx/dl.shengxian/data/pengfei/tafeng_sub.textnet.test.1";
  string embedding_file = ""; 
  // string embedding_file = "/home/pangliang/matching/data/wikicorp_50_msr.txt";
  // int nTrain = 1000;// nValid = 7, nTest = 7;

  if (argc >= 2) {
    base_lr = atof(argv[1]);
  }
  map<string, SettingV> &g_updater = *(new map<string, SettingV>());
  // g_updater["updater_type"] = SettingV(updater::kAdaDelta);
  g_updater["updater_type"] = SettingV(updater::kAdagrad);
  g_updater["batch_size"] = SettingV(batch_size);
  g_updater["l2"] = SettingV(l2);
  g_updater["max_iter"] = SettingV(ADA_GRAD_MAX_ITER);
  g_updater["eps"] = SettingV(ADA_GRAD_EPS);
  g_updater["lr"] = SettingV(base_lr);
  // g_updater["decay"] = SettingV(decay);  
  // g_updater["momentum"] = SettingV(0.0f);

  
  senti_net.push_back(CreateLayer<cpu>(kNextBasketData));
  senti_net[senti_net.size()-1]->layer_name = "data";
  senti_net.push_back(CreateLayer<cpu>(kEmbedding));
  senti_net[senti_net.size()-1]->layer_name = "user_embedding";
  senti_net.push_back(CreateLayer<cpu>(kEmbedding));
  senti_net[senti_net.size()-1]->layer_name = "item_embedding";
  senti_net.push_back(CreateLayer<cpu>(kWholePooling));
  senti_net[senti_net.size()-1]->layer_name = "wholepooling";
  senti_net.push_back(CreateLayer<cpu>(kConcat));
  senti_net[senti_net.size()-1]->layer_name = "concat";
  senti_net.push_back(CreateLayer<cpu>(kFullConnect));
  senti_net[senti_net.size()-1]->layer_name = "fullconnect";
  senti_net.push_back(CreateLayer<cpu>(kRectifiedLinear));
  senti_net[senti_net.size()-1]->layer_name = "activation";
  senti_net.push_back(CreateLayer<cpu>(kDropout));
  senti_net[senti_net.size()-1]->layer_name = "dropout";
  senti_net.push_back(CreateLayer<cpu>(kFullConnect));
  senti_net[senti_net.size()-1]->layer_name = "fullconnect";
  senti_net.push_back(CreateLayer<cpu>(kSoftmax));
  senti_net[senti_net.size()-1]->layer_name = "softmax";
  senti_net.push_back(CreateLayer<cpu>(kAccuracy));
  senti_net[senti_net.size()-1]->layer_name = "accuracy";
  

  senti_train = senti_net;
  senti_test = senti_net;
  senti_valid = senti_net;
  senti_train[0] = CreateLayer<cpu>(kNextBasketData);
  senti_test[0] = CreateLayer<cpu>(kNextBasketData);
  senti_valid[0]= CreateLayer<cpu>(kNextBasketData);
  
  vector<vector<Node<cpu>*> > bottom_vecs(senti_net.size(), vector<Node<cpu>*>());
  vector<vector<Node<cpu>*> > top_vecs(senti_net.size(), vector<Node<cpu>*>());
  vector<Node<cpu>*> nodes;
  for (int i = 0; i < senti_net.size(); ++i) { // last layers are softmax layer and accuracy layers
    int top_node_num = 0;
    top_node_num = senti_net[i]->TopNodeNum();
    for (int j = 0; j < top_node_num; ++j) {
      Node<cpu>* node = new Node<cpu>();
      stringstream ss;
      ss << nodes.size();
      node->node_name = ss.str();
      nodes.push_back(node);
      top_vecs[i].push_back(node);
      senti_net[i]->top_nodes.push_back(node->node_name);
    }
  }

  int layerIdx = 0;
  // data
  senti_net[0]->top_nodes.push_back(nodes[0]->node_name); // label
  senti_net[0]->top_nodes.push_back(nodes[1]->node_name); // label set
  senti_net[0]->top_nodes.push_back(nodes[2]->node_name); // user
  senti_net[0]->top_nodes.push_back(nodes[3]->node_name); // item
  senti_valid[0]->top_nodes.push_back(nodes[0]->node_name); // label
  senti_valid[0]->top_nodes.push_back(nodes[1]->node_name); // label set
  senti_valid[0]->top_nodes.push_back(nodes[2]->node_name); // user
  senti_valid[0]->top_nodes.push_back(nodes[3]->node_name); // item
  ++layerIdx;
  // user_embedding
  bottom_vecs[1].push_back(top_vecs[0][2]);
  senti_net[1]->bottom_nodes.push_back(nodes[2]->node_name);
  senti_net[1]->top_nodes.push_back(nodes[4]->node_name); 
  ++layerIdx;
  // item_embedding
  bottom_vecs[2].push_back(top_vecs[0][3]);
  senti_net[2]->bottom_nodes.push_back(nodes[3]->node_name);
  senti_net[2]->top_nodes.push_back(nodes[5]->node_name); 
  ++layerIdx;
  // whole_pooling
  bottom_vecs[3].push_back(top_vecs[2][0]); 
  senti_net[3]->bottom_nodes.push_back(nodes[5]->node_name);
  senti_net[3]->top_nodes.push_back(nodes[6]->node_name); 
  ++layerIdx; // 4
  // concat
  bottom_vecs[layerIdx].push_back(top_vecs[layerIdx-3][0]); 
  bottom_vecs[layerIdx].push_back(top_vecs[layerIdx-1][0]); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[4]->node_name); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[6]->node_name); 
  senti_net[layerIdx]->top_nodes.push_back(nodes[7]->node_name);
  ++layerIdx;
  // fullconnect
  bottom_vecs[layerIdx].push_back(top_vecs[layerIdx-1][0]); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[layerIdx+3-1]->node_name); 
  senti_net[layerIdx]->top_nodes.push_back(nodes[layerIdx+3]->node_name);
  ++layerIdx;
  // activation
  bottom_vecs[layerIdx].push_back(top_vecs[layerIdx-1][0]); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[layerIdx+3-1]->node_name); 
  senti_net[layerIdx]->top_nodes.push_back(nodes[layerIdx+3]->node_name);
  ++layerIdx;
  // dropout
  bottom_vecs[layerIdx].push_back(top_vecs[layerIdx-1][0]); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[layerIdx+3-1]->node_name); 
  senti_net[layerIdx]->top_nodes.push_back(nodes[layerIdx+3]->node_name);
  ++layerIdx;
  // fullconnect
  bottom_vecs[layerIdx].push_back(top_vecs[layerIdx-1][0]); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[layerIdx+3-1]->node_name); 
  senti_net[layerIdx]->top_nodes.push_back(nodes[layerIdx+3]->node_name);
  ++layerIdx;
  // softmax
  bottom_vecs[layerIdx].push_back(top_vecs[layerIdx-1][0]); 
  bottom_vecs[layerIdx].push_back(top_vecs[0][0]); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[layerIdx+3-1]->node_name); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[0]->node_name); 
  senti_net[layerIdx]->top_nodes.push_back(nodes[layerIdx+3]->node_name);
  ++layerIdx;
  // hit_cnt 
  bottom_vecs[layerIdx].push_back(top_vecs[layerIdx-2][0]);
  bottom_vecs[layerIdx].push_back(top_vecs[0][1]); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[layerIdx+3-2]->node_name); 
  senti_net[layerIdx]->bottom_nodes.push_back(nodes[1]->node_name); 
  senti_net[layerIdx]->top_nodes.push_back(nodes[layerIdx+3]->node_name);

  // Fill Settings vector
  vector<map<string, SettingV> > setting_vec;
  // kNextBasketLayer
  {
    map<string, SettingV> setting;
    // orc
    setting["data_file"] = SettingV(train_data_file);
    setting["batch_size"] = SettingV(batch_size);
    setting["max_session_len"] = SettingV(max_session_len);
    setting["context_window"] = SettingV(context_window);
    setting_vec.push_back(setting);
    
    // test 
    setting["batch_size"] = SettingV(batch_size);
    setting["data_file"] = SettingV(valid_data_file);
    senti_valid[0]->PropAll();
    senti_valid[0]->SetupLayer(setting, bottom_vecs[0], top_vecs[0], &rnd);
    senti_valid[0]->Reshape(bottom_vecs[0], top_vecs[0]);
    
    setting["batch_size"] = SettingV(batch_size);
    setting["data_file"] = SettingV(train_pred_data_file);
    senti_train[0]->PropAll();
    senti_train[0]->SetupLayer(setting, bottom_vecs[0], top_vecs[0], &rnd);
    senti_train[0]->Reshape(bottom_vecs[0], top_vecs[0]);



    setting["batch_size"] = SettingV(batch_size);
    setting["data_file"] = SettingV(test_data_file);
    senti_test[0]->PropAll();
    senti_test[0]->SetupLayer(setting, bottom_vecs[0], top_vecs[0], &rnd);
    senti_test[0]->Reshape(bottom_vecs[0], top_vecs[0]);
  }
  // kEmbedding user 
  {
    map<string, SettingV> setting;
    // orc
    setting["embedding_file"] = SettingV(embedding_file);
    setting["word_count"] = SettingV(num_user);
    setting["feat_size"] = SettingV(word_rep_dim);
    setting["pad_value"] = SettingV((float)(NAN));
      
    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kUniform);
      w_setting["range"] = SettingV(1.f/word_rep_dim);
    setting["w_filler"] = SettingV(&w_setting);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    setting["w_updater"] = SettingV(&w_updater);
    setting_vec.push_back(setting);
  }
  // kEmbedding item
  {
    map<string, SettingV> setting;
    // orc
    setting["embedding_file"] = SettingV(embedding_file);
    setting["word_count"] = SettingV(num_item);
    setting["feat_size"] = SettingV(word_rep_dim);
    setting["pad_value"] = SettingV((float)(NAN));
      
    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
    w_setting = *setting_vec[setting_vec.size()-1]["w_filler"].mVal();
    setting["w_filler"] = SettingV(&w_setting);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    setting["w_updater"]= SettingV(&w_updater);

    setting_vec.push_back(setting);
  }
  // kWholeMaxPooling
  {
    map<string, SettingV> setting;
    setting["pool_type"] = "ave";
    setting_vec.push_back(setting);
  }
  // kConcat
  {
    map<string, SettingV> setting;
    setting["bottom_node_num"] = context_window + 1;
    setting_vec.push_back(setting);
  }
  // kFullConnect
  {
    map<string, SettingV> setting;
    setting["num_hidden"] = SettingV(num_hidden);
    setting["no_bias"] = SettingV(no_bias);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kUniform);
      w_setting["range"] = SettingV(1.f/word_rep_dim);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
    b_updater = g_updater;

    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
    setting_vec.push_back(setting);
  }
  //  kActivation
  {
    map<string, SettingV> setting;
    setting_vec.push_back(setting);
  }
  //  kDropout
  {
    map<string, SettingV> setting;
    setting["rate"] = SettingV(0.5f);
    setting_vec.push_back(setting);
  }
  // kFullConnect
  {
    map<string, SettingV> setting;
    setting["num_hidden"] = SettingV(num_class);
    setting["no_bias"] = SettingV(no_bias);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      // w_setting["init_type"] = SettingV(initializer::kZero);
      w_setting["init_type"] = SettingV(initializer::kUniform);
      w_setting["range"] = SettingV(1.f/word_rep_dim);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
    w_updater = g_updater;
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
    b_updater = g_updater;
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
    setting_vec.push_back(setting);
  }
  // kSoftmax
  {
    map<string, SettingV> setting;
    // map<string, SettingV> &w_setting = *(new map<string, SettingV>());
    //   // w_setting["init_type"] = SettingV(initializer::kZero);
    //   w_setting["init_type"] = SettingV(initializer::kZero);
    // map<string, SettingV> &b_setting = *(new map<string, SettingV>());
    //   b_setting["init_type"] = SettingV(initializer::kZero);
    // setting["w_filler"] = SettingV(&w_setting);
    // setting["b_filler"] = SettingV(&b_setting);
    setting_vec.push_back(setting);
  }
  // kAccuracy
  {
    map<string, SettingV> setting;
    setting["topk"] = SettingV(5);
    setting_vec.push_back(setting);
  }
  cout << "Setting Vector Filled." << endl;

  // Set up Layers
  for (index_t i = 0; i < senti_net.size(); ++i) {
    cout << "Begin set up layer " << i << endl;
    senti_net[i]->PropAll();
    cout << "\tPropAll" << endl;
    senti_net[i]->SetupLayer(setting_vec[i], bottom_vecs[i], top_vecs[i], &rnd);
    cout << "\tSetup Layer" << endl;
    senti_net[i]->Reshape(bottom_vecs[i], top_vecs[i]);
    cout << "\tReshape" << endl;
  }

  // Save Initial Model
  {
  ofstream _of("./next.basket.model");
  Json::StyledWriter writer;
  Json::Value net_root;
  net_root["net_name"] = "next_basket";
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

  // Begin Training 
  float maxValidAcc = 0., testAccByValid = 0.;
  for (int iter = 0; iter < max_iter; ++iter) {
    // if (iter % 100 == 0) { cout << "iter:" << iter << endl; }
    if (iter % iter_eval == 0) {
      EvalRet train_ret, valid_ret, test_ret;
      // eval(senti_net, bottom_vecs, top_vecs, (nTrain/batch_size), train_ret);
      eval(senti_train, bottom_vecs, top_vecs, (nTrainPred/batch_size), train_ret);
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
        if (senti_net[i]->layer_name == "fullconnect" && j == 1) continue; // bias is shit
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

