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
          senti_net[j]->Forward(bottoms[j], tops[j]);
      }
      // if (bottoms[bottoms.size()-1][0]->data.dptr_[0] == 1. ||
      //     bottoms[bottoms.size()-1][0]->data.dptr_[0] == 0.) {
      //     assert(false);
      // }

      ret.loss += tops[tops.size()-2][0]->data_d1()[0];
      ret.acc  += tops[tops.size()-1][0]->data_d1()[0];
  }
  ret.loss /= float(nBatch);
  ret.acc /= float(nBatch);
}
    

int main(int argc, char *argv[]) {
  mshadow::Random<cpu> rnd(37);
  vector<Layer<cpu>*> senti_net, senti_valid, senti_test;

  int lstm_hidden_dim = 2;
  int word_rep_dim = 100;
  int max_doc_len = 100;
  int min_doc_len = 1;
  int batch_size = 1;
  int vocab_size = 200000;
  int num_class = 2;
  float base_lr = 0.1;
  float ADA_GRAD_EPS = 0.01;
  float decay = 0.00;
  if (argc == 2) {
    base_lr = atof(argv[1]);
  }
  
  senti_net.push_back(CreateLayer<cpu>(kTextData));
  senti_net[senti_net.size()-1]->layer_name = "text";
  senti_net.push_back(CreateLayer<cpu>(kEmbedding));
  senti_net[senti_net.size()-1]->layer_name = "embedding";
  // senti_net.push_back(CreateLayer<cpu>(kLstm));
  // senti_net[senti_net.size()-1].name == "kLstm";
  senti_net.push_back(CreateLayer<cpu>(kWholeMaxPooling));
  senti_net[senti_net.size()-1]->layer_name = "wholepooling";
  senti_net.push_back(CreateLayer<cpu>(kFullConnect));
  senti_net[senti_net.size()-1]->layer_name = "fullconnect";
  senti_net.push_back(CreateLayer<cpu>(kSoftmax));
  senti_net[senti_net.size()-1]->layer_name = "softmax";
  senti_net.push_back(CreateLayer<cpu>(kAccuracy));
  senti_net[senti_net.size()-1]->layer_name = "accuracy";

  senti_test = senti_net;
  senti_valid = senti_net;
  senti_test[0] = CreateLayer<cpu>(kTextData);
  senti_valid[0]= CreateLayer<cpu>(kTextData);
  
  vector<vector<Node<cpu>*> > bottom_vecs(senti_net.size(), vector<Node<cpu>*>());
  vector<vector<Node<cpu>*> > top_vecs(senti_net.size(), vector<Node<cpu>*>());
  vector<Node<cpu>*> nodes;

  // Manual connect layers
  for (int i = 0; i < senti_net.size(); ++i) { // last layers are softmax layer and accuracy layers
    for (int j = 0; j < senti_net[i]->TopNodeNum(); ++j) {
      Node<cpu>* node = new Node<cpu>();
      nodes.push_back(node);
      top_vecs[i].push_back(node);
    }
    if (i == 0) {
      continue;
    } else if (i == 1) {
      bottom_vecs[i].push_back(nodes[0]);
    } else if (senti_net[i]->layer_name == "accuracy") {
      bottom_vecs[i].push_back(top_vecs[i-2][0]);
    } else {
      bottom_vecs[i].push_back(top_vecs[i-1][0]);
    }
  }
  bottom_vecs[senti_net.size()-2].push_back(nodes[1]); // softmax
  bottom_vecs[senti_net.size()-1].push_back(nodes[1]); // accuracy
  cout << "Total node count: " << nodes.size() << endl;
  // senti_net_test[0]->top_nodes.push_back(nodes[0]);
  // senti_net_test[0]->top_nodes.push_back(nodes[1]);
  
  // Fill Settings vector
  vector<map<string, SettingV> > setting_vec;
  // kTextData
  {
    map<string, SettingV> setting;
    // orc
    // setting["data_file"] = SettingV("/home/pangliang/matching/data/msr_paraphrase_train_wid_dup.txt");
    setting["data_file"] = SettingV("/home/wsx/dl.shengxian/data/mr/all.lstm.shuffle");
    // setting["data_file"] = SettingV("/home/wsx/dl.shengxian/data/treebank/trees/train.seq.allnode.unique.pad.binary.shuffle");
    setting["batch_size"] = SettingV(batch_size);
    setting["max_doc_len"] = SettingV(max_doc_len);
    setting["min_doc_len"] = SettingV(1);
    setting_vec.push_back(setting);
    
    // valid
    // setting["batch_size"] = SettingV(1);
    // setting["data_file"] = SettingV("/home/wsx/dl.shengxian/data/treebank/trees/dev.seq.pad.binary");
    // senti_valid[0]->PropAll();
    // senti_valid[0]->SetupLayer(setting, bottom_vecs[0], top_vecs[0], &rnd);
    // senti_valid[0]->Reshape(bottom_vecs[0], top_vecs[0]);
    // // test 
    // // setting["batch_size"] = SettingV(1);
    // setting["data_file"] = SettingV("/home/wsx/dl.shengxian/data/treebank/trees/test.seq.pad.binary");
    // senti_test[0]->PropAll();
    // senti_test[0]->SetupLayer(setting, bottom_vecs[0], top_vecs[0], &rnd);
    // senti_test[0]->Reshape(bottom_vecs[0], top_vecs[0]);
  }
  // kEmbedding
  {
    map<string, SettingV> setting;
    // orc
    // setting["embedding_file"] = SettingV("/home/pangliang/matching/data/wikicorp_50_msr.txt");
    // setting["embedding_file"] = SettingV("/home/wsx/repo.other/textnet/data/wikicorp_50_msr.txt");
    // setting["embedding_file"] = SettingV("/home/wsx/dl.shengxian/data/mr/embedding.test");
    setting["word_count"] = SettingV(vocab_size);
    setting["feat_size"] = SettingV(word_rep_dim);
      
    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kUniform);
      w_setting["range"] = SettingV(0.001f);
    setting["w_filler"] = SettingV(&w_setting);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      // w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["eps"] = SettingV(ADA_GRAD_EPS);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);  
    setting["w_updater"] = SettingV(&w_updater);
    setting_vec.push_back(setting);
  }
  // kLstm
  // {
  //   map<string, SettingV> setting;
  //   setting["d_input"] = SettingV(word_rep_dim);
  //   setting["d_mem"] = SettingV(lstm_hidden_dim);
  //   setting["no_bias"] = SettingV(true);
  //     
  //   map<string, SettingV> &w_filler = *(new map<string, SettingV>());
  //     w_filler["init_type"] = SettingV(initializer::kUniform);
  //     w_filler["range"] = SettingV(0.01f);
  //   setting["w_filler"] = SettingV(&w_filler);
  //   setting["u_filler"] = SettingV(&w_filler);
  //   // map<string, SettingV> &u_filler = *(new map<string, SettingV>());
  //   //   u_filler["init_type"] = SettingV(initializer::kUniform);
  //   //   u_filler["range"] = SettingV(0.01f);
  //   // setting["u_filler"] = SettingV(&u_filler);
  //   map<string, SettingV> &b_filler = *(new map<string, SettingV>());
  //     b_filler["init_type"] = SettingV(initializer::kZero);
  //   setting["b_filler"] = SettingV(&b_filler);
  //     
  //   map<string, SettingV> &w_updater = *(new map<string, SettingV>());
  //     w_updater["updater_type"] = SettingV(updater::kAdagrad);
  //     w_updater["momentum"] = SettingV(0.0f);
  //     w_updater["eps"] = SettingV(ADA_GRAD_EPS);
  //     w_updater["lr"] = SettingV(base_lr);
  //     w_updater["decay"] = SettingV(decay);  
  //   setting["w_updater"] = SettingV(&w_updater);
  //   setting["u_updater"] = SettingV(&w_updater);
  //   setting["b_updater"] = SettingV(&w_updater);
  //   // map<string, SettingV> &u_updater = *(new map<string, SettingV>());
  //   //   u_updater["updater_type"] = SettingV(updater::kAdagrad);
  //   //   u_updater["momentum"] = SettingV(0.0f);
  //   //   u_updater["eps"] = SettingV(ADA_GRAD_EPS);
  //   //   u_updater["lr"] = SettingV(base_lr);
  //   //   u_updater["decay"] = SettingV(decay);  
  //   // setting["u_updater"] = SettingV(&u_updater);
  //   // map<string, SettingV> &b_updater = *(new map<string, SettingV>());
  //   //   b_updater["updater_type"] = SettingV(updater::kAdagrad);
  //   //   b_updater["momentum"] = SettingV(0.0f);
  //   //   b_updater["eps"] = SettingV(ADA_GRAD_EPS);
  //   //   b_updater["lr"] = SettingV(base_lr);
  //   //   b_updater["decay"] = SettingV(decay);  
  //   // setting["b_updater"] = SettingV(&b_updater);
  //   
  //   setting_vec.push_back(setting);
  // }
  // kWholeMaxPooling
  {
    map<string, SettingV> setting;
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
      w_setting["range"] = SettingV(0.1f);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      // w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["eps"] = SettingV(ADA_GRAD_EPS);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      // b_updater["updater_type"] = SettingV(updater::kAdagrad);
      b_updater["updater_type"] = SettingV(updater::kSGD);
      b_updater["eps"] = SettingV(ADA_GRAD_EPS);
      b_updater["momentum"] = SettingV(0.0f);
      b_updater["lr"] = SettingV(base_lr);
      b_updater["decay"] = SettingV(decay);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
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
    senti_net[i]->PropAll();
    // cout << "\tPropAll" << endl;
    senti_net[i]->SetupLayer(setting_vec[i], bottom_vecs[i], top_vecs[i], &rnd);
    // cout << "\tSetup Layer" << endl;
    senti_net[i]->Reshape(bottom_vecs[i], top_vecs[i]);
    // cout << "\tReshape" << endl;
  }
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.1f);
  setting_checker["range_max"] = SettingV(0.1f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, &rnd);
  // for (index_t i = 0; i < senti_net.size(); ++i) {
  //   // cout << "Forward layer " << i << endl;
  //   senti_net[i]->Forward(bottom_vecs[i], top_vecs[i]);
  //   // cker->CheckError(senti_net[i], bottom_vecs[i], top_vecs[i]);
  // }
  // exit(0);

  // Begin Training 
  int max_iters = 300000;
  float maxValidAcc = 0., testAccByValid = 0.;
  for (int iter = 0; iter < max_iters; ++iter) {
    if (iter % 10000 == 0) {
      EvalRet train_ret, valid_ret, test_ret;
      eval(senti_net, bottom_vecs, top_vecs, (10000/batch_size), train_ret);
      // eval(senti_valid, bottom_vecs, top_vecs, (872/batch_size), valid_ret);
      // eval(senti_test,  bottom_vecs, top_vecs, (1821/batch_size), test_ret);
      fprintf(stdout, "****%f,%f,%f,%f,%f,%f", train_ret.loss, valid_ret.loss, test_ret.loss, train_ret.acc, valid_ret.acc, test_ret.acc);
      if (valid_ret.acc > maxValidAcc) {
          maxValidAcc = valid_ret.acc;
          testAccByValid = test_ret.acc;
          fprintf(stdout, "*^_^*");
      }
      fprintf(stdout, "\n");
    }

    // cout << "Begin iter " << iter << endl;
    for (index_t i = 0; i < senti_net.size(); ++i) {
      // cout << "Forward layer " << i << endl;
      senti_net[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }
    assert(true);
    
    int tmp = 1;
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
    

    for (int i = senti_net.size()-2; i >= 0; --i) {
      // cout << "Backprop layer " << i << endl;
      senti_net[i]->Backprop(bottom_vecs[i], top_vecs[i]);
    }
    // cout << "HELLO" << endl;
    tmp = 2;
    assert(true);
    for (index_t i = 0; i < senti_net.size(); ++i) {
      for (index_t j = 0; j < senti_net[i]->ParamNodeNum(); ++j) {
        // cout << "Update param in layer " << i << " params " << j << endl;
        // cout << "param data" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].data[0][0][0][0] << endl;
        // cout << "param diff" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].diff[0][0][0][0] << endl;
        if (i == 3 && j == 1) continue;
        senti_net[i]->GetParams()[j].Update();
        // cout << "param data" << i << " , " << j << ": " << senti_net[i]->GetParams()[j].data[0][0][0][0] << endl<<endl;
      }
    }
    // if (iter % 10000 == 0) {
    //   cout << "Average loss:" << aveAcc/10000 << endl;
    //   aveAcc = 0.;
    // }
    // aveAcc += nodes[nodes.size()-1]->data_d1()[0];
    
    // Output informations
    // cout << "###### Iter " << iter << ": error =\t" << nodes[nodes.size()-2]->data_d1()[0] << endl;

    
          // cout << "****" << valid_ret.loss << test_ret.loss << 
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

