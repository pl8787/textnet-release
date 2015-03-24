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
  vector<Layer<cpu>*> matching_net;
  vector<Layer<cpu>*> matching_net_test;
  vector<vector<Node<cpu>*> > bottom_vecs;
  vector<vector<Node<cpu>*> > top_vecs;
  vector<Node<cpu>*> nodes;
  
  matching_net.push_back(CreateLayer<cpu>(kTextData));
  matching_net.push_back(CreateLayer<cpu>(kEmbedding));
  matching_net.push_back(CreateLayer<cpu>(kSplit));
  matching_net.push_back(CreateLayer<cpu>(kConv));
  matching_net.push_back(CreateLayer<cpu>(kConv));
  matching_net.push_back(CreateLayer<cpu>(kCross));
  matching_net.push_back(CreateLayer<cpu>(kMaxPooling));
  matching_net.push_back(CreateLayer<cpu>(kRectifiedLinear));
  matching_net.push_back(CreateLayer<cpu>(kConv));
  matching_net.push_back(CreateLayer<cpu>(kMaxPooling));
  matching_net.push_back(CreateLayer<cpu>(kRectifiedLinear));
  matching_net.push_back(CreateLayer<cpu>(kConv));
  matching_net.push_back(CreateLayer<cpu>(kMaxPooling));
  matching_net.push_back(CreateLayer<cpu>(kRectifiedLinear));
  matching_net.push_back(CreateLayer<cpu>(kFullConnect));
  matching_net.push_back(CreateLayer<cpu>(kRectifiedLinear));
  matching_net.push_back(CreateLayer<cpu>(kDropout));
  matching_net.push_back(CreateLayer<cpu>(kFullConnect));
  matching_net.push_back(CreateLayer<cpu>(kSoftmax));
  
  matching_net_test = matching_net;
  matching_net_test[0] = CreateLayer<cpu>(kTextData);
  
  for (int i = 0; i < matching_net.size(); ++i) {
    vector<Node<cpu>*> bottoms;
    vector<Node<cpu>*> tops;
    bottom_vecs.push_back(bottoms);
    top_vecs.push_back(tops);
    for (int j = 0; j < matching_net[i]->TopNodeNum(); ++j) {
      Node<cpu>* node = new Node<cpu>();
      nodes.push_back(node);
    }
  }
  
  // Name the layers
  matching_net[0]->layer_name = "textdata";
  matching_net[1]->layer_name = "embedding";
  matching_net[2]->layer_name = "split";
  matching_net[3]->layer_name = "conv11";
  matching_net[4]->layer_name = "conv12";
  matching_net[5]->layer_name = "cross";
  matching_net[6]->layer_name = "maxpool1";
  matching_net[7]->layer_name = "relu1";
  matching_net[8]->layer_name = "conv2";
  matching_net[9]->layer_name = "maxpool2";
  matching_net[10]->layer_name = "relu2";
  matching_net[11]->layer_name = "conv3";
  matching_net[12]->layer_name = "maxpool3";
  matching_net[13]->layer_name = "relu3";
  matching_net[14]->layer_name = "fc1";
  matching_net[15]->layer_name = "relu4";
  matching_net[16]->layer_name = "dropout";
  matching_net[17]->layer_name = "fc2";
  matching_net[18]->layer_name = "softmax";
  
  matching_net_test[0]->layer_name = "textdata_test";
  matching_net_test[0]->layer_idx = 0;
  
  for (int i = 0; i < matching_net.size(); ++i) {
    matching_net[i]->layer_idx = i;
  }
  
  // Name the nodes
  nodes[0]->node_name = "data";
  nodes[1]->node_name = "label";
  nodes[2]->node_name = "embed";
  nodes[3]->node_name = "splt1";
  nodes[4]->node_name = "splt2";
  nodes[5]->node_name = "conv11";
  nodes[6]->node_name = "conv12";
  nodes[7]->node_name = "cross";
  nodes[8]->node_name = "pool1";
  nodes[9]->node_name = "relu1";
  nodes[10]->node_name = "conv2";
  nodes[11]->node_name = "pool2";
  nodes[12]->node_name = "relu2";
  nodes[13]->node_name = "conv3";
  nodes[14]->node_name = "pool3";
  nodes[15]->node_name = "relu3";
  nodes[16]->node_name = "fc1";
  nodes[17]->node_name = "relu4";
  nodes[18]->node_name = "drop";
  nodes[19]->node_name = "fc2";
  nodes[20]->node_name = "loss";
  
  for (int i = 0; i < nodes.size(); ++i) {
    nodes[i]->node_idx = i;
  }

  cout << "Total node count: " << nodes.size() << endl;
  
  // Manual connect layers
  // kTextData
  top_vecs[0].push_back(nodes[0]);
  top_vecs[0].push_back(nodes[1]);
  // kEmbedding
  bottom_vecs[1].push_back(nodes[0]);
  top_vecs[1].push_back(nodes[2]);
  // kSplit
  bottom_vecs[2].push_back(nodes[2]);
  top_vecs[2].push_back(nodes[3]);
  top_vecs[2].push_back(nodes[4]);
  // kConv
  bottom_vecs[3].push_back(nodes[3]);
  top_vecs[3].push_back(nodes[5]);
  // kConv
  bottom_vecs[4].push_back(nodes[4]);
  top_vecs[4].push_back(nodes[6]);
  // kCross
  bottom_vecs[5].push_back(nodes[5]);
  bottom_vecs[5].push_back(nodes[6]);
  top_vecs[5].push_back(nodes[7]);
  // kMaxPooling
  bottom_vecs[6].push_back(nodes[7]);
  top_vecs[6].push_back(nodes[8]);
  // kRectifiedLinear 
  bottom_vecs[7].push_back(nodes[8]);
  top_vecs[7].push_back(nodes[9]);
  // kConv 
  bottom_vecs[8].push_back(nodes[9]);
  top_vecs[8].push_back(nodes[10]);
  // kMaxPooling
  bottom_vecs[9].push_back(nodes[10]);
  top_vecs[9].push_back(nodes[11]);
  // kRectifiedLinear 
  bottom_vecs[10].push_back(nodes[11]);
  top_vecs[10].push_back(nodes[12]);
  // kConv 
  bottom_vecs[11].push_back(nodes[12]);
  top_vecs[11].push_back(nodes[13]);
  // kMaxPooling 
  bottom_vecs[12].push_back(nodes[13]);
  top_vecs[12].push_back(nodes[14]);
  // kRectifiedLinear 
  bottom_vecs[13].push_back(nodes[14]);
  top_vecs[13].push_back(nodes[15]);
  // kFullConnect
  bottom_vecs[14].push_back(nodes[15]);
  top_vecs[14].push_back(nodes[16]);
  // kRectifiedLinear
  bottom_vecs[15].push_back(nodes[16]);
  top_vecs[15].push_back(nodes[17]);
  // kDropout
  bottom_vecs[16].push_back(nodes[17]);
  top_vecs[16].push_back(nodes[18]);
  // kFullConnect 
  bottom_vecs[17].push_back(nodes[18]);
  top_vecs[17].push_back(nodes[19]);
  // kHingeLoss
  bottom_vecs[18].push_back(nodes[19]);
  bottom_vecs[18].push_back(nodes[1]);
  top_vecs[18].push_back(nodes[20]);
  
  // Manual connect layers node name
  // kTextData
  matching_net[0]->top_nodes.push_back(nodes[0]->node_name);
  matching_net[0]->top_nodes.push_back(nodes[1]->node_name);
  // kEmbedding
  matching_net[1]->bottom_nodes.push_back(nodes[0]->node_name);
  matching_net[1]->top_nodes.push_back(nodes[2]->node_name);
  // kSplit
  matching_net[2]->bottom_nodes.push_back(nodes[2]->node_name);
  matching_net[2]->top_nodes.push_back(nodes[3]->node_name);
  matching_net[2]->top_nodes.push_back(nodes[4]->node_name);
  // kConv
  matching_net[3]->bottom_nodes.push_back(nodes[3]->node_name);
  matching_net[3]->top_nodes.push_back(nodes[5]->node_name);
  // kConv
  matching_net[4]->bottom_nodes.push_back(nodes[4]->node_name);
  matching_net[4]->top_nodes.push_back(nodes[6]->node_name);
  // kCross
  matching_net[5]->bottom_nodes.push_back(nodes[5]->node_name);
  matching_net[5]->bottom_nodes.push_back(nodes[6]->node_name);
  matching_net[5]->top_nodes.push_back(nodes[7]->node_name);
  // kMaxPooling
  matching_net[6]->bottom_nodes.push_back(nodes[7]->node_name);
  matching_net[6]->top_nodes.push_back(nodes[8]->node_name);
  // kRectifiedLinear 
  matching_net[7]->bottom_nodes.push_back(nodes[8]->node_name);
  matching_net[7]->top_nodes.push_back(nodes[9]->node_name);
  // kConv 
  matching_net[8]->bottom_nodes.push_back(nodes[9]->node_name);
  matching_net[8]->top_nodes.push_back(nodes[10]->node_name);
  // kMaxPooling
  matching_net[9]->bottom_nodes.push_back(nodes[10]->node_name);
  matching_net[9]->top_nodes.push_back(nodes[11]->node_name);
  // kRectifiedLinear 
  matching_net[10]->bottom_nodes.push_back(nodes[11]->node_name);
  matching_net[10]->top_nodes.push_back(nodes[12]->node_name);
  // kConv 
  matching_net[11]->bottom_nodes.push_back(nodes[12]->node_name);
  matching_net[11]->top_nodes.push_back(nodes[13]->node_name);
  // kMaxPooling 
  matching_net[12]->bottom_nodes.push_back(nodes[13]->node_name);
  matching_net[12]->top_nodes.push_back(nodes[14]->node_name);
  // kRectifiedLinear 
  matching_net[13]->bottom_nodes.push_back(nodes[14]->node_name);
  matching_net[13]->top_nodes.push_back(nodes[15]->node_name);
  // kFullConnect
  matching_net[14]->bottom_nodes.push_back(nodes[15]->node_name);
  matching_net[14]->top_nodes.push_back(nodes[16]->node_name);
  // kRectifiedLinear
  matching_net[15]->bottom_nodes.push_back(nodes[16]->node_name);
  matching_net[15]->top_nodes.push_back(nodes[17]->node_name);
  // kDropout
  matching_net[16]->bottom_nodes.push_back(nodes[17]->node_name);
  matching_net[16]->top_nodes.push_back(nodes[18]->node_name);
  // kFullConnect 
  matching_net[17]->bottom_nodes.push_back(nodes[18]->node_name);
  matching_net[17]->top_nodes.push_back(nodes[19]->node_name);
  // kHingeLoss
  matching_net[18]->bottom_nodes.push_back(nodes[19]->node_name);
  matching_net[18]->bottom_nodes.push_back(nodes[1]->node_name);
  matching_net[18]->top_nodes.push_back(nodes[20]->node_name);
  
  // kTextData Test
  matching_net_test[0]->top_nodes.push_back(nodes[0]->node_name);
  matching_net_test[0]->top_nodes.push_back(nodes[1]->node_name);
  
  float base_lr = 0.01;
  float decay = 0.01;
  // Fill Settings vector
  vector<map<string, SettingV> > setting_vec;
  // kTextData
  {
    map<string, SettingV> setting;
    setting["data_file"] = SettingV("/home/pangliang/matching/data/msr_paraphrase_train_wid.txt");
    setting["batch_size"] = SettingV(50);
    setting["max_doc_len"] = SettingV(31);
    setting["min_doc_len"] = SettingV(5);
    setting_vec.push_back(setting);
  }
  // kEmbedding
  {
    map<string, SettingV> setting;
    setting["embedding_file"] = SettingV("/home/pangliang/matching/data/wikicorp_50_msr.txt");
    setting["word_count"] = SettingV(14727);
    setting["feat_size"] = SettingV(50);
      
    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kUniform);
      w_setting["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_setting);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);  
    setting["w_updater"] = SettingV(&w_updater);
    setting_vec.push_back(setting);
  }
  // kSplit
  {
    map<string, SettingV> setting;
    setting_vec.push_back(setting);
  }
  // kConv
  {
    map<string, SettingV> setting;
    setting["kernel_x"] = SettingV(50);
    setting["kernel_y"] = SettingV(3);
    setting["pad_x"] = SettingV(0);
    setting["pad_y"] = SettingV(1);
    setting["stride"] = SettingV(1);
    setting["channel_out"] = SettingV(200);
    setting["no_bias"] = SettingV(false);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kGaussian);
      w_setting["mu"] = SettingV(0.0f);
      w_setting["sigma"] = SettingV(0.02f);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      b_updater["updater_type"] = SettingV(updater::kSGD);
      b_updater["momentum"] = SettingV(0.0f);
      b_updater["lr"] = SettingV(base_lr);
      b_updater["decay"] = SettingV(decay);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
    setting_vec.push_back(setting);
  }
  // kConv
  {
    map<string, SettingV> setting;
    setting["kernel_x"] = SettingV(50);
    setting["kernel_y"] = SettingV(3);
    setting["pad_x"] = SettingV(0);
    setting["pad_y"] = SettingV(1);
    setting["stride"] = SettingV(1);
    setting["channel_out"] = SettingV(200);
    setting["no_bias"] = SettingV(false);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kGaussian);
      w_setting["mu"] = SettingV(0.0f);
      w_setting["sigma"] = SettingV(0.02f);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      b_updater["updater_type"] = SettingV(updater::kSGD);
      b_updater["momentum"] = SettingV(0.0f);
      b_updater["lr"] = SettingV(base_lr);
      b_updater["decay"] = SettingV(decay);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
    setting_vec.push_back(setting);
  }
  // kCross
  {
    map<string, SettingV> setting;
    setting_vec.push_back(setting);
  }
  // kMaxPooling
  {
    map<string, SettingV> setting;
    setting["kernel_x"] = SettingV(2);
    setting["kernel_y"] = SettingV(2);
    setting["stride"] = SettingV(2);
    setting_vec.push_back(setting);
  }
  // kRectifiedLinear 
  {
    map<string, SettingV> setting;
    setting_vec.push_back(setting);
  }
  // kConv 
  {
    map<string, SettingV> setting;
    setting["kernel_x"] = SettingV(2);
    setting["kernel_y"] = SettingV(2);
    setting["pad_x"] = SettingV(0);
    setting["pad_y"] = SettingV(0);
    setting["stride"] = SettingV(1);
    setting["channel_out"] = SettingV(300);
    setting["no_bias"] = SettingV(false);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kGaussian);
      w_setting["mu"] = SettingV(0.0f);
      w_setting["sigma"] = SettingV(0.01f);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      b_updater["updater_type"] = SettingV(updater::kSGD);
      b_updater["momentum"] = SettingV(0.0f);
      b_updater["lr"] = SettingV(base_lr);
      b_updater["decay"] = SettingV(decay);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
    setting_vec.push_back(setting);
  }
  // kMaxPooling
  {
    map<string, SettingV> setting;
    setting["kernel_x"] = SettingV(2);
    setting["kernel_y"] = SettingV(2);
    setting["stride"] = SettingV(2);
    setting_vec.push_back(setting);
  }
  // kRectifiedLinear 
  {
    map<string, SettingV> setting;
    setting_vec.push_back(setting);
  }
  // kConv 
  {
    map<string, SettingV> setting;
    setting["kernel_x"] = SettingV(2);
    setting["kernel_y"] = SettingV(2);
    setting["pad_x"] = SettingV(0);
    setting["pad_y"] = SettingV(0);
    setting["stride"] = SettingV(1);
    setting["channel_out"] = SettingV(400);
    setting["no_bias"] = SettingV(false);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kGaussian);
      w_setting["mu"] = SettingV(0.0f);
      w_setting["sigma"] = SettingV(0.01f);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      b_updater["updater_type"] = SettingV(updater::kSGD);
      b_updater["momentum"] = SettingV(0.0f);
      b_updater["lr"] = SettingV(base_lr);
      b_updater["decay"] = SettingV(decay);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
    setting_vec.push_back(setting);
  }
  // kMaxPooling 
  {
    map<string, SettingV> setting;
    setting["kernel_x"] = SettingV(2);
    setting["kernel_y"] = SettingV(2);
    setting["stride"] = SettingV(2);
    setting_vec.push_back(setting);
  }
  // kRectifiedLinear 
  {
    map<string, SettingV> setting;
    setting_vec.push_back(setting);
  }
  // kFullConnect
  {
    map<string, SettingV> setting;
    setting["num_hidden"] = SettingV(512);
    setting["no_bias"] = SettingV(false);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kGaussian);
      w_setting["mu"] = SettingV(0.0f);
      w_setting["sigma"] = SettingV(0.005f);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      b_updater["updater_type"] = SettingV(updater::kSGD);
      b_updater["momentum"] = SettingV(0.0f);
      b_updater["lr"] = SettingV(base_lr);
      b_updater["decay"] = SettingV(decay);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
    setting_vec.push_back(setting);
  }
  // kRectifiedLinear
  {
    map<string, SettingV> setting;
    setting_vec.push_back(setting);
  }
  // kDropout
  {
    map<string, SettingV> setting;
    setting["rate"] = SettingV(0.5f);
    setting_vec.push_back(setting);
  }
  // kFullConnect 
  {
    map<string, SettingV> setting;
    setting["num_hidden"] = SettingV(2);
    setting["no_bias"] = SettingV(false);

    map<string, SettingV> &w_setting = *(new map<string, SettingV>());
      w_setting["init_type"] = SettingV(initializer::kGaussian);
      w_setting["mu"] = SettingV(0.0f);
      w_setting["sigma"] = SettingV(0.1f);
    map<string, SettingV> &b_setting = *(new map<string, SettingV>());
      b_setting["init_type"] = SettingV(initializer::kZero);
    setting["w_filler"] = SettingV(&w_setting);
    setting["b_filler"] = SettingV(&b_setting);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kSGD);
      w_updater["momentum"] = SettingV(0.0f);
      w_updater["lr"] = SettingV(base_lr);
      w_updater["decay"] = SettingV(decay);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      b_updater["updater_type"] = SettingV(updater::kSGD);
      b_updater["momentum"] = SettingV(0.0f);
      b_updater["lr"] = SettingV(base_lr);
      b_updater["decay"] = SettingV(decay);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&b_updater);
    setting_vec.push_back(setting);
  }
  // kHingeLoss
  {
    map<string, SettingV> setting;
    setting["delta"] = SettingV(1.0f);
    setting_vec.push_back(setting);
  }
  
  // kTextData
  {
    map<string, SettingV> setting;
    setting["data_file"] = SettingV("/home/pangliang/matching/data/msr_paraphrase_test_wid.txt");
    setting["batch_size"] = SettingV(50);//1725);
    setting["max_doc_len"] = SettingV(31);
    setting["min_doc_len"] = SettingV(5);
    setting_vec.push_back(setting);
  }
  
  cout << "Setting Vector Filled." << endl;

  // Set up Layers
  for (int i = 0; i < matching_net.size(); ++i) {
    cout << "Begin set up layer " << i << endl;
    matching_net[i]->PropAll();
    cout << "\tPropAll" << endl;
    matching_net[i]->SetupLayer(setting_vec[i], bottom_vecs[i], top_vecs[i], &rnd);
    cout << "\tSetup Layer" << endl;
    matching_net[i]->Reshape(bottom_vecs[i], top_vecs[i]);
    cout << "\tReshape" << endl;
  }
  matching_net_test[0]->PropAll();
  matching_net_test[0]->SetupLayer(setting_vec[setting_vec.size()-1], bottom_vecs[0], top_vecs[0], &rnd);
  matching_net_test[0]->Reshape(bottom_vecs[0], top_vecs[0]);
  
  // Save Initial Model
  {
  ofstream _of("model/matching0.model");
  Json::StyledWriter writer;
  Json::Value net_root;
  net_root["net_name"] = "matching_net";
  Json::Value layers_root;
  for (int i = 0; i < matching_net.size(); ++i) {
      Json::Value layer_root;
      matching_net[i]->SaveModel(layer_root);
      layers_root.append(layer_root);
  }
  net_root["layers"] = layers_root;
  string json_file = writer.write(net_root);
  _of << json_file;
  _of.close();
  }

  // Begin Training 
  int max_iters = 10000;
  for (int iter = 0; iter < max_iters; ++iter) {
    cout << "Begin iter " << iter << endl;
    for (int i = 0; i < matching_net.size(); ++i) {
      //cout << "Forward layer " << i << endl;
      matching_net[i]->Forward(bottom_vecs[i], top_vecs[i]);
    }
    
    for (int i = 0; i < nodes.size(); ++i) {
      cout << "# Data " << nodes[i]->node_name << " : ";
      for (int j = 0; j < 5; ++j) {
          cout << nodes[i]->data[0][0][0][j] << "\t";
      }
      cout << endl;
      cout << "# Diff " << nodes[i]->node_name << " : ";
      for (int j = 0; j < 5; ++j) {
          cout << nodes[i]->diff[0][0][0][j] << "\t";
      }
      cout << endl;
    }

    for (int i = matching_net.size()-1; i >= 0; --i) {
      //cout << "Backprop layer " << i << endl;
      matching_net[i]->Backprop(bottom_vecs[i], top_vecs[i]);
    }
    for (int i = 0; i < matching_net.size(); ++i) {
      for (int j = 0; j < matching_net[i]->ParamNodeNum(); ++j) {
        //cout << "Update param in layer " << i << " params " << j << endl;
        cout << "param data" << i << " , " << j << ": " << matching_net[i]->GetParams()[j].data[0][0][0][0] << endl;
        cout << "param diff" << i << " , " << j << ": " << matching_net[i]->GetParams()[j].diff[0][0][0][0] << endl;
        matching_net[i]->GetParams()[j].Update();
        cout << "param data" << i << " , " << j << ": " << matching_net[i]->GetParams()[j].data[0][0][0][0] << endl<<endl;
      }
    }
    
    // Output informations
    cout << "###### Iter " << iter << ": error = " << nodes[20]->data_d1()[0] << endl;
    
    if (iter % 100 == 0) {
      float loss = 0.0;
      int max_test_iter = 34;
      for (int test_iter = 0; test_iter < max_test_iter; ++test_iter) {
        for (int i = 0; i < matching_net.size(); ++i) {
          matching_net_test[i]->Forward(bottom_vecs[i], top_vecs[i]);
        }
        loss += nodes[20]->data_d1()[0];
      }
      loss /= max_test_iter;
      cout << endl << "****** Test error = " << loss << endl;
    }
  }
  
  // Save Initial Model
  {
  ofstream _of("model/matching1.model");
  Json::StyledWriter writer;
  Json::Value net_root;
  net_root["net_name"] = "matching_net";
  Json::Value layers_root;
  for (int i = 0; i < matching_net.size(); ++i) {
      Json::Value layer_root;
      matching_net[i]->SaveModel(layer_root);
      layers_root.append(layer_root);
  }
  net_root["layers"] = layers_root;
  string json_file = writer.write(net_root);
  _of << json_file;
  _of.close();
  }
  return 0;
}

