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
#include <sstream>

#include "./net/net.h"
#include "./net/test_net-inl.hpp"
#include "./layer/layer.h"
#include "./statistic/statistic.h"
#include "./io/json/json.h"
#include "global.h"

using namespace std;
using namespace textnet;
using namespace textnet::layer;
using namespace mshadow;
using namespace textnet::net;

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

float SIGMOID_MAX_INPUT = 20.f;
int SIGMOID_TABLE_SIZE = 1000000;
float *p_sigmoid_lookup_table = NULL;

float TANH_MAX_INPUT = 20.f;
int TANH_TABLE_SIZE = 1000000;
float *p_tanh_lookup_table = NULL;

int main(int argc, char *argv[]) {
  string model_file = "";
  string test_config_file = "";

  if (argc > 2) {
    model_file = string(argv[1]);
	test_config_file = string(argv[2]);
  }

  Json::Value test_config;
  ifstream _if(test_config_file.c_str());
  _if >> test_config;
  _if.close();

  int per_file_iter = test_config["per_file_iter"].asInt();
  int max_iter = test_config["max_iter"].asInt();
  vector<string> node_names;
  Json::Value node_names_root = test_config["node_names"];
  for (int i = 0; i < (int)node_names_root.size(); i++) {
	node_names.push_back(node_names_root[i].asString());
  }
  string file_prefix = test_config["file_prefix"].asString();
  string tag = test_config["tag"].asString();
 
  TestNet<cpu> net = TestNet<cpu>(per_file_iter, max_iter, node_names, file_prefix, tag);
  net.LoadModel(model_file);
  net.Start();

  return 0;
}

