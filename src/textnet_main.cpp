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

void run_one(Json::Value &cfg_root, int netTagType) {
  DeviceType device_type = CPU_DEVICE;
  INet* net = CreateNet(device_type, netTagType);
  if (cfg_root["layers_params"].isNull()) { // new model
    net->InitNet(cfg_root);
  } else {
    net->LoadModel(cfg_root);
  }
#if REALTIME_SERVER==1
    using namespace textnet::statistic;
    Statistic statistic;
    statistic.SetNet(net);
    statistic.Start();
#endif
  net->Start();
}

void run_cv(Json::Value &cfg_root, int netTagType, int cv_fold) {
  vector<int> data_file_layer_idx;
  if (netTagType == kTrainValidTest) {
    data_file_layer_idx.push_back(0);
    data_file_layer_idx.push_back(1);
    data_file_layer_idx.push_back(2);
  } else {
      mshadow::utils::Check(false, "CV: need to set cv data file layer idx.");
  }

  for (int i = 0; i < cv_fold; ++i) {
    cout << "PROCESSING CROSS VALIDATION FOLD " << i << "..." << endl;
    stringstream ss;
    ss << i;
    Json::Value cv_cfg = cfg_root;
    for (size_t j = 0; j < data_file_layer_idx.size(); ++j) {
      int layer_idx = data_file_layer_idx[j];
      if (cv_cfg["layers"][layer_idx]["setting"]["data_file"].isNull()) {
        mshadow::utils::Check(false, "CV: no data file section in this layer.");
      }
      string data_file = cfg_root["layers"][layer_idx]["setting"]["data_file"].asString();
      data_file += "."+ss.str();
      cv_cfg["layers"][layer_idx]["setting"]["data_file"] = data_file;
    }
    run_one(cv_cfg, netTagType);
  }
}

float SIGMOID_MAX_INPUT = 20.f;
int SIGMOID_TABLE_SIZE = 1000000;
float *p_sigmoid_lookup_table = NULL;

float TANH_MAX_INPUT = 20.f;
int TANH_TABLE_SIZE = 1000000;
float *p_tanh_lookup_table = NULL;

int main(int argc, char *argv[]) {
  string model_file = "model/matching.tvt.model";
  bool need_cross_valid = false;
  // bool need_param = false;
  if (argc > 1) {
    model_file = string(argv[1]);
  }
  float max = SIGMOID_MAX_INPUT;
  int len = SIGMOID_TABLE_SIZE;
  p_sigmoid_lookup_table = new float[len];
  for (int i = 0; i < len; ++i) {
    float exp_val = exp((float(i)*2*max)/len - max); // map position to value, frow small to large
    p_sigmoid_lookup_table[i] = exp_val/(exp_val+1.f);
  }
  max = TANH_MAX_INPUT;
  len = TANH_TABLE_SIZE;
  p_tanh_lookup_table = new float[len];
  for (int i = 0; i < len; ++i) {
    float val = (float(i)*2.f*max)/len - max; // map position to value, frow small to large
    p_tanh_lookup_table[i] = tanhf(val);
  }

  /*
  for (int i = 2; i < argc; ++i) {
	if (string(argv[i]) == "cpu") {
		device_type = CPU_DEVICE;
	}
	if (string(argv[i]) == "gpu") {
		device_type = GPU_DEVICE;
	}
	if (string(argv[i]) == "-cv") {
		need_cross_valid = true;
	}
	if (string(argv[i]) == "-param") {
		need_param = true;
	}
  }*/
  Json::Value net_root;
  ifstream ifs(model_file.c_str());
  ifs >> net_root;
  ifs.close();

  if (net_root["layers_params"].isNull() && !net_root["cross_validation"].isNull()) { 
    need_cross_valid = true;
  }

  int netTagType = kTrainValidTest;
  //int netTagType = kTestOnly;
  if (!need_cross_valid) {
    run_one(net_root, netTagType);
  } else {
    int n_fold = net_root["cross_validation"].asInt();
    run_cv(net_root, netTagType, n_fold);
  }
  delete [] p_sigmoid_lookup_table;
  delete [] p_tanh_lookup_table;
}

