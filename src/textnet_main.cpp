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

int main(int argc, char *argv[]) {
  string model_file = "model/matching.tvt.model";
  DeviceType device_type = CPU_DEVICE;
  if (argc > 1) {
    model_file = string(argv[1]);
  }
  if (argc > 2) {
	if (string(argv[2]) == "cpu") {
		device_type = CPU_DEVICE;
	}
	if (string(argv[2]) == "gpu") {
		device_type = GPU_DEVICE;
	}
  }
  Json::Value cfg;
  ifstream ifs(model_file.c_str());
  ifs >> cfg;
  string log_file = cfg["log"].asString();
  if (!log_file.empty()) {
      freopen(log_file.c_str(), "w", stdout);
      setvbuf(stdout, NULL, _IOLBF, 0);
  }

  int netTagType = kTrainValidTest;

  if (cfg["cross_validation"].isNull()) {
    INet* net = CreateNet(device_type, kTrainValidTest);
    net->InitNet(model_file);
#if REALTIME_SERVER==1
    using namespace textnet::statistic;
    Statistic statistic;
    statistic.SetNet(net);
    statistic.Start();
#endif
    net->Start();
  } else {
    int n_fold = cfg["cross_validation"].asInt();
    for (int i = 0; i < n_fold; ++i) {
      cout << "PROCESSING CROSS VALIDATION FOLD " << i << "..." << endl;
      stringstream ss;
      ss << i;
      Json::Value net_cfg = cfg;
      string data_file = cfg["layers"][0]["settings"]["data_file"].asString();
      data_file += "."+ss.str();
      net_cfg["layers"][0]["settings"]["data_file"] = data_file;
      data_file = cfg["layers"][1]["settings"]["data_file"].asString();
      data_file += "."+ss.str();
      net_cfg["layers"][1]["settings"]["data_file"] = data_file;
      data_file = cfg["layers"][2]["settings"]["data_file"].asString();
      data_file += "."+ss.str();
      net_cfg["layers"][2]["settings"]["data_file"] = data_file;

      INet* net = CreateNet(device_type, netTagType);
      net->InitNet(net_cfg);
      net->Start();
    }
  }

  ifs.close();
  
  return 0;
}

