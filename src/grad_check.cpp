#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <ctime>
#include <string>
#include <cstring>
#include <vector>
#include <map>
#include <climits>

#include "./layer/layer.h"
#include "./checker/checker.h"
#include "global.h"

// orc for read interal variable in layer classes
#include "./layer/common/convolutional_lstm_layer-inl.hpp"

using namespace std;
using namespace textnet;
using namespace textnet::layer;
using namespace mshadow;

template<int dim> 
void FillTensor(Tensor<cpu, dim> x, vector<float> v) {
	for (int i = 0; i < v.size(); ++i) {
		x.dptr_[i] = v[i];
	}
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
    cout << endl;
}
void TestSwapAxisLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check SwapAxis Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.data.Resize(Shape4(2,1,5,5), 1.0);
  bottom.diff.Resize(Shape4(2,1,5,5), 0.0);
  
  map<string, SettingV> setting;
  setting["axis1"] = SettingV(1);
  setting["axis2"] = SettingV(3);
  
  /// Test Activation Layer
  Layer<cpu> * layer_swap = CreateLayer<cpu>(kSwapAxis);
  layer_swap->PropAll();
  layer_swap->SetupLayer(setting, bottoms, tops, prnd);
  layer_swap->Reshape(bottoms, tops);
  layer_swap->Forward(bottoms, tops);

  PrintTensor("bottom", bottom.data);
  PrintTensor("top", top.data);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.1f);
  setting_checker["range_max"] = SettingV(0.1f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cker->CheckError(layer_swap, bottoms, tops);
}

void TestFlattenLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Flatten Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.data.Resize(Shape4(2,1,5,5), 1.0);
  bottom.diff.Resize(Shape4(2,1,5,5), 0.0);
 
  map<string, SettingV> setting;
  setting["axis1"] = SettingV(1);
  setting["axis2"] = SettingV(3);
  
  /// Test Activation Layer
  Layer<cpu> * layer_flat = CreateLayer<cpu>(kFlatten);
  layer_flat->PropAll();
  layer_flat->SetupLayer(setting, bottoms, tops, prnd);
  layer_flat->Reshape(bottoms, tops);
  layer_flat->Forward(bottoms, tops);

  PrintTensor("bottom", bottom.data);
  PrintTensor("top", top.data);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.1f);
  setting_checker["range_max"] = SettingV(0.1f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cker->CheckError(layer_flat, bottoms, tops);
}

void TestHingeLossLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check HingeLoss Layer." << endl;
  Node<cpu> bottom0;
  Node<cpu> bottom1;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(4,1,1,1));
  bottom1.Resize(Shape4(4,1,1,1));

  bottom0.data[0][0][0][0] = 0.2;
  bottom0.data[1][0][0][0] = 0.3;
  bottom0.data[2][0][0][0] = 1.5;
  bottom0.data[3][0][0][0] = -0.5;

  bottom1.data[0][0][0][0] = 1;
  bottom1.data[1][0][0][0] = 0;
  bottom1.data[2][0][0][0] = 1;
  bottom1.data[3][0][0][0] = 0;
  
  map<string, SettingV> setting;
  setting["delta"] = SettingV(1.0f);
  
  /// Test Activation Layer
  Layer<cpu> * layer_hingeloss = CreateLayer<cpu>(kHingeLoss);
  layer_hingeloss->PropAll();
  layer_hingeloss->SetupLayer(setting, bottoms, tops, prnd);
  layer_hingeloss->Reshape(bottoms, tops);
  layer_hingeloss->Forward(bottoms, tops);
  layer_hingeloss->Backprop(bottoms, tops);
  
  PrintTensor("top data", top.data);
  PrintTensor("bottom diff", bottom0.diff);
}

void TestListwiseMeasureLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check ListwiseMeasure Layer." << endl;
  Node<cpu> bottom0;
  Node<cpu> bottom1;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(6,1,1,1));
  bottom1.Resize(Shape4(6,1,1,1));

  bottom0.data[0][0][0][0] = 2.2;
  bottom0.data[1][0][0][0] = 1.3;
  bottom0.data[2][0][0][0] = 1.0;
  bottom0.data[3][0][0][0] = -0.5;
  bottom0.data[4][0][0][0] = -1.5;
  bottom0.data[5][0][0][0] = -2.5;

  bottom1.data[0][0][0][0] = 3;
  bottom1.data[1][0][0][0] = 2;
  bottom1.data[2][0][0][0] = 3;
  bottom1.data[3][0][0][0] = 0;
  bottom1.data[4][0][0][0] = 1;
  bottom1.data[5][0][0][0] = 2;
  
  map<string, SettingV> setting;
  setting["k"] = SettingV(6);
  setting["method"] = SettingV("nDCG@k");
  // setting["method"] = SettingV("MRR");
  // setting["method"] = SettingV("P@k");
  
  /// Test Activation Layer
  Layer<cpu> * layer_listwise = CreateLayer<cpu>(kListwiseMeasure);
  layer_listwise->PropAll();
  layer_listwise->SetupLayer(setting, bottoms, tops, prnd);
  layer_listwise->Reshape(bottoms, tops);
  layer_listwise->Forward(bottoms, tops);
  layer_listwise->Backprop(bottoms, tops);
  
  PrintTensor("top data", top.data);
  PrintTensor("bottom diff", bottom0.diff);
}

void TestMatchWeightedDotLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Match Weightd Dot Layer." << endl;
  Node<cpu> bottom1;
  Node<cpu> bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);

  bottom1.Resize(2, 1, 5, 3);
  bottom2.Resize(2, 1, 5, 3);

  prnd->SampleUniform(&bottom1.data, -1.0, 1.0);
  prnd->SampleUniform(&bottom2.data, -1.0, 1.0);

  bottom1.length = 5;
  bottom2.length = 5;

  map<string, SettingV> setting;
  {
    setting["d_hidden"] = SettingV(2);
      
    map<string, SettingV> &t_filler = *(new map<string, SettingV>());
      t_filler["init_type"] = SettingV(initializer::kUniform);
      t_filler["range"] = SettingV(0.01f);
    setting["t_filler"] = SettingV(&t_filler);
    setting["w_filler"] = SettingV(&t_filler);
    setting["b_filler"] = SettingV(&t_filler);

      
    map<string, SettingV> &t_updater = *(new map<string, SettingV>());
      t_updater["updater_type"] = SettingV(updater::kAdagrad);
      t_updater["eps"] = SettingV(0.01f);
      t_updater["batch_size"] = SettingV(1);
      t_updater["max_iter"] = SettingV(10000);
      t_updater["lr"] = SettingV(0.1f);
    setting["t_updater"] = SettingV(&t_updater);
    setting["w_updater"] = SettingV(&t_updater);
    setting["b_updater"] = SettingV(&t_updater);
  }

  // Test Match Layer
  Layer<cpu> * layer_match = CreateLayer<cpu>(kMatchWeightedDot);
  layer_match->PropAll();
  layer_match->SetupLayer(setting, bottoms, tops, prnd);
  layer_match->Reshape(bottoms, tops);

  layer_match->Forward(bottoms, tops);
  PrintTensor("bottom1", bottom1.data);
  PrintTensor("bottom2", bottom2.data);
  PrintTensor("top", top.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_match, bottoms, tops);
}

void TestMatchTensorLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Match Tensor Layer." << endl;
  Node<cpu> bottom1;
  Node<cpu> bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);

  bottom1.Resize(2, 1, 5, 3);
  bottom2.Resize(2, 1, 5, 3);

  prnd->SampleUniform(&bottom1.data, -1.0, 1.0);
  prnd->SampleUniform(&bottom2.data, -1.0, 1.0);

  bottom1.length = 3;
  bottom2.length = 4;

  map<string, SettingV> setting;
  {
    setting["d_hidden"] = SettingV(2);
    setting["d_factor"] = SettingV(2);
    setting["is_use_linear"] = SettingV(true);
    // setting["t_l2"] = SettingV(0);
    // setting["is_init_as_I"] = SettingV(false);
      
    map<string, SettingV> &t_filler = *(new map<string, SettingV>());
      t_filler["init_type"] = SettingV(initializer::kUniform);
      t_filler["range"] = SettingV(0.01f);
    setting["t_filler"] = SettingV(&t_filler);
    setting["w_filler"] = SettingV(&t_filler);
    setting["b_filler"] = SettingV(&t_filler);

      
    map<string, SettingV> &t_updater = *(new map<string, SettingV>());
      t_updater["updater_type"] = SettingV(updater::kAdagrad);
      t_updater["eps"] = SettingV(0.01f);
      t_updater["batch_size"] = SettingV(1);
      t_updater["max_iter"] = SettingV(10000);
      t_updater["lr"] = SettingV(0.1f);
    setting["t_updater"] = SettingV(&t_updater);
    setting["w_updater"] = SettingV(&t_updater);
    setting["b_updater"] = SettingV(&t_updater);
  }

  // Test Match Layer
  Layer<cpu> * layer_match = CreateLayer<cpu>(kMatchTensor);
  layer_match->PropAll();
  layer_match->SetupLayer(setting, bottoms, tops, prnd);
  layer_match->Reshape(bottoms, tops);

  layer_match->Forward(bottoms, tops);
  layer_match->Backprop(bottoms, tops);
  layer_match->Forward(bottoms, tops);
  layer_match->Backprop(bottoms, tops);
  PrintTensor("t_data", layer_match->params[0].data);
  PrintTensor("w_data", layer_match->params[1].data);
  PrintTensor("bottom1", bottom1.data);
  PrintTensor("bottom2", bottom2.data);
  PrintTensor("top", top.data);
  // exit(0);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);

  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_match, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_match, bottoms, tops);
}



void TestMatchTensorFactLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Match Tensor Fact Layer." << endl;
  Node<cpu> bottom1;
  Node<cpu> bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);

  bottom1.Resize(2, 1, 5, 3);
  bottom2.Resize(2, 1, 5, 3);

  prnd->SampleUniform(&bottom1.data, -1.0, 1.0);
  prnd->SampleUniform(&bottom2.data, -1.0, 1.0);

  bottom1.length = 5;
  bottom2.length = 5;

  map<string, SettingV> setting;
  {
    setting["d_hidden"] = SettingV(2);
    setting["d_factor"] = SettingV(2);
    setting["t_l2"] = SettingV(0);
    setting["is_init_as_I"] = SettingV(false);
      
    map<string, SettingV> &t_filler = *(new map<string, SettingV>());
      t_filler["init_type"] = SettingV(initializer::kUniform);
      t_filler["range"] = SettingV(0.01f);
    setting["t_filler"] = SettingV(&t_filler);
    setting["w_filler"] = SettingV(&t_filler);
    setting["b_filler"] = SettingV(&t_filler);

      
    map<string, SettingV> &t_updater = *(new map<string, SettingV>());
      t_updater["updater_type"] = SettingV(updater::kAdagrad);
      t_updater["eps"] = SettingV(0.01f);
      t_updater["batch_size"] = SettingV(1);
      t_updater["max_iter"] = SettingV(10000);
      t_updater["lr"] = SettingV(0.1f);
    setting["t_updater"] = SettingV(&t_updater);
    setting["w_updater"] = SettingV(&t_updater);
    setting["b_updater"] = SettingV(&t_updater);
  }

  // Test Match Layer
  Layer<cpu> * layer_match = CreateLayer<cpu>(kMatchTensorFact);
  layer_match->PropAll();
  layer_match->SetupLayer(setting, bottoms, tops, prnd);
  layer_match->Reshape(bottoms, tops);

  // layer_match->Forward(bottoms, tops);
  // PrintTensor("bottom1", bottom1.data);
  // PrintTensor("bottom2", bottom2.data);
  // PrintTensor("top", top.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);

  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_match, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_match, bottoms, tops);
}


void TestMatchLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Match Layer." << endl;
  Node<cpu> bottom1;
  Node<cpu> bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);

  bottom1.Resize(2, 1, 5, 2);
  bottom2.Resize(2, 1, 5, 2);

  prnd->SampleUniform(&bottom1.data, -1.0, 1.0);
  prnd->SampleUniform(&bottom2.data, -1.0, 1.0);

  bottom1.length = 5;
  bottom2.length = 5;

  map<string, SettingV> setting;
  setting["op"] = SettingV("cos");

  // Test Match Layer
  Layer<cpu> * layer_match = CreateLayer<cpu>(kMatch);
  layer_match->PropAll();
  layer_match->SetupLayer(setting, bottoms, tops, prnd);
  layer_match->Reshape(bottoms, tops);

  layer_match->Forward(bottoms, tops);
  PrintTensor("bottom1", bottom1.data);
  PrintTensor("bottom2", bottom2.data);
  PrintTensor("top", top.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_match, bottoms, tops);
}

void TestCrossLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Cross Layer." << endl;
  Node<cpu> bottom1;
  Node<cpu> bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);

  bottom1.Resize(2, 1, 5, 1);
  bottom2.Resize(2, 1, 5, 1);

  prnd->SampleUniform(&bottom1.data, -1.0, 1.0);
  prnd->SampleUniform(&bottom2.data, -1.0, 1.0);

  map<string, SettingV> setting;
  // Test Cross Layer
  Layer<cpu> * layer_cross = CreateLayer<cpu>(kCross);
  layer_cross->PropAll();
  layer_cross->SetupLayer(setting, bottoms, tops, prnd);
  layer_cross->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.01f);
  setting_checker["range_max"] = SettingV(0.01f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cker->CheckError(layer_cross, bottoms, tops);
}

void TestDropoutLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Dropout Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.data.Resize(Shape4(2,1,5,5), 1.0);
  bottom.diff.Resize(Shape4(2,1,5,5), 0.0);
  
  map<string, SettingV> setting;
  setting["rate"] = SettingV(0.5f);
  
  /// Test Activation Layer
  Layer<cpu> * layer_dropout = CreateLayer<cpu>(kDropout);
  layer_dropout->PropAll();
  layer_dropout->SetupLayer(setting, bottoms, tops, prnd);
  layer_dropout->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.1f);
  setting_checker["range_max"] = SettingV(0.1f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cker->CheckError(layer_dropout, bottoms, tops);
}

void TestConvLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Conv Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,5,5), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  setting["kernel_x"] = SettingV(3);
  setting["kernel_y"] = SettingV(3);
  setting["pad_x"] = SettingV(1);
  setting["pad_y"] = SettingV(1);
  setting["stride"] = SettingV(1);
  setting["channel_out"] = SettingV(2);
  setting["no_bias"] = SettingV(false);
    map<string, SettingV> w_setting;
    w_setting["init_type"] = SettingV(initializer::kGaussian);
    w_setting["mu"] = SettingV(0.0f);
    w_setting["sigma"] = SettingV(1.0f);
    map<string, SettingV> b_setting;
    b_setting["init_type"] = SettingV(initializer::kZero);
  setting["w_filler"] = SettingV(&w_setting);
  setting["b_filler"] = SettingV(&b_setting);
    map<string, SettingV> w_updater;
    w_updater["init_type"] = SettingV(updater::kSGD);
    w_updater["momentum"] = SettingV(0.0f);
    w_updater["lr"] = SettingV(0.001f);
    w_updater["decay"] = SettingV(0.001f);
    map<string, SettingV> b_updater;
    b_updater["init_type"] = SettingV(updater::kSGD);
    b_updater["momentum"] = SettingV(0.0f);
    b_updater["lr"] = SettingV(0.001f);
    b_updater["decay"] = SettingV(0.001f);
  setting["w_updater"] = SettingV(&w_updater);
  setting["b_updater"] = SettingV(&b_updater);
  
  /// Test Activation Layer
  Layer<cpu> * layer_conv = CreateLayer<cpu>(kConv);
  layer_conv->PropAll();
  layer_conv->SetupLayer(setting, bottoms, tops, prnd);
  layer_conv->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.01f);
  setting_checker["range_max"] = SettingV(0.01f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer_conv, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_conv, bottoms, tops);

}

void TestConvVarLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check ConvVar Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(1,1,5,5), true);

  float bottom_data_[] = {1,1,1,1,1,
						 1,1,1,1,1,
						 1,1,1,1,1,
						 1,1,1,1,1,
						 1,1,1,1,1};
  vector<float> bottom_data(bottom_data_, bottom_data_ + sizeof(bottom_data_) / sizeof(float));
  FillTensor(bottom.data, bottom_data);
  
  float bottom_len_[] = {2,3};
  vector<float> bottom_len(bottom_len_, bottom_len_ + sizeof(bottom_len_) / sizeof(float));
  FillTensor(bottom.length, bottom_len);
  //prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  setting["kernel_x"] = SettingV(5);
  setting["kernel_y"] = SettingV(3);
  setting["pad_x"] = SettingV(0);
  setting["pad_y"] = SettingV(1);
  setting["stride_x"] = SettingV(1);
  setting["stride_y"] = SettingV(1);
  setting["channel_out"] = SettingV(2);
  setting["dim"] = SettingV(1);
  setting["no_bias"] = SettingV(false);
    map<string, SettingV> w_setting;
    w_setting["init_type"] = SettingV(initializer::kConstant);
    w_setting["value"] = SettingV(0.1f);
    map<string, SettingV> b_setting;
    b_setting["init_type"] = SettingV(initializer::kConstant);
	b_setting["value"] = SettingV(0.1f);
  setting["w_filler"] = SettingV(&w_setting);
  setting["b_filler"] = SettingV(&b_setting);
    map<string, SettingV> w_updater;
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    map<string, SettingV> b_updater;
      b_updater["updater_type"] = SettingV(updater::kAdagrad);
      b_updater["eps"] = SettingV(0.01f);
      b_updater["batch_size"] = SettingV(1);
      b_updater["max_iter"] = SettingV(10000);
      b_updater["lr"] = SettingV(0.1f);
  setting["w_updater"] = SettingV(&w_updater);
  setting["b_updater"] = SettingV(&b_updater);
  
  /// Test Activation Layer
  Layer<cpu> * layer_conv = CreateLayer<cpu>(kConvVar);
  layer_conv->PropAll();
  layer_conv->SetupLayer(setting, bottoms, tops, prnd);
  layer_conv->Reshape(bottoms, tops, true);

  layer_conv->Forward(bottoms, tops);
  PrintTensor("bottom", bottom.data);
  PrintTensor("top", top.data);
  PrintTensor("weight", layer_conv->params[0].data);
  PrintTensor("bias", layer_conv->params[1].data);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.01f);
  setting_checker["range_max"] = SettingV(0.01f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer_conv, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_conv, bottoms, tops);

}

void TestLocalLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Local Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(1,1,5,5), true);

  float bottom_data_[] = {1,1,1,1,1,
						 1,1,1,1,1,
						 1,1,1,1,1,
						 1,1,1,1,1,
						 1,1,1,1,1};
  vector<float> bottom_data(bottom_data_, bottom_data_ + sizeof(bottom_data_) / sizeof(float));
  FillTensor(bottom.data, bottom_data);
  
  float bottom_len_[] = {2,3};
  vector<float> bottom_len(bottom_len_, bottom_len_ + sizeof(bottom_len_) / sizeof(float));
  FillTensor(bottom.length, bottom_len);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  setting["kernel_x"] = SettingV(3);
  setting["kernel_y"] = SettingV(3);
  setting["pad_x"] = SettingV(1);
  setting["pad_y"] = SettingV(1);
  setting["stride_x"] = SettingV(1);
  setting["stride_y"] = SettingV(1);
  setting["channel_out"] = SettingV(2);
  setting["dim"] = SettingV(2);
  setting["no_bias"] = SettingV(false);
    map<string, SettingV> w_setting;
    w_setting["init_type"] = SettingV(initializer::kGaussian);
    w_setting["mu"] = SettingV(0.0f);
    w_setting["sigma"] = SettingV(1.0f);
    //w_setting["init_type"] = SettingV(initializer::kConstant);
    //w_setting["value"] = SettingV(0.1f);
    map<string, SettingV> b_setting;
    b_setting["init_type"] = SettingV(initializer::kConstant);
	b_setting["value"] = SettingV(0.1f);
  setting["w_filler"] = SettingV(&w_setting);
  setting["b_filler"] = SettingV(&b_setting);
    map<string, SettingV> w_updater;
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    map<string, SettingV> b_updater;
      b_updater["updater_type"] = SettingV(updater::kAdagrad);
      b_updater["eps"] = SettingV(0.01f);
      b_updater["batch_size"] = SettingV(1);
      b_updater["max_iter"] = SettingV(10000);
      b_updater["lr"] = SettingV(0.1f);
  setting["w_updater"] = SettingV(&w_updater);
  setting["b_updater"] = SettingV(&b_updater);
  
  /// Test Activation Layer
  Layer<cpu> * layer_conv = CreateLayer<cpu>(kLocal);
  layer_conv->PropAll();
  layer_conv->SetupLayer(setting, bottoms, tops, prnd);
  layer_conv->Reshape(bottoms, tops, true);

  layer_conv->Forward(bottoms, tops);
  PrintTensor("bottom", bottom.data);
  PrintTensor("top", top.data);
  PrintTensor("weight", layer_conv->params[0].data);
  PrintTensor("bias", layer_conv->params[1].data);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.01f);
  setting_checker["range_max"] = SettingV(0.01f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer_conv, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_conv, bottoms, tops);
  
}

void TestPoolLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Pool Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.data.Resize(Shape4(2,1,5,5), 1.0);
  bottom.diff.Resize(Shape4(2,1,5,5), 0.0);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  setting["kernel_x"] = SettingV(2);
  setting["kernel_y"] = SettingV(2);
  setting["stride"] = SettingV(2);
  
  /// Test MaxPooling Layer
  Layer<cpu> * layer_pool = CreateLayer<cpu>(kMaxPooling);
  layer_pool->PropAll();
  layer_pool->SetupLayer(setting, bottoms, tops, prnd);
  layer_pool->Reshape(bottoms, tops);
  layer_pool->Forward(bottoms, tops);
  prnd->SampleUniform(&top.diff, -1.0, 1.0);
  layer_pool->Backprop(bottoms, tops);
  
  PrintTensor("bottom data", bottom.data);
  PrintTensor("top data", top.data);
  PrintTensor("bottom diff", bottom.diff);
  
  /// Test MaxPooling Layer
  Layer<cpu> * layer_pool1 = CreateLayer<cpu>(kAvgPooling);
  layer_pool1->PropAll();
  layer_pool1->SetupLayer(setting, bottoms, tops, prnd);
  layer_pool1->Reshape(bottoms, tops);
  layer_pool1->Forward(bottoms, tops);
  prnd->SampleUniform(&top.diff, -1.0, 1.0);
  layer_pool1->Backprop(bottoms, tops);
  
  PrintTensor("bottom data", bottom.data);
  PrintTensor("top data", top.data);
  PrintTensor("bottom diff", bottom.diff);


}

void TestSequenceDimReductionLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Sequence Dim Reduction Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,5,4), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  {
    setting["num_hidden"] = SettingV(3);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_filler);

      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
  }

  
  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kSequenceDimReduction);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer, bottoms, tops);
}
void TestTensorLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Lstm Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,1,10), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  {
    setting["num_hidden"] = SettingV(3);
    setting["mode"] = SettingV("t1w1b0");
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.01f);
    setting["t_filler"] = SettingV(&w_filler);
    setting["w_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["t_updater"] = SettingV(&w_updater);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
  }

  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kTensorFullConnect);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer, bottoms, tops);
}

void TestMaxRnnLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check MaxRnn Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,10,5), true);
  bottom.length = 10;
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  {
    setting["d_mem"] = SettingV(3);
    setting["no_bias"] = SettingV(true);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_filler);
    setting["u_filler"] = SettingV(&w_filler);
    setting["t_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
    setting["u_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
    setting["t_updater"] = SettingV(&w_updater);
  }

  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kMaxRecurrent);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer, bottoms, tops);
}

void TestTopkPoolingLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Gate Layer." << endl;
  Node<cpu> gate, rep, top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&gate);
  bottoms.push_back(&rep);
  tops.push_back(&top);
  
  gate.Resize(Shape4(2,1,4,1), true);
  rep.Resize(Shape4(2,1,4,2), true);
  prnd->SampleUniform(&gate.data, -1.0, 1.0);
  prnd->SampleUniform(&rep.data,  -1.0, 1.0);
  gate.length = 3;
  rep.length = 3;
  
  map<string, SettingV> setting;
  setting["k"] = SettingV(2);

  Layer<cpu> * layer = CreateLayer<cpu>(kTopkPooling);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // prnd->SampleUniform(&ttom.data, -0.1, 0.1);
  // prnd->SampleUniform(&top.diff, -0.1, 0.1);
  // top.diff = gate.data;
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestSoftmaxVarLenFuncLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check SoftmaxVarLenFunc Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,2,10,4), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  PrintTensor("b0_data", bottom.data);
  bottom.length = 6;
  
  map<string, SettingV> setting;

  Layer<cpu> * layer = CreateLayer<cpu>(kSoftmaxFuncVarLen);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  // prnd->SampleUniform(&top.diff, -10.0, 10.0);
  // top.diff = bottom.data;
  layer->Forward(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("t_data",  tops[0]->data);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.00001f);
  setting_checker["range_max"] = SettingV(0.00001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestSumLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Sum Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,2,5,4), true);
  prnd->SampleUniform(&bottom.data, -5.0, 5.0);
  prnd->SampleUniform(&top.diff, -5.0, 5.0);
  bottom.length = 5;
  
  map<string, SettingV> setting;
  setting["axis"] = SettingV(2);

  Layer<cpu> * layer = CreateLayer<cpu>(kSumByAxis);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);

  // top.diff = bottom.data;
  layer->Forward(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("t_data",  tops[0]->data);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestSoftmaxFuncLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check SoftmaxVarLenFunc Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,2,5,4), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  prnd->SampleUniform(&top.diff, -1.0, 1.0);
  bottom.length = 3;
  
  map<string, SettingV> setting;

  Layer<cpu> * layer = CreateLayer<cpu>(kSoftmaxFuncVarLen);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);

  // top.diff = bottom.data;
  layer->Forward(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("t_data",  tops[0]->data);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestWordClassSoftmaxLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Test Word Class Softmax Layer." << endl;
  Node<cpu> bottom0, bottom1;
  Node<cpu> top0, top1, top2, top3;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  tops.push_back(&top0);
  tops.push_back(&top1);
  tops.push_back(&top2);
  tops.push_back(&top3);
  
  bottom0.Resize(Shape4(1,1,1,4), true);
  bottom1.Resize(Shape4(1,1,1,1), true);
  prnd->SampleUniform(&bottom0.data, -1, 1);
  bottom1.data[0][0][0][0] = 1;
  // bottom1.data[1][0][0][0] = 3;
  
  map<string, SettingV> setting;
  {
    setting["word_class_file"] = "./tmp.wordclass";
    setting["feat_size"]  = 4;
    setting["vocab_size"] = 5;
    setting["class_num"]  = 2;

    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.1f);
    setting["w_class_filler"] = SettingV(&w_filler);
    setting["b_class_filler"] = SettingV(&w_filler);
    setting["w_word_filler"]  = SettingV(&w_filler);
    setting["b_word_filler"]  = SettingV(&w_filler);
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_class_updater"] = SettingV(&w_updater);
    setting["b_class_updater"] = SettingV(&w_updater);
    setting["w_word_updater"] = SettingV(&w_updater);
    setting["b_word_updater"] = SettingV(&w_updater);
  }
  Layer<cpu> *layer = CreateLayer<cpu>(kWordClassSoftmaxLoss);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);

  float eps = 0.001;

  layer->ClearDiff(bottoms, tops);
  layer->Forward(bottoms, tops);
  PrintTensor("top0.data", top0.data);
  PrintTensor("top1.data", top1.data);
  PrintTensor("top2.data", top2.data);
  PrintTensor("top3.data", top3.data);
  layer->Backprop(bottoms, tops);
  PrintTensor("bottom0.diff", bottom0.diff);
  PrintTensor("w_class.diff", layer->params[0].diff);
  PrintTensor("b_class.diff", layer->params[1].diff);
  PrintTensor("w_embed.diff", layer->params[2].diff);
  PrintTensor("b_embed.diff", layer->params[3].diff);

  float *p = &layer->params[3].data[0][0][0][1];
  layer->ClearDiff(bottoms, tops);
  *p += eps;
  layer->Forward(bottoms, tops);
  float loss1 = top3.data[0][0][0][0] * 2; // batch_size
  *p -= 2 * eps;
  layer->Forward(bottoms, tops);
  float loss2 = top3.data[0][0][0][0] * 2; // batch_size

  cout << "loss 1:" << loss1 << endl;
  cout << "loss 2:" << loss2 << endl;
  float gradient = (loss1-loss2)/(2*eps);
  cout << "gradient by eps:" << gradient << endl;

  cout << "Done." << endl;
}

void TestPosPredRepLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check PosPredRepLayer." << endl;
  Node<cpu> bottom0, bottom1, bottom2;
  Node<cpu> top0;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top0);
  
  bottom0.Resize(Shape4(2,1,4,5), true);
  bottom1.Resize(Shape4(2,1,4,5), true);
  bottom0.length = 3;
  bottom1.length = 3;

  bottom2.Resize(Shape4(2,2,1,1), true);
  bottom2.data = 0;
  bottom2.data[0][0][0][0] = 0;
  bottom2.data[0][1][0][0] = 2;
  bottom2.data[1][0][0][0] = 1;
  bottom2.data[1][1][0][0] = 2;
  prnd->SampleUniform(&bottom0.data, -1, 1);
  prnd->SampleUniform(&bottom1.data, -1, 1);
  
  map<string, SettingV> setting;
  Layer<cpu> *layer = CreateLayer<cpu>(kPosPredRep);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // // top0.diff = top0.data;
  // PrintTensor("bottom0", bottom0.data);
  // PrintTensor("bottom1", bottom1.data);
  // PrintTensor("bottom2", bottom2.data);
  // PrintTensor("top0", top0.data);
  // PrintTensor("bottom0_diff", bottom0.diff);
  // PrintTensor("bottom1_diff", bottom1.diff);
  // PrintTensor("bottom2_diff", bottom2.diff);
  // PrintTensor("top0_diff", top0.diff);
  // exit(0);
  // PrintTensor("top_diff", top.diff);
  // PrintTensor("param_diff", layer->GetParams()[0].diff);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);
  // // PrintTensor("bottom0_diff", bottom0.diff);
  // // PrintTensor("bottom1_diff", bottom1.diff);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // PrintTensor("t_diff",  tops[0]->diff);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("t_diff",  tops[0]->diff);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("w_data", bottoms[0]->diff);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}



void TestNegativeSampleLossLayer(mshadow::Random<cpu>* prnd) {
  cout << "This layer can not be checked." << endl;
  cout << "G Check Negative Sample Loss Layer." << endl;
  Node<cpu> bottom0, bottom1, bottom2;
  Node<cpu> top0, top1;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top0);
  tops.push_back(&top1);
  
  bottom0.Resize(Shape4(2,3,1,5), true);
  bottom1.Resize(Shape4(2,3,4,6), true);
  bottom2.Resize(Shape4(2,3,4,1), true);
  bottom2.data = 0;
  bottom2.data[0][0][0][0] = 1;
  bottom2.data[0][1][0][0] = 1;
  bottom2.data[0][2][0][0] = 1;
  bottom2.data[1][0][0][0] = 1;
  bottom2.data[1][1][0][0] = 1;
  bottom2.data[1][2][0][0] = 1;
  prnd->SampleUniform(&bottom0.data, -1, 1);
  prnd->SampleUniform(&bottom1.data, -1, 1);
  
  map<string, SettingV> setting;
  Layer<cpu> *layer = CreateLayer<cpu>(kNegativeSampleLoss);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // // top0.diff = top0.data;
  // PrintTensor("bottom0", bottom0.data);
  // PrintTensor("bottom1", bottom1.data);
  // PrintTensor("bottom2", bottom2.data);
  // PrintTensor("top0", top0.data);
  // PrintTensor("top1", top1.data);
  // PrintTensor("bottom0_diff", bottom0.diff);
  // PrintTensor("bottom1_diff", bottom1.diff);
  // PrintTensor("bottom2_diff", bottom2.diff);
  // PrintTensor("top0_diff", top0.diff);
  // PrintTensor("top1_diff", top1.diff);
  // exit(0);
  // PrintTensor("top_diff", top.diff);
  // PrintTensor("param_diff", layer->GetParams()[0].diff);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);
  // // PrintTensor("bottom0_diff", bottom0.diff);
  // // PrintTensor("bottom1_diff", bottom1.diff);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // PrintTensor("t_diff",  tops[0]->diff);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("t_diff",  tops[0]->diff);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("w_data", bottoms[0]->diff);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}



void TestDiagRecurrentLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Diag Recurrent Layer." << endl;
  Node<cpu> bottom0, bottom1, bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(1,10,12, 5), true);
  bottom1.Resize(Shape4(1,1,10,2), true);
  bottom2.Resize(Shape4(1,1,12,2), true);
  bottom1.length = 4;
  bottom2.length = 5;
  prnd->SampleUniform(&bottom0.data, -0.1, 0.1);
  
  map<string, SettingV> setting;
  {
    setting["d_mem"] = 2;
    setting["reverse"] = false;
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(1.f);
    setting["w_filler"] = SettingV(&w_filler);
    setting["u_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
    setting["u_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
  }

  Layer<cpu> *layer = CreateLayer<cpu>(kDiagRecurrent);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // layer->Forward(bottoms, tops);
  // top.diff = top.data;
  // PrintTensor("bottom0", bottom0.data);
  // PrintTensor("bottom1", bottom1.data);
  // PrintTensor("top", top.data);
  // PrintTensor("top_diff", top.diff);
  // PrintTensor("param_diff", layer->GetParams()[0].diff);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);
  // // PrintTensor("bottom0_diff", bottom0.diff);
  // // PrintTensor("bottom1_diff", bottom1.diff);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // PrintTensor("t_diff",  tops[0]->diff);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("t_diff",  tops[0]->diff);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("w_data", bottoms[0]->diff);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestDynamicKMaxPoolingLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Product Layer." << endl;
  Node<cpu> bottom0;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom0);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(2,2,6,3), true);
  bottom0.length = 5;
  prnd->SampleUniform(&bottom0.data, -0.1, 0.1);
  
  map<string, SettingV> setting;
  {
      setting["L"] = 2;
      setting["l"] = 1;
      setting["max_sentence_length"] = 6;
      setting["min_rep_length"] = 3;
  }

  Layer<cpu> *layer = CreateLayer<cpu>(kDynamicKMaxPooling);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // layer->Forward(bottoms, tops);
  // top.diff = 1.f; // bottom.data;
  // PrintTensor("bottom0", bottom0.data);
  // PrintTensor("bottom1", bottom1.data);
  // PrintTensor("top", top.data);
  // PrintTensor("top_diff", top.diff);
  // PrintTensor("param_diff", layer->GetParams()[0].diff);
  
  // using namespace checker;
  // Checker<cpu> * cker = CreateChecker<cpu>();
  // map<string, SettingV> setting_checker;
  // setting_checker["range_min"] = SettingV(-0.0001f);
  // setting_checker["range_max"] = SettingV(0.0001f);
  // setting_checker["delta"] = SettingV(0.0001f);
  // cker->SetupChecker(setting_checker, prnd);
  // cout << "Check Error." << endl;
  // cker->CheckError(layer, bottoms, tops);
  // PrintTensor("bottom0_diff", bottom0.diff);
  // PrintTensor("bottom1_diff", bottom1.diff);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  layer->Forward(bottoms, tops);
  top.diff = 1.f; // bottom.data;
  layer->Backprop(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("t_data",  tops[0]->data);
  PrintTensor("b0_length", bottoms[0]->length);
  PrintTensor("t_length", tops[0]->length);
  PrintTensor("b0_diff", bottoms[0]->diff);
  PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestSelectSubRepByTokenLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check SelectSubRepByTokenLayer." << endl;
  Node<cpu> bottom0, bottom1;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(2,1,1,4), true);
  bottom1.Resize(Shape4(2,1,4,2), true);
  bottom0.length = 3;
  bottom1.length = 3;
  bottom0.data = 1;
  bottom0.data[0][0][0][0] = 0;
  bottom0.data[0][0][0][2] = 0;
  bottom0.data[1][0][0][1] = 0;
  prnd->SampleUniform(&bottom1.data, -0.1, 0.1);
  
  map<string, SettingV> setting;
  {
    setting["token"] = 0;
  }

  Layer<cpu> *layer = CreateLayer<cpu>(kSelectSubRepByToken);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  layer->Forward(bottoms, tops);
  top.diff = 1.f; // bottom.data;
  PrintTensor("bottom0", bottom0.data);
  PrintTensor("bottom1", bottom1.data);
  PrintTensor("top", top.data);
  PrintTensor("top", top.length);
  exit(0);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);
  // PrintTensor("bottom0_diff", bottom0.diff);
  // PrintTensor("bottom1_diff", bottom1.diff);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}
void TestMatchTopKPoolingLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check MatchTopKPoolingLayer." << endl;
  Node<cpu> bottom0, bottom1, bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(1,1,3,3), true);
  bottom1.Resize(Shape4(1,1,3,2), true);
  bottom2.Resize(Shape4(1,1,3,2), true);
  bottom1.length = 3;
  bottom2.length = 3;
  prnd->SampleUniform(&bottom0.data, -0.1, 0.1);
  
  map<string, SettingV> setting;
  {
    setting["k"] = 5;
  }

  Layer<cpu> *layer = CreateLayer<cpu>(kMatchTopKPooling);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  layer->Forward(bottoms, tops);
  top.diff = 1.f; // bottom.data;
  PrintTensor("bottom0", bottom0.data);
  PrintTensor("top", top.data);
  exit(0);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);
  // PrintTensor("bottom0_diff", bottom0.diff);
  // PrintTensor("bottom1_diff", bottom1.diff);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}



void TestDynamicPoolingLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Product Layer." << endl;
  Node<cpu> bottom0, bottom1, bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(2,2,20,30), true);
  bottom1.Resize(Shape4(2,2,20,2), true);
  bottom2.Resize(Shape4(2,2,30,2), true);
  bottom1.length = 4;
  bottom2.length = 5;
  prnd->SampleUniform(&bottom0.data, -0.1, 0.1);
  
  map<string, SettingV> setting;
  {
      setting["row"] = 10;
      setting["col"] = 25;
  }

  Layer<cpu> *layer = CreateLayer<cpu>(kDynamicPooling);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // layer->Forward(bottoms, tops);
  // top.diff = 1.f; // bottom.data;
  // PrintTensor("bottom0", bottom0.data);
  // PrintTensor("bottom1", bottom1.data);
  // PrintTensor("top", top.data);
  // PrintTensor("top_diff", top.diff);
  // PrintTensor("param_diff", layer->GetParams()[0].diff);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);
  // PrintTensor("bottom0_diff", bottom0.diff);
  // PrintTensor("bottom1_diff", bottom1.diff);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}


void TestProductLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Product Layer." << endl;
  Node<cpu> bottom0, bottom1;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(2,2,3,5), true);
  bottom1.Resize(Shape4(2,2,3,1), true);
  prnd->SampleUniform(&bottom0.data, -0.1, 0.1);
  prnd->SampleUniform(&bottom1.data, -0.1, 0.1);
  
  map<string, SettingV> setting;

  Layer<cpu> *layer = CreateLayer<cpu>(kProduct);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // prnd->SampleUniform(&bottom.data, -0.1, 0.1);
  // prnd->SampleUniform(&top.diff, -0.1, 0.1);
  layer->Forward(bottoms, tops);
  top.diff = 1.f; // bottom.data;
  PrintTensor("bottom0", bottom0.data);
  PrintTensor("bottom1", bottom1.data);
  PrintTensor("top", top.data);
  PrintTensor("top_diff", top.diff);
  // PrintTensor("param_diff", layer->GetParams()[0].diff);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.000001f);
  setting_checker["range_max"] = SettingV(0.000001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);
  PrintTensor("bottom0_diff", bottom0.diff);
  PrintTensor("bottom1_diff", bottom1.diff);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}




void TestGateLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Gate Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,20,30,50), true);
  prnd->SampleUniform(&bottom.data, -0.1, 0.1);
  bottom.length = 1;
  // bottom.data[0][0][0][0] = 0.2;
  // bottom.data[0][0][0][1] = 0.3;
  // bottom.data[0][0][1][0] = 0.2;
  // bottom.data[0][0][1][1] = 0.3;
  // bottom.data[0][0][2][0] = 0.2;
  // bottom.data[0][0][2][1] = 0.2;
  // bottom.data[0][1][0][0] = 0.2;
  // bottom.data[0][1][0][1] = 0.2;
  // bottom.data[0][1][1][0] = 0.2;
  // bottom.data[0][1][1][1] = 0.3;
  // bottom.data[0][1][2][0] = 1.5;
  // bottom.data[0][1][2][1] = -0.5;
  // bottom.data[1][0][0][0] = 0.3;
  // bottom.data[1][0][0][1] = 1.5;
  // bottom.data[1][0][1][0] = -0.5;
  // bottom.data[1][0][1][1] = 0.3;
  // bottom.data[1][0][2][0] = 1.5;
  // bottom.data[1][0][2][1] = -0.5;
  // bottom.data[1][1][0][0] = 0.3;
  // bottom.data[1][1][0][1] = 1.5;
  // bottom.data[1][1][1][0] = -0.5;
  // bottom.data[1][1][1][1] = 0.3;
  // bottom.data[1][1][2][0] = 1.5;
  // bottom.data[1][1][2][1] = -0.5;
  
  map<string, SettingV> setting;
  {
    setting["no_bias"] = SettingV(true);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kConstant);
      w_filler["value"] = SettingV(0.1f);
    setting["w_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
  }

  Layer<cpu> * layer = CreateLayer<cpu>(kGate);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // prnd->SampleUniform(&bottom.data, -0.1, 0.1);
  // prnd->SampleUniform(&top.diff, -0.1, 0.1);
  layer->Forward(bottoms, tops);
  top.diff = 1.f; // bottom.data;
  PrintTensor("top", top.data);
  PrintTensor("top_diff", top.diff);
  PrintTensor("bottom_diff", bottom.diff);
  PrintTensor("param_diff", layer->GetParams()[0].diff);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  // layer->Forward(bottoms, tops);
  // layer->Backprop(bottoms, tops);
  // PrintTensor("b0_data", bottoms[0]->data);
  // PrintTensor("t_data",  tops[0]->data);
  // PrintTensor("b0_length", bottoms[0]->length);
  // PrintTensor("t_length", tops[0]->length);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestGatingLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Gating Layer." << endl;
  Node<cpu> bottom0;
  Node<cpu> bottom1;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(2,2,3,2), true);
  bottom1.Resize(Shape4(2,2,3,1), true);

  // prnd->SampleUniform(&bottom0.data, -0.1, 0.1);

  bottom1.data = 1;

  bottom0.data[0][0][0][0] = 0.2;
  bottom0.data[0][0][0][1] = 0.3;
  bottom0.data[0][0][1][0] = 0.2;
  bottom0.data[0][0][1][1] = 0.3;
  bottom0.data[0][0][2][0] = 0.2;
  bottom0.data[0][0][2][1] = 0.2;
  bottom0.data[0][1][0][0] = 0.2;
  bottom0.data[0][1][0][1] = 0.2;
  bottom0.data[0][1][1][0] = 0.2;
  bottom0.data[0][1][1][1] = 0.3;
  bottom0.data[0][1][2][0] = 1.5;
  bottom0.data[0][1][2][1] = -0.5;
  bottom0.data[1][0][0][0] = 0.3;
  bottom0.data[1][0][0][1] = 1.5;
  bottom0.data[1][0][1][0] = -0.5;
  bottom0.data[1][0][1][1] = 0.3;
  bottom0.data[1][0][2][0] = 1.5;
  bottom0.data[1][0][2][1] = -0.5;
  bottom0.data[1][1][0][0] = 0.3;
  bottom0.data[1][1][0][1] = 1.5;
  bottom0.data[1][1][1][0] = -0.5;
  bottom0.data[1][1][1][1] = 0.3;
  bottom0.data[1][1][2][0] = 1.5;
  bottom0.data[1][1][2][1] = -0.5;

  bottom1.data[0][0][0][0] = 2;
  bottom1.data[0][0][1][0] = 5;
  bottom1.data[0][0][2][0] = -1;
  bottom1.data[0][1][0][0] = 2;
  bottom1.data[0][1][1][0] = 2;
  bottom1.data[0][1][2][0] = 5;
  bottom1.data[1][0][0][0] = 3;
  bottom1.data[1][0][1][0] = -1;
  bottom1.data[1][0][2][0] = -1;
  bottom1.data[1][1][0][0] = 3;
  bottom1.data[1][1][1][0] = 5;
  bottom1.data[1][1][2][0] = 5;

  bottom0.length = 1; // useless
  
  map<string, SettingV> setting;
  {
	setting["gate_type"] = SettingV("word-share");
	setting["activefun_type"] = SettingV("sigmoid");
	setting["word_count"] = SettingV(10);
	setting["feat_size"] = SettingV(2);

    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      //w_filler["init_type"] = SettingV(initializer::kUniform);
      //w_filler["range"] = SettingV(0.01f);
      w_filler["init_type"] = SettingV(initializer::kConstant);
      w_filler["value"] = SettingV(0.1f);
    setting["w_filler"] = SettingV(&w_filler);

    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
  }

  Layer<cpu> * layer = CreateLayer<cpu>(kGating);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  
  layer->Forward(bottoms, tops);

  PrintTensor("bottom0", bottom0.data);
  PrintTensor("bottom1", bottom1.data);
  PrintTensor("param", layer->GetParams()[0].data);
  PrintTensor("top", top.data);
  
  top.diff = 1.0f;
  // layer->Backprop(bottoms, tops);
  PrintTensor("top", top.data);
  PrintTensor("top_diff", top.diff);
  PrintTensor("bottom_diff", bottom0.diff);
  PrintTensor("bottom_diff", bottom1.diff);
  PrintTensor("param_diff", layer->GetParams()[0].diff);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer, bottoms, tops);

  cout << "Done." << endl;
}


void TestConvolutionLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Convolution Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,8,5), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  bottom.length = 4;
  
  map<string, SettingV> setting;
  {
    setting["pad_x"] = SettingV(0);
    setting["pad_y"] = SettingV(2);
    setting["kernel_x"] = SettingV(5);
    setting["kernel_y"] = SettingV(3);
    setting["channel_out"] = SettingV(4);
    setting["no_bias"] = SettingV(true);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
  }

  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kConv);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  prnd->SampleUniform(&top.diff, -0.1, 0.1);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer, bottoms, tops);
  layer->Forward(bottoms, tops);
  layer->Backprop(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("t_data",  tops[0]->data);
  PrintTensor("b0_length", bottoms[0]->length);
  PrintTensor("t_length", tops[0]->length);
  PrintTensor("b0_diff", bottoms[0]->diff);
  PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestConvResultTransformLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Conv result transform Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,3,10,1), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  bottom.length = 4;
  
  map<string, SettingV> setting;

  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kConvResultTransform);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  prnd->SampleUniform(&top.diff, -0.1, 0.1);
  
  using namespace checker;
  // Checker<cpu> * cker = CreateChecker<cpu>();
  // map<string, SettingV> setting_checker;
  // setting_checker["range_min"] = SettingV(-0.001f);
  // setting_checker["range_max"] = SettingV(0.001f);
  // setting_checker["delta"] = SettingV(0.0001f);
  // cker->SetupChecker(setting_checker, prnd);
  // cout << "Check Error." << endl;
  // cker->CheckError(layer, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);
  layer->Forward(bottoms, tops);
  layer->Backprop(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("t_data",  tops[0]->data);
  PrintTensor("b0_length", bottoms[0]->length);
  PrintTensor("t_length", tops[0]->length);
  PrintTensor("b0_diff", bottoms[0]->diff);
  PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;

}


void TestRnnLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Lstm Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,10,5), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  bottom.length = 10;
  
  map<string, SettingV> setting;
  {
    setting["d_mem"] = SettingV(3);
    setting["no_bias"] = SettingV(true);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_filler);
    setting["u_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
    setting["u_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
  }

  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kRecurrent);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer, bottoms, tops);
}

void TestLstmAutoencoderLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Lstm Autoencoder Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,10,5), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  bottom.length[0][0] = 6;
  bottom.length[1][0] = 8;
  
  map<string, SettingV> setting;
  {
    setting["d_mem"] = SettingV(3);
    setting["no_bias"] = SettingV(false);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.1f);
    setting["w_ec_filler"] = SettingV(&w_filler);
    setting["u_ec_filler"] = SettingV(&w_filler);
    setting["b_ec_filler"] = SettingV(&w_filler);
    setting["w_dc_filler"] = SettingV(&w_filler);
    setting["u_dc_filler"] = SettingV(&w_filler);
    setting["b_dc_filler"] = SettingV(&w_filler);

    // map<string, SettingV> &b_filler = *(new map<string, SettingV>());
    //   b_filler["init_type"] = SettingV(initializer::kZero);
    // setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(2);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_ec_updater"] = SettingV(&w_updater);
    setting["u_ec_updater"] = SettingV(&w_updater);
    setting["b_ec_updater"] = SettingV(&w_updater);
    setting["w_dc_updater"] = SettingV(&w_updater);
    setting["u_dc_updater"] = SettingV(&w_updater);
    setting["b_dc_updater"] = SettingV(&w_updater);
  }

  /// Test Activation Layer
  Layer<cpu> * layer_fc = CreateLayer<cpu>(kLstmAutoencoder);
  layer_fc->PropAll();
  layer_fc->SetupLayer(setting, bottoms, tops, prnd);
  layer_fc->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer_fc, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_fc, bottoms, tops);
}

void TestGruLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Gru Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,10,5), true);
  bottom.length = 10;
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  {
    setting["d_mem"] = SettingV(3);
    setting["reverse"] = SettingV(false);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.5f);
    setting["w_g_filler"] = SettingV(&w_filler);
    setting["u_g_filler"] = SettingV(&w_filler);
    setting["w_c_filler"] = SettingV(&w_filler);
    setting["u_c_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_g_filler"] = SettingV(&b_filler);
    setting["b_c_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["batch_size"] = SettingV(1);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_g_updater"] = SettingV(&w_updater);
    setting["u_g_updater"] = SettingV(&w_updater);
    setting["b_g_updater"] = SettingV(&w_updater);
    setting["w_c_updater"] = SettingV(&w_updater);
    setting["u_c_updater"] = SettingV(&w_updater);
    setting["b_c_updater"] = SettingV(&w_updater);
  }

  
  /// Test Activation Layer
  Layer<cpu> * layer_fc = CreateLayer<cpu>(kGru);
  layer_fc->PropAll();
  layer_fc->SetupLayer(setting, bottoms, tops, prnd);
  layer_fc->Reshape(bottoms, tops);
  PrintTensor("b0", bottoms[0]->data);
  layer_fc->Forward(bottoms, tops);
  PrintTensor("t0", tops[0]->data);
  // exit(0);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.00001f);
  setting_checker["range_max"] = SettingV(0.00001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer_fc, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_fc, bottoms, tops);
}

void TestLstmD2Layer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Lstm D2 Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  int batch_size = 2;
  int max_len = 6;
  int d_mem = 3;
  bottom.Resize(Shape4(batch_size, max_len, max_len, d_mem), Shape2(batch_size,2), true);
  bottom.length[0][0] = 2;
  bottom.length[0][1] = 4;
  bottom.length[1][0] = 3;
  bottom.length[1][1] = 1;
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  {
    setting["d_mem"] = SettingV(d_mem);
    setting["no_bias"] = SettingV(true);
    setting["reverse"] = SettingV(true);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.1f);
    setting["w_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["batch_size"] = SettingV(1);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
  }
  /// Test Activation Layer
  Layer<cpu> * layer_fc = CreateLayer<cpu>(kLstmD2Optimize);
  layer_fc->PropAll();
  layer_fc->SetupLayer(setting, bottoms, tops, prnd);
  layer_fc->Reshape(bottoms, tops);
  layer_fc->Forward(bottoms, tops);
  prnd->SampleUniform(&top.diff, -1.0, 1.0);
  bottom.diff = 0.f;
  // PrintTensor("t0_diff", tops[0]->diff);
  layer_fc->Backprop(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("t0_data", tops[0]->data);
  PrintTensor("b0_diff", bottoms[0]->diff);
  PrintTensor("t0_diff", tops[0]->diff);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  layer_fc->Forward(bottoms, tops);
  prnd->SampleUniform(&top.diff, -1.0, 1.0);
  layer_fc->Backprop(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("t0_data", tops[0]->data);
  PrintTensor("b0_diff", bottoms[0]->diff);
  PrintTensor("t0_diff", tops[0]->diff);
  // PrintTensor("t0_diff", tops[0]->diff);
  // PrintTensor("b0", bottoms[0]->data);
  // PrintTensor("t0", tops[0]->data);
  // PrintTensor("t0_diff", tops[0]->diff);
  // PrintTensor("b0_diff", bottoms[0]->diff);
  // PrintTensor("w_diff", layer_fc->params[0].diff);
  // PrintTensor("b_diff", layer_fc->params[1].diff);
  
  // using namespace checker;
  // Checker<cpu> * cker = CreateChecker<cpu>();
  // map<string, SettingV> setting_checker;
  // setting_checker["range_min"] = SettingV(-0.001f);
  // setting_checker["range_max"] = SettingV(0.001f);
  // setting_checker["delta"] = SettingV(0.0001f);
  // cker->SetupChecker(setting_checker, prnd);
  // cout << "Check Error." << endl;
  // cker->CheckError(layer_fc, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer_fc, bottoms, tops);
}

void TestLstmLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Lstm Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,10,5), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  {
    setting["d_input"] = SettingV(5);
    setting["d_mem"] = SettingV(3);
    setting["no_bias"] = SettingV(true);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.1f);
    setting["w_filler"] = SettingV(&w_filler);
    setting["u_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
    setting["w_updater"] = SettingV(&w_updater);
    setting["u_updater"] = SettingV(&w_updater);
    setting["b_updater"] = SettingV(&w_updater);
  }

  
  /// Test Activation Layer
  Layer<cpu> * layer_fc = CreateLayer<cpu>(kLstm);
  layer_fc->PropAll();
  layer_fc->SetupLayer(setting, bottoms, tops, prnd);
  layer_fc->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer_fc, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_fc, bottoms, tops);
}

void TestConvolutionalLstmLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Conv Lstm Layer." << endl;
  Node<cpu> bottom_0, bottom_1, bottom_2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom_0);
  bottoms.push_back(&bottom_1);
  bottoms.push_back(&bottom_2);
  tops.push_back(&top);
  
  bottom_0.Resize(Shape4(2,1,10,5), true);
  bottom_1.Resize(Shape4(2,1,10,5), true);
  bottom_2.Resize(Shape4(2,1,10,5), true);
  prnd->SampleUniform(&bottom_0.data, -0.1, 0.1);
  prnd->SampleUniform(&bottom_1.data, -0.1, 0.1);
  prnd->SampleUniform(&bottom_2.data, -0.1, 0.1);

  bottom_0.data[0][0].Slice(0, 2) = NAN; // padding
  bottom_0.data[0][0].Slice(6, 10)= NAN; // padding
  bottom_0.data[1][0].Slice(0, 2) = NAN; // padding
  bottom_0.data[1][0].Slice(6, 10)= NAN; // padding
  bottom_1.data[0][0].Slice(0, 2) = NAN; // padding
  bottom_1.data[0][0].Slice(6, 10)= NAN; // padding
  bottom_1.data[1][0].Slice(0, 2) = NAN; // padding
  bottom_1.data[1][0].Slice(6, 10)= NAN; // padding
  bottom_2.data[0][0].Slice(0, 2) = NAN; // padding
  bottom_2.data[0][0].Slice(6, 10)= NAN; // padding
  bottom_2.data[1][0].Slice(0, 2) = NAN; // padding
  bottom_2.data[1][0].Slice(6, 10)= NAN; // padding
  
  map<string, SettingV> setting;
  {
    setting["num_hidden"] = SettingV(3);
    setting["pad_value"] = SettingV(NAN);
    setting["no_bias"] = SettingV(true);
    setting["batch_size"] = SettingV(2);
    setting["output_padding_zero"] = SettingV(false);
      
    map<string, SettingV> &w_filler = *(new map<string, SettingV>());
      w_filler["init_type"] = SettingV(initializer::kUniform);
      w_filler["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["max_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
      w_updater["batch_size"] = SettingV(2);
    setting["w_updater"] = SettingV(&w_updater);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      b_updater["updater_type"] = SettingV(updater::kAdagrad);
      b_updater["eps"] = SettingV(0.01f);
      b_updater["max_iter"] = SettingV(10000);
      b_updater["lr"] = SettingV(0.1f);
      b_updater["batch_size"] = SettingV(2);
    setting["b_updater"] = SettingV(&b_updater);
  }

  
  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kConvolutionalLstm);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();

  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer, bottoms, tops);


  layer->Forward(bottoms, tops);
  PrintTensor("b0", bottoms[0]->data);
  PrintTensor("b1", bottoms[1]->data);
  PrintTensor("b2", bottoms[2]->data);
  PrintTensor("t0", tops[0]->data);
  PrintTensor("cc_data",((ConvolutionalLstmLayer<cpu> *)(layer))->concat_input_data);
  PrintTensor("cc_diff",((ConvolutionalLstmLayer<cpu> *)(layer))->concat_input_diff);
  cout << "Done." << endl;
}

void TestWholePoolingLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check WholePooling Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,6,3), true);
  prnd->SampleUniform(&bottom.data, -0.1, 0.1);

  bottom.data[0][0].Slice(0, 2) = NAN; // padding
  bottom.data[0][0].Slice(4, 6) = NAN; // padding
  bottom.data[1][0].Slice(0, 1) = NAN; // padding
  bottom.data[1][0].Slice(4, 6) = NAN; // padding
  
  map<string, SettingV> setting;
  {
    setting["pool_type"] = SettingV("last");
    setting["pad_value"] = SettingV((float)(NAN));
  }

  
  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kWholePooling);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  prnd->SampleUniform(&top.diff, -0.1, 0.1);
  
  using namespace checker;
  // Checker<cpu> * cker = CreateChecker<cpu>();

  // map<string, SettingV> setting_checker;
  // setting_checker["range_min"] = SettingV(-0.001f);
  // setting_checker["range_max"] = SettingV(0.001f);
  // setting_checker["delta"] = SettingV(0.0001f);
  // cker->SetupChecker(setting_checker, prnd);

  // cout << "Check Error." << endl;
  // cker->CheckError(layer, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);


  layer->Forward(bottoms, tops);
  layer->Backprop(bottoms, tops);
  PrintTensor("b_data", bottoms[0]->data);
  PrintTensor("t_data", tops[0]->data);
  PrintTensor("b_diff", bottoms[0]->diff);
  PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestConcatLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Concat Layer." << endl;
  Node<cpu> bottom_0, bottom_1, bottom_2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom_0);
  bottoms.push_back(&bottom_1);
  bottoms.push_back(&bottom_2);
  tops.push_back(&top);
  
  bottom_0.Resize(Shape4(2,1,2,3), true);
  bottom_1.Resize(Shape4(2,2,2,3), true);
  bottom_2.Resize(Shape4(2,1,2,3), true);
  prnd->SampleUniform(&bottom_0.data, -0.1, 0.1);
  prnd->SampleUniform(&bottom_1.data, -0.1, 0.1);
  prnd->SampleUniform(&bottom_2.data, -0.1, 0.1);

  map<string, SettingV> setting;
  {
    setting["bottom_node_num"] = SettingV(3);
    setting["concat_dim_index"] = SettingV(1);
  }

  
  /// Test Activation Layer
  Layer<cpu> * layer = CreateLayer<cpu>(kConcat);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  prnd->SampleUniform(&top.diff, -0.1, 0.1);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();

  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.001f);
  setting_checker["range_max"] = SettingV(0.001f);
  setting_checker["delta"] = SettingV(0.0001f);

  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer, bottoms, tops);

  // cout << "Check Grad." << endl;
  // cker->CheckGrad(layer, bottoms, tops);

  layer->Forward(bottoms, tops);
  layer->Backprop(bottoms, tops);
  PrintTensor("b0_data", bottoms[0]->data);
  PrintTensor("b1_data", bottoms[1]->data);
  PrintTensor("b2_data", bottoms[2]->data);
  PrintTensor("t_data",  tops[0]->data);
  PrintTensor("b0_diff", bottoms[0]->diff);
  PrintTensor("b1_diff", bottoms[1]->diff);
  PrintTensor("b2_diff", bottoms[2]->diff);
  PrintTensor("t_diff", tops[0]->diff);
  cout << "Done." << endl;
}

void TestFcLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Fc Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,5,5), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  
  map<string, SettingV> setting;
  setting["num_hidden"] = SettingV(10);
  setting["no_bias"] = SettingV(false);
    map<string, SettingV> w_setting;
    w_setting["init_type"] = SettingV(initializer::kGaussian);
    w_setting["mu"] = SettingV(0.0f);
    w_setting["sigma"] = SettingV(1.0f);
    map<string, SettingV> b_setting;
    b_setting["init_type"] = SettingV(initializer::kZero);
  setting["w_filler"] = SettingV(&w_setting);
  setting["b_filler"] = SettingV(&b_setting);
  map<string, SettingV> w_updater;
    w_updater["init_type"] = SettingV(updater::kSGD);
    w_updater["momentum"] = SettingV(0.0f);
    w_updater["lr"] = SettingV(0.001f);
    w_updater["decay"] = SettingV(0.001f);
    map<string, SettingV> b_updater;
    b_updater["init_type"] = SettingV(updater::kSGD);
    b_updater["momentum"] = SettingV(0.0f);
    b_updater["lr"] = SettingV(0.001f);
    b_updater["decay"] = SettingV(0.001f);
  setting["w_updater"] = SettingV(&w_updater);
  setting["b_updater"] = SettingV(&b_updater);
  
  /// Test Activation Layer
  Layer<cpu> * layer_fc = CreateLayer<cpu>(kFullConnect);
  layer_fc->PropAll();
  layer_fc->SetupLayer(setting, bottoms, tops, prnd);
  layer_fc->Reshape(bottoms, tops);
  
  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.01f);
  setting_checker["range_max"] = SettingV(0.01f);
  setting_checker["delta"] = SettingV(0.0001f);
  cker->SetupChecker(setting_checker, prnd);
  cout << "Check Error." << endl;
  cker->CheckError(layer_fc, bottoms, tops);

  cout << "Check Grad." << endl;
  cker->CheckGrad(layer_fc, bottoms, tops);
}

void TestTextDataLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check TextData Layer." << endl;
  Node<cpu> top1;
  Node<cpu> top2;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  tops.push_back(&top1);
  tops.push_back(&top2);

  map<string, SettingV> setting;
  setting["data_file"] = SettingV("/home/pangliang/matching/data/msr_paraphrase_train_wid.txt");
  setting["batch_size"] = SettingV(2);
  setting["max_doc_len"] = SettingV(31);
  setting["min_doc_len"] = SettingV(5);
  
  /// Test TextData Layer
  Layer<cpu> * layer_textdata = CreateLayer<cpu>(kTextData);
  layer_textdata->PropAll();
  layer_textdata->SetupLayer(setting, bottoms, tops, prnd);
  layer_textdata->Reshape(bottoms, tops);
  layer_textdata->Forward(bottoms, tops);
  layer_textdata->Backprop(bottoms, tops);

  PrintTensor("top1", top1.data);
  PrintTensor("top2", top2.data);

  vector<Node<cpu>*> bottoms_wv;
  vector<Node<cpu>*> tops_wv;
  Node<cpu> top_wv;

  bottoms_wv.push_back(&top1);
  tops_wv.push_back(&top_wv);

  map<string, SettingV> setting_wv;
  map<string, SettingV> setting_wfiller;
  setting_wv["embedding_file"] = SettingV("/home/pangliang/matching/data/wikicorp_50_msr.txt");
  setting_wv["word_count"] = SettingV(14727);
  setting_wv["feat_size"] = SettingV(50);
  setting_wfiller["init_type"] = SettingV(initializer::kGaussian);
  setting_wfiller["mu"] = SettingV(0.0f);
  setting_wfiller["sigma"] = SettingV(1.0f);
  setting_wv["w_filler"] = SettingV(&setting_wfiller);
  map<string, SettingV> w_updater;
    w_updater["init_type"] = SettingV(updater::kSGD);
    w_updater["momentum"] = SettingV(0.0f);
    w_updater["lr"] = SettingV(0.001f);
    w_updater["decay"] = SettingV(0.001f);
  setting_wv["w_updater"] = SettingV(&w_updater);

  // Test Embedding Layer
  Layer<cpu> * layer_embedding = CreateLayer<cpu>(kEmbedding);
  layer_embedding->PropAll();
  layer_embedding->SetupLayer(setting_wv, bottoms_wv, tops_wv, prnd);
  layer_embedding->Reshape(bottoms_wv, tops_wv);
  layer_embedding->Forward(bottoms_wv, tops_wv);
  top_wv.diff = 1.0;
  layer_embedding->Backprop(bottoms_wv, tops_wv);

  PrintTensor("top_wv", top_wv.data);
  PrintTensor("top1_diff", top1.diff);
  PrintTensor("weight_diff", layer_embedding->GetParams()[0].diff);
  PrintTensor("weight_idx", layer_embedding->GetParams()[0].idx);

  vector<Node<cpu>*> bottoms_sp;
  vector<Node<cpu>*> tops_sp;
  Node<cpu> top1_sp;
  Node<cpu> top2_sp;

  bottoms_sp.push_back(&top_wv);
  tops_sp.push_back(&top1_sp);
  tops_sp.push_back(&top2_sp);

  // Test Split Layer
  Layer<cpu> * layer_split = CreateLayer<cpu>(kSplit);
  layer_split->PropAll();
  layer_split->SetupLayer(setting, bottoms_sp, tops_sp, prnd);
  layer_split->Reshape(bottoms_sp, tops_sp);
  layer_split->Forward(bottoms_sp, tops_sp);
  top1_sp.diff = 1.0;
  top2_sp.diff = 2.0;
  layer_split->Backprop(bottoms_sp, tops_sp);

  PrintTensor("top1_sp", top1_sp.data);
  PrintTensor("top2_sp", top2_sp.data);
  PrintTensor("top_wv_diff", top_wv.diff);
}

void TestPairTextDataLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check PairTextData Layer." << endl;
  Node<cpu> top1;
  Node<cpu> top2;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  tops.push_back(&top1);
  tops.push_back(&top2);

  map<string, SettingV> setting;
  // setting["data_file"] = SettingV("/home/pangliang/matching/data/webap/qaSentWordIndex.dat.sample");
  setting["data_file"] = SettingV("/home/pangliang/matching/data/webscope/xrear10.3.32/qa.xrear10.3.32.train.dat");
  setting["batch_size"] = SettingV(2);
  setting["max_doc_len"] = SettingV(32);
  setting["min_doc_len"] = SettingV(5);
  setting["shuffle"] = SettingV(false);
  
  /// Test PairTextData Layer
  Layer<cpu> * layer_pair_textdata = CreateLayer<cpu>(kPairTextData);
  layer_pair_textdata->PropAll();
  layer_pair_textdata->SetupLayer(setting, bottoms, tops, prnd);
  layer_pair_textdata->Reshape(bottoms, tops);
  layer_pair_textdata->Forward(bottoms, tops);
  layer_pair_textdata->Backprop(bottoms, tops);

  PrintTensor("top1", top1.data);
  PrintTensor("top2", top2.data);
}

void TestListTextDataLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check ListTextData Layer." << endl;
  Node<cpu> top1;
  Node<cpu> top2;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  tops.push_back(&top1);
  tops.push_back(&top2);

  map<string, SettingV> setting;
  setting["data_file"] = SettingV("/home/pangliang/matching/data/webscope/qa_instances.test.dat");
  setting["max_doc_len"] = SettingV(32);
  setting["min_doc_len"] = SettingV(5);
  
  /// Test PairTextData Layer
  Layer<cpu> * layer_list_textdata = CreateLayer<cpu>(kListTextData);
  layer_list_textdata->PropAll();
  layer_list_textdata->SetupLayer(setting, bottoms, tops, prnd);
  layer_list_textdata->Reshape(bottoms, tops);
  layer_list_textdata->Forward(bottoms, tops);
  layer_list_textdata->Backprop(bottoms, tops);
  layer_list_textdata->Forward(bottoms, tops);
  layer_list_textdata->Backprop(bottoms, tops);

  PrintTensor("top1", top1.data);
  PrintTensor("top2", top2.data);
}

void TestMapTextDataLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check MapTextData Layer." << endl;
  Node<cpu> top1;
  Node<cpu> top2;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  tops.push_back(&top1);
  tops.push_back(&top2);

  map<string, SettingV> setting;
  setting["data1_file"] = SettingV("data1.wid");
  setting["data2_file"] = SettingV("data2.wid");
  setting["rel_file"] = SettingV("rel.dat");
  setting["batch_size"] = SettingV(5);
  setting["max_doc_len"] = SettingV(3);
  setting["min_doc_len"] = SettingV(3);
  setting["mode"] = SettingV("batch");
  setting["shuffle"] = SettingV("false");
  
  /// Test MapTextData Layer
  Layer<cpu> * layer_map_textdata = CreateLayer<cpu>(kMapTextData);
  layer_map_textdata->PropAll();
  layer_map_textdata->SetupLayer(setting, bottoms, tops, prnd);
  layer_map_textdata->Reshape(bottoms, tops);
  layer_map_textdata->Forward(bottoms, tops);
  layer_map_textdata->Backprop(bottoms, tops);

  PrintTensor("top1", top1.data);
  PrintTensor("top2", top2.data);
  PrintTensor("top1", top1.length);
  PrintTensor("top2", top2.length);
}

void TestQATextDataLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check QATextData Layer." << endl;
  Node<cpu> top1;
  Node<cpu> top2;
  Node<cpu> top3;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  tops.push_back(&top1);
  tops.push_back(&top2);
  tops.push_back(&top3);

  map<string, SettingV> setting;
  setting["question_data_file"] = SettingV("/home/pangliang/matching/data/webscope/x.32/question.full.dat.wid");
  setting["answer_data_file"] = SettingV("/home/pangliang/matching/data/webscope/x.32/answer.dat.wid");
  setting["question_rel_file"] = SettingV("/home/pangliang/matching/data/webscope/x.32/q.x.32.train.dat");
  setting["answer_rel_file"] = SettingV("/home/pangliang/matching/data/webscope/x.32/a.x.32.train.dat");
  setting["batch_size"] = SettingV(2);
  setting["max_doc_len"] = SettingV(32);
  setting["candids"] = SettingV(10);
  setting["mode"] = SettingV("list");
  
  /// Test QATextData Layer
  Layer<cpu> * layer_qa_textdata = CreateLayer<cpu>(kQATextData);
  layer_qa_textdata->PropAll();
  layer_qa_textdata->SetupLayer(setting, bottoms, tops, prnd);
  layer_qa_textdata->Reshape(bottoms, tops);
  layer_qa_textdata->Forward(bottoms, tops);
  layer_qa_textdata->Backprop(bottoms, tops);

  PrintTensor("top1", top1.data);
  PrintTensor("top2", top2.data);
  PrintTensor("top3", top3.data);
}

void TestActivationLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Act Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.data.Resize(Shape4(2,1,5,5), 1.0);
  bottom.diff.Resize(Shape4(2,1,5,5), 0.0);
  
  map<string, SettingV> setting;
  //setting["layer_type"] = SettingV(kRectifiedLinear);

  /// Test Activation Layer
  Layer<cpu> * layer_rectify = CreateLayer<cpu>(kRectifiedLinear);
  layer_rectify->PropAll();
  layer_rectify->SetupLayer(setting, bottoms, tops, prnd);
  layer_rectify->Reshape(bottoms, tops);
  layer_rectify->Forward(bottoms, tops);
  top.diff.Resize(Shape4(2,1,5,5), 1.0);
  layer_rectify->Backprop(bottoms, tops);

  cout << "bottom size: " << bottom.data.shape_[0] << "x" << bottom.data.shape_[1] << "x" << bottom.data.shape_[2] << "x" << bottom.data.shape_[3] << endl;
  cout << "top size: " << top.data.shape_[0] << "x" << top.data.shape_[1] << "x" << top.data.shape_[2] << "x" << top.data.shape_[3] << endl;
  
  cout << "top data: " << top.data[0][0][0][0] << endl;
  cout << "bottom diff: " << bottom.diff[0][0][0][0] << endl;
  
  Layer<cpu> * layer_sigmoid = CreateLayer<cpu>(kSigmoid);
  layer_sigmoid->PropAll();
  layer_sigmoid->SetupLayer(setting, bottoms, tops, prnd);
  layer_sigmoid->Reshape(bottoms, tops);
  layer_sigmoid->Forward(bottoms, tops);
  top.diff.Resize(Shape4(2,1,5,5), 1.0);
  layer_sigmoid->Backprop(bottoms, tops);

  cout << "bottom size: " << bottom.data.shape_[0] << "x" << bottom.data.shape_[1] << "x" << bottom.data.shape_[2] << "x" << bottom.data.shape_[3] << endl;
  cout << "top size: " << top.data.shape_[0] << "x" << top.data.shape_[1] << "x" << top.data.shape_[2] << "x" << top.data.shape_[3] << endl;
  
  cout << "top data: " << top.data[0][0][0][0] << endl;
  cout << "bottom diff: " << bottom.diff[0][0][0][0] << endl;
}

void TestMatchMultiLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check MatchMulti Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  Node<cpu> len1;
  Node<cpu> len2;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom);
  tops.push_back(&top);
  tops.push_back(&len1);
  tops.push_back(&len2);

  bottom.Resize(9, 1, 5, 2);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  bottom.length = 5;

  map<string, SettingV> setting;
  setting["op"] = SettingV("euc_exp");
  setting["candids"] = SettingV(2);
  setting["output_len"] = SettingV(true);

  // Test MatchMulti Layer
  Layer<cpu> * layer_match_multi = CreateLayer<cpu>(kMatchMulti);
  layer_match_multi->PropAll();
  layer_match_multi->SetupLayer(setting, bottoms, tops, prnd);
  layer_match_multi->Reshape(bottoms, tops);

  layer_match_multi->Forward(bottoms, tops);
  PrintTensor("bottom", bottom.data);
  PrintTensor("top", top.data);
  PrintTensor("len1", len1.data);
  PrintTensor("len2", len2.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_match_multi, bottoms, tops);
}

void TestBatchCombineLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Batch Combine Layer." << endl;
  Node<cpu> bottom1;
  Node<cpu> bottom2;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom1);
  bottoms.push_back(&bottom2);
  tops.push_back(&top);

  bottom1.Resize(4, 2, 3, 2);
  bottom2.Resize(4, 2, 3, 2);

  prnd->SampleUniform(&bottom1.data, -1.0, 1.0);
  prnd->SampleUniform(&bottom2.data, -1.0, 1.0);

  map<string, SettingV> setting;
  setting["op"] = SettingV("euc_exp");
  setting["element"] = SettingV(false);
  setting["candids"] = SettingV(2);

  // Test BatchCombine Layer
  Layer<cpu> * layer_batch_combine = CreateLayer<cpu>(kBatchCombine);
  layer_batch_combine->PropAll();
  layer_batch_combine->SetupLayer(setting, bottoms, tops, prnd);
  layer_batch_combine->Reshape(bottoms, tops);

  layer_batch_combine->Forward(bottoms, tops);
  PrintTensor("bottom1", bottom1.data);
  PrintTensor("bottom2", bottom2.data);
  PrintTensor("top", top.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_batch_combine, bottoms, tops);
}

void TestBatchSelectLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Batch Select Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom);
  tops.push_back(&top);

  bottom.Resize(4, 2, 3, 2);

  prnd->SampleUniform(&bottom.data, -1.0, 1.0);

  map<string, SettingV> setting;
  setting["step"] = SettingV(2);

  // Test BatchCombine Layer
  Layer<cpu> * layer_batch_select = CreateLayer<cpu>(kBatchSelect);
  layer_batch_select->PropAll();
  layer_batch_select->SetupLayer(setting, bottoms, tops, prnd);
  layer_batch_select->Reshape(bottoms, tops);

  layer_batch_select->Forward(bottoms, tops);
  PrintTensor("bottom", bottom.data);
  PrintTensor("top", top.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_batch_select, bottoms, tops);
}

void TestBatchSplitLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Batch Split Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top0;
  Node<cpu> top1;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom);
  tops.push_back(&top0);
  tops.push_back(&top1);

  bottom.Resize(6, 2, 3, 2);

  prnd->SampleUniform(&bottom.data, -1.0, 1.0);

  map<string, SettingV> setting;
  setting["batch_step"] = SettingV(3);
  setting["batch_count"] = SettingV(1);

  // Test BatchCombine Layer
  Layer<cpu> * layer_batch_split = CreateLayer<cpu>(kBatchSplit);
  layer_batch_split->PropAll();
  layer_batch_split->SetupLayer(setting, bottoms, tops, prnd);
  layer_batch_split->Reshape(bottoms, tops);

  layer_batch_split->Forward(bottoms, tops);
  PrintTensor("bottom", bottom.data);
  PrintTensor("top0", top0.data);
  PrintTensor("top1", top1.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_batch_split, bottoms, tops);
}

void TestBatchConcatLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Batch Concat Layer." << endl;
  Node<cpu> bottom0;
  Node<cpu> bottom1;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  tops.push_back(&top);

  bottom0.Resize(2, 2, 3, 2);
  bottom1.Resize(4, 2, 3, 2);

  prnd->SampleUniform(&bottom0.data, -1.0, 1.0);
  prnd->SampleUniform(&bottom1.data, -1.0, 1.0);

  map<string, SettingV> setting;
  setting["batch_step"] = SettingV(3);
  setting["batch_count"] = SettingV(1);

  // Test BatchCombine Layer
  Layer<cpu> * layer_batch_concat = CreateLayer<cpu>(kBatchConcat);
  layer_batch_concat->PropAll();
  layer_batch_concat->SetupLayer(setting, bottoms, tops, prnd);
  layer_batch_concat->Reshape(bottoms, tops);

  layer_batch_concat->Forward(bottoms, tops);
  PrintTensor("bottom0", bottom0.data);
  PrintTensor("bottom1", bottom1.data);
  PrintTensor("top", top.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_batch_concat, bottoms, tops);
}

void TestBatchDuplicateLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check Batch Duplicate Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;

  bottoms.push_back(&bottom);
  tops.push_back(&top);

  bottom.Resize(2, 2, 3, 2);

  prnd->SampleUniform(&bottom.data, -1.0, 1.0);

  map<string, SettingV> setting;
  setting["dup_count"] = SettingV(3);

  // Test BatchCombine Layer
  Layer<cpu> * layer_batch_duplicate = CreateLayer<cpu>(kBatchDuplicate);
  layer_batch_duplicate->PropAll();
  layer_batch_duplicate->SetupLayer(setting, bottoms, tops, prnd);
  layer_batch_duplicate->Reshape(bottoms, tops);

  layer_batch_duplicate->Forward(bottoms, tops);
  PrintTensor("bottom", bottom.data);
  PrintTensor("top", top.data);

  using namespace checker;
  Checker<cpu> * cker = CreateChecker<cpu>();
  map<string, SettingV> setting_checker;
  setting_checker["range_min"] = SettingV(-0.0001f);
  setting_checker["range_max"] = SettingV(0.0001f);
  setting_checker["delta"] = SettingV(0.001f);
  cker->SetupChecker(setting_checker, prnd);

  cout << "Check Error." << endl;
  cker->CheckError(layer_batch_duplicate, bottoms, tops);
}

float SIGMOID_MAX_INPUT = 20;
int SIGMOID_TABLE_SIZE = 10000;
float *p_sigmoid_lookup_table;

float TANH_MAX_INPUT = 20;
int TANH_TABLE_SIZE = 10000;
float *p_tanh_lookup_table;


int main(int argc, char *argv[]) {
  float max = SIGMOID_MAX_INPUT;
  int len = SIGMOID_TABLE_SIZE;
  p_sigmoid_lookup_table = new float[len];
  for (int i = 0; i < len; ++i) {
    float exp_val = exp((float(i)*2*max)/len - max); // map position to value, frow small to large
    p_sigmoid_lookup_table[i] = exp_val/(exp_val+1);
  }
  max = TANH_MAX_INPUT;
  len = TANH_TABLE_SIZE;
  p_tanh_lookup_table = new float[len];
  for (int i = 0; i < len; ++i) {
    float exp_val = exp(2*((float(i)*2*max)/len - max)); // map position to value, frow small to large
    p_tanh_lookup_table[i] = (exp_val-1)/(exp_val+1);
  }
  mshadow::Random<cpu> rnd(37);
  // TestActivationLayer(&rnd);
  // TestFcLayer(&rnd);
  // TestConvLayer(&rnd);
  // TestConvVarLayer(&rnd);
  // TestLocalLayer(&rnd);
  // TestPoolLayer(&rnd);
  // TestCrossLayer(&rnd);
  // TestDropoutLayer(&rnd);
  // TestLstmLayer(&rnd);
  // TestLstmAutoencoderLayer(&rnd);
  // TestRnnLayer(&rnd);
  // TestMaxRnnLayer(&rnd);
  // TestTensorLayer(&rnd);
  // TestConvolutionalLstmLayer(&rnd);
  // TestSequenceDimReductionLayer(&rnd);
  // TestWholePoolingLayer(&rnd);
  // TestConcatLayer(&rnd);
  // TestConvResultTransformLayer(&rnd);
  // TestConvolutionLayer(&rnd);
  // TestMatchLayer(&rnd);
  // TestMatchTensorLayer(&rnd);
  // TestMatchTopKPoolingLayer(&rnd);
  // TestLstmD2Layer(&rnd);
  // TestSelectSubRepByTokenLayer(&rnd);
  // TestMatchWeightedDotLayer(&rnd);
  // TestGruLayer(&rnd);
  // TestMatchMultiLayer(&rnd);
  // TestDynamicKMaxPoolingLayer(&rnd);
  // TestBatchCombineLayer(&rnd);
  // TestBatchSelectLayer(&rnd);
  // TestBatchSplitLayer(&rnd);
  // TestBatchConcatLayer(&rnd);
  // TestBatchDuplicateLayer(&rnd);
  // TestPairTextDataLayer(&rnd);
  // TestListTextDataLayer(&rnd);
  // TestGateLayer(&rnd);
  // TestDiagRecurrentLayer(&rnd);
  // TestNegativeSampleLossLayer(&rnd);
  // TestPosPredRepLayer(&rnd);
  // TestSwapAxisLayer(mshadow::Random<cpu>* prnd);
  // TestFlattenLayer(mshadow::Random<cpu>* prnd);
  // TestSoftmaxFuncLayer(&rnd);
  // TestWordClassSoftmaxLayer(&rnd);
  // TestGatingLayer(&rnd);
  // TestSoftmaxVarLenFuncLayer(&rnd);
  // TestSumLayer(&rnd);
  // TestTopkPoolingLayer(&rnd);
  // TestHingeLossLayer(&rnd);
  // TestListwiseMeasureLayer(&rnd);
  // TestQATextDataLayer(&rnd);
  TestMapTextDataLayer(&rnd);
  return 0;
}
