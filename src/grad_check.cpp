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
      w_updater["mat_iter"] = SettingV(10000);
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
      w_updater["mat_iter"] = SettingV(10000);
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
      w_updater["mat_iter"] = SettingV(10000);
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

void TestSoftmaxFuncLayer(mshadow::Random<cpu>* prnd) {
  cout << "G Check SoftmaxFunc Layer." << endl;
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(1,2,3,4), true);
  prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  PrintTensor("b0_data", bottom.data);
  bottom.length = 1;
  
  map<string, SettingV> setting;
  {
    setting["axis"] = SettingV(2);
  }

  Layer<cpu> * layer = CreateLayer<cpu>(kSoftmaxFunc);
  layer->PropAll();
  layer->SetupLayer(setting, bottoms, tops, prnd);
  layer->Reshape(bottoms, tops);
  // prnd->SampleUniform(&bottom.data, -1.0, 1.0);
  prnd->SampleUniform(&top.diff, -1.0, 1.0);
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
  PrintTensor("bottom0_diff", bottom0.diff);
  PrintTensor("bottom1_diff", bottom1.diff);
  // PrintTensor("param_diff", layer->GetParams()[0].diff);
  
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
      w_updater["mat_iter"] = SettingV(10000);
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
      w_updater["mat_iter"] = SettingV(10000);
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
      w_updater["mat_iter"] = SettingV(10000);
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
      w_filler["range"] = SettingV(0.01f);
    setting["w_filler"] = SettingV(&w_filler);
    setting["u_filler"] = SettingV(&w_filler);

    map<string, SettingV> &b_filler = *(new map<string, SettingV>());
      b_filler["init_type"] = SettingV(initializer::kZero);
    setting["b_filler"] = SettingV(&b_filler);
      
    map<string, SettingV> &w_updater = *(new map<string, SettingV>());
      w_updater["updater_type"] = SettingV(updater::kAdagrad);
      w_updater["eps"] = SettingV(0.01f);
      w_updater["mat_iter"] = SettingV(10000);
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
      w_updater["mat_iter"] = SettingV(10000);
      w_updater["lr"] = SettingV(0.1f);
      w_updater["batch_size"] = SettingV(2);
    setting["w_updater"] = SettingV(&w_updater);
    map<string, SettingV> &b_updater = *(new map<string, SettingV>());
      b_updater["updater_type"] = SettingV(updater::kAdagrad);
      b_updater["eps"] = SettingV(0.01f);
      b_updater["mat_iter"] = SettingV(10000);
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

int main(int argc, char *argv[]) {
  mshadow::Random<cpu> rnd(37);
  //TestActivationLayer(&rnd);
  // TestFcLayer(&rnd);
  // TestConvLayer(&rnd);
  //TestPoolLayer(&rnd);
  // TestCrossLayer(&rnd);
  //TestDropoutLayer(&rnd);
  // TestLstmLayer(&rnd);
  // TestRnnLayer(&rnd);
  // TestMaxRnnLayer(&rnd);
  // TestTensorLayer(&rnd);
  // TestConvolutionalLstmLayer(&rnd);
  // TestSequenceDimReductionLayer(&rnd);
  // TestWholePoolingLayer(&rnd);
  // TestConcatLayer(&rnd);
  // TestConvResultTransformLayer(&rnd);
  // TestConvolutionLayer(&rnd);
  // 
  // TestGateLayer(&rnd);
  TestProductLayer(&rnd);
  // TestSoftmaxFuncLayer(&rnd);
  // TestTopkPoolingLayer(&rnd);
  //TestHingeLossLayer(&rnd);
  return 0;
}

