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

void TestCrossLayer(mshadow::Random<cpu>* prnd) {
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
  bottom1.data = 1.0;
  bottom2.data = 2.0;

  map<string, SettingV> setting;
  // Test Cross Layer
  Layer<cpu> * layer_cross = CreateLayer<cpu>(kCross);
  layer_cross->PropAll();
  layer_cross->SetupLayer(setting, bottoms, tops, prnd);
  layer_cross->Reshape(bottoms, tops);
  layer_cross->Forward(bottoms, tops);
  top.diff = 1.0;
  top.diff[0][0][2][3] = 2.0;
  layer_cross->Backprop(bottoms, tops);
  
  PrintTensor("top", top.data);
  PrintTensor("bottom1 diff", bottom1.diff);
  PrintTensor("bottom2 diff", bottom2.diff);
}

void TestDropoutLayer(mshadow::Random<cpu>* prnd) {
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
  layer_dropout->Forward(bottoms, tops);
  top.diff.Resize(Shape4(2,1,5,5), 1.0);
  layer_dropout->Backprop(bottoms, tops);
  
  PrintTensor("top data", top.data);
  PrintTensor("bottom diff", bottom.diff);
}

void TestHingeLossLayer(mshadow::Random<cpu>* prnd) {
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

void TestAccuracyLayer(mshadow::Random<cpu>* prnd) {
  Node<cpu> bottom0;
  Node<cpu> bottom1;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom0);
  bottoms.push_back(&bottom1);
  tops.push_back(&top);
  
  bottom0.Resize(Shape4(4,2,1,1));
  bottom1.Resize(Shape4(4,1,1,1));

  bottom0.data[0][0][0][0] = 0.2;
  bottom0.data[0][1][0][0] = 0.8;
  bottom0.data[1][0][0][0] = 0.3;
  bottom0.data[1][1][0][0] = 0.7;
  bottom0.data[2][0][0][0] = 0.9;
  bottom0.data[2][1][0][0] = 0.1;
  bottom0.data[3][0][0][0] = 0.7;
  bottom0.data[3][1][0][0] = 0.3;

  bottom1.data[0][0][0][0] = 1;
  bottom1.data[1][0][0][0] = 0;
  bottom1.data[2][0][0][0] = 1;
  bottom1.data[3][0][0][0] = 0;
  
  map<string, SettingV> setting;
  setting["topk"] = SettingV(1);
  
  /// Test Activation Layer
  Layer<cpu> * layer_acc = CreateLayer<cpu>(kAccuracy);
  layer_acc->PropAll();
  layer_acc->SetupLayer(setting, bottoms, tops, prnd);
  layer_acc->Reshape(bottoms, tops);
  layer_acc->Forward(bottoms, tops);
  layer_acc->Backprop(bottoms, tops);
  
  PrintTensor("top data", top.data);
  PrintTensor("bottom diff", bottom0.diff);
}

void TestConvLayer(mshadow::Random<cpu>* prnd) {
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,5,5), true);
  bottom.data = 1.0;
  
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
  PrintTensor("param", layer_conv->GetParams()[0].data);
  layer_conv->Forward(bottoms, tops);
  top.diff = 1.0;
  layer_conv->Backprop(bottoms, tops);
  
  PrintTensor("top data", top.data);
  PrintTensor("bottom diff", bottom.diff);

}

void TestPoolLayer(mshadow::Random<cpu>* prnd) {
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

void TestFcLayer(mshadow::Random<cpu>* prnd) {
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.Resize(Shape4(2,1,5,5), true);
  bottom.data = 1.0;
  
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
  PrintTensor("param_before", layer_fc->GetParams()[0].data);
  layer_fc->Forward(bottoms, tops);
  top.diff = 1.0;
  layer_fc->Backprop(bottoms, tops);
  
  layer_fc->GetParams()[0].Update();
  layer_fc->GetParams()[1].Update();

  PrintTensor("param_after", layer_fc->GetParams()[0].data);

  PrintTensor("top data", top.data);
  PrintTensor("bottom diff", bottom.diff);

}

void TestTextDataLayer(mshadow::Random<cpu>* prnd) {
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
  //TestFcLayer(&rnd);
  //TestConvLayer(&rnd);
  //TestPoolLayer(&rnd);
  //TestCrossLayer(&rnd);
  //TestDropoutLayer(&rnd);
  //TestHingeLossLayer(&rnd);
  TestAccuracyLayer(&rnd);
  return 0;
}

