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

void PrintTensor(Tensor<cpu, 3> x) {
	Shape<3> s = x.shape_;
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


void PrintTensor(Tensor<cpu, 4> x) {
	Shape<4> s = x.shape_;
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
}

void PrintTensorP(Tensor<cpu, 4> x) {
	Shape<4> s = x.shape_;
	float * p = x.dptr_;
	for (int i = 0; i < x.shape_.Size(); ++i) {
		cout << *p << " ";
		p++;
	}
}

void TestTextDataLayer() {
  Node<cpu> top1;
  Node<cpu> top2;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  tops.push_back(&top1);
  tops.push_back(&top2);

  cout << "Init test ok." << endl;

  map<string, SettingV> setting;
  setting["data_file"] = SettingV("/home/pangliang/matching/data/msr_paraphrase_train_wid.txt");
  setting["batch_size"] = SettingV(10);
  setting["max_doc_len"] = SettingV(31);
  setting["min_doc_len"] = SettingV(5);
  
  /// Test TextData Layer
  Layer<cpu> * layer_textdata = CreateLayer<cpu>(kTextData);
  layer_textdata->PropAll();
  layer_textdata->SetupLayer(setting, bottoms, tops);
  layer_textdata->Reshape(bottoms, tops);
  layer_textdata->Forward(bottoms, tops);
  layer_textdata->Backprop(bottoms, tops);

  cout << "top1 size: " << top1.data.shape_[0] << "x" << top1.data.shape_[1] << "x" << top1.data.shape_[2] << "x" << top1.data.shape_[3] << " : " << top1.data.stride_ << endl;
  
  cout << "top1 data: " << top1.data[0][0][0][0] << endl;

  PrintTensor(top1.data);

  cout << "top2 size: " << top2.data.shape_[0] << "x" << top2.data.shape_[1] << "x" << top2.data.shape_[2] << "x" << top2.data.shape_[3] << " : " << top2.data.stride_ << endl;
  
  cout << "top2 data: " << top2.data[0][0][0][0] << endl;

  PrintTensor(top2.data);
}

void TestActivationLayer() {
  Node<cpu> bottom;
  Node<cpu> top;
  vector<Node<cpu>*> bottoms;
  vector<Node<cpu>*> tops;
  
  bottoms.push_back(&bottom);
  tops.push_back(&top);
  
  bottom.data.Resize(Shape4(2,1,5,5), 1.0);
  bottom.diff.Resize(Shape4(2,1,5,5), 0.0);
  
  cout << "Init test ok." << endl;

  map<string, SettingV> setting;
  //setting["layer_type"] = SettingV(kRectifiedLinear);
  
  /// Test Activation Layer
  Layer<cpu> * layer_rectify = CreateLayer<cpu>(kRectifiedLinear);
  layer_rectify->PropAll();
  layer_rectify->SetupLayer(setting, bottoms, tops);
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
  layer_sigmoid->SetupLayer(setting, bottoms, tops);
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
  TestTextDataLayer();
  return 0;
}

