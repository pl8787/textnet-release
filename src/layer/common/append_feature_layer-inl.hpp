#ifndef TEXTNET_LAYER_APPEND_FEATURE_LAYER_INL_HPP_
#define TEXTNET_LAYER_APPEND_FEATURE_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class AppendFeatureLayer : public Layer<xpu>{
 public:
  AppendFeatureLayer(LayerType type) { this->layer_type = type; }
  virtual ~AppendFeatureLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
void PrintTensor(const char * name, mshadow::Tensor<xpu, 1> x) {
    mshadow::Shape<1> s = x.shape_;
    cout << name << " shape " << s[0] << endl;
    for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
      cout << x[d1] << " ";
    }
    cout << endl;
}

void PrintTensor(const char * name, mshadow::Tensor<xpu, 2> x) {
    mshadow::Shape<2> s = x.shape_;
    cout << name << " shape " << s[0] << "x" << s[1] << endl;
    for (unsigned int d1 = 0; d1 < s[0]; ++d1) {
      for (unsigned int d2 = 0; d2 < s[1]; ++d2) {
        cout << x[d1][d2] << " ";
      }
      cout << endl;
    }
    cout << endl;
}

void PrintTensor(const char * name, mshadow::Tensor<xpu, 3> x) {
    mshadow::Shape<3> s = x.shape_;
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

void PrintTensor(const char * name, mshadow::Tensor<xpu, 4> x) {
    mshadow::Shape<4> s = x.shape_;
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
}

  virtual void Require() {
    // default value, just set the value you want
    this->defaults["D0"] = SettingV(0);
    this->defaults["D1"] = SettingV(0);
    this->defaults["D2"] = SettingV(0);
    this->defaults["D3"] = SettingV(0);
    this->defaults["feature_mode"] = SettingV("Number"); // Number or Category
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["feature_file"] = SettingV();
    this->defaults["key"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), 
        "AppendFeatureLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), 
        "AppendFeatureLayer:top size problem.");

    // Dx represents data axis x shape 
    // Lx represents length axis x shape
    // if set to 0, copy the shape size as bottom axis
    // if set to -1, automatically compute shape. 
    //   So only one -1 can be set, and its value can be derived from other values.
    D0 = setting["D0"].iVal();
    D1 = setting["D1"].iVal();
    D2 = setting["D2"].iVal();
    D3 = setting["D3"].iVal();

    feature_mode = setting["feature_mode"].sVal();
    feature_file = setting["feature_file"].sVal();
    key = setting["key"].sVal();

    ReadFeatureMap(feature_file, feature_map);

  }

  void ReadFeatureMap(string &feature_file, unordered_map<string, vector<float> > &feature_map) { 
    utils::Printf("Open data file: %s\n", feature_file.c_str());    

    std::ifstream fin(feature_file.c_str());
    std::string s;
    std::string key;
    std::string last_key;
    float value;
    utils::Check(fin.is_open(), "AppendFeatureLayer: Open data file problem.");

    istringstream iss;
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      
      iss.clear();
      iss.seekg(0, iss.beg);
      iss.str(s);
      iss >> key;

      feature_map[key] = vector<float>();

      last_key = key;
      while(!iss.eof()) {
        iss >> value;
        feature_map[key].push_back(value);
      }

      utils::Check(feature_map[key].size() == D1 * D2 * D3, 
              "AppendFeatureLayer: Feature length %d not equal to %d, on line %d.", feature_map[key].size(), D1 * D2 * D3, feature_map.size());
    }
    fin.close();

    std::cout << last_key.c_str() << " ";
    for (int i = 0; i < feature_map[last_key].size(); ++i) {
        std::cout << feature_map[last_key][i] << " ";
    }
    std::cout << std::endl;

    utils::Printf("Line count in file: %d\n", feature_map.size());
  }

  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), 
        "AppendFeatureLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), 
        "AppendFeatureLayer:top size problem.");
                  
    in_data_shape = bottom[0]->data.shape_;
    in_len_shape = bottom[0]->length.shape_;

    utils::Check(in_data_shape[1] == D1, "AppendFeatureLayer: D1 not match.");
    utils::Check(in_data_shape[2] == D2, "AppendFeatureLayer: D2 not match.");

    D0 = in_data_shape[0];
    out_data_shape[0] = in_data_shape[0];
    out_data_shape[1] = D1;
    out_data_shape[2] = D2;
    out_data_shape[3] = in_data_shape[3] + D3;

    out_len_shape = in_len_shape;

    top[0]->Resize(out_data_shape, out_len_shape, true);

    if (show_info) {
      bottom[0]->PrintShape("bottom0");
      top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (in_data_shape.Size() != bottom[0]->data.shape_.Size()) {
      need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
      this->Reshape(bottom, top);
    }
  }
 
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> bottom0_data = bottom[0]->data_d3_middle();
    mshadow::Tensor<xpu, 1> bottom0_len = bottom[0]->length_d1();
    mshadow::Tensor<xpu, 3> top0_data = top[0]->data_d3_middle();
    mshadow::Tensor<xpu, 1> top0_len = top[0]->length_d1();
    
    top0_len = F<op::identity>(bottom0_len);
    
    // Check for global data
    utils::Check(Layer<xpu>::global_data.count(key), 
            "AppendFeatureLayer: %s key not found in global data.", key.c_str());

    for (int i = 0; i < D0; ++i) {
      // Extract key string from global data
      string cur_key = Layer<xpu>::global_data[key][i];
      utils::Check(feature_map.count(cur_key), 
            "AppendFeatureLayer: %s key not found in feature map.", cur_key.c_str());
      vector<float> &cur_feature_map = feature_map[cur_key];
      int p_kk = 0;
      for (int j = 0; j < D1*D2; ++j) {
        for (int k = 0; k < in_data_shape[3]; ++k) {
          top0_data[i][j][k] = bottom0_data[i][j][k];
        }
        for (int k = in_data_shape[3], kk = 0; kk < D3; ++k, ++kk, ++p_kk) {
          top0_data[i][j][k] = cur_feature_map[p_kk];
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 3> bottom0_diff = bottom[0]->diff_d3_middle();
    mshadow::Tensor<xpu, 3> top0_diff = top[0]->diff_d3_middle();

    for (int i = 0; i < D0; ++i) {
      for (int j = 0; j < D1*D2; ++j) {
        for (int k = 0; k < in_data_shape[3]; ++k) {
          bottom0_diff[i][j][k] = top0_diff[i][j][k];
        }
      }
    }

  }
  
 protected:
    int D0;
    int D1;
    int D2;
    int D3;
    string feature_file;
    string feature_mode;
    string key;
    unordered_map<string, vector<float> > feature_map;

    mshadow::Shape<4> in_data_shape;
    mshadow::Shape<2> in_len_shape;
    mshadow::Shape<4> out_data_shape;
    mshadow::Shape<2> out_len_shape;

};
}  // namespace layer
}  // namespace textnet
#endif  

