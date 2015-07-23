#ifndef TEXTNET_FILE_INIT_INL_HPP_
#define TEXTNET_FILE_INIT_INL_HPP_

#include <mshadow/tensor.h>
#include "./initializer.h"
#include <fstream>

namespace textnet {
namespace initializer {

template<typename xpu, int dim>
class FileInitializer : public Initializer<xpu, dim>{
 public:
  FileInitializer(std::map<std::string, SettingV> &setting, 
                      mshadow::Random<xpu>* prnd) {
    this->prnd_ = prnd;
    SetupInitializer(setting);
  }
  virtual ~FileInitializer(void) {}
  
  virtual void Require(std::map<std::string, SettingV> &setting) {
    // default value, just set the value you want
    
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["file_path"] = SettingV();
    
    Initializer<xpu, dim>::Require(setting);
  }
  
  virtual void SetupInitializer(std::map<std::string, SettingV> &setting) {
    Initializer<xpu, dim>::SetupInitializer(setting);
    
    this->init_type = setting["init_type"].iVal();
    file_path = setting["file_path"].sVal();
  }
  
  virtual void DoInitialize(mshadow::Tensor<xpu, dim> data) {
    std::ifstream ifs(file_path.c_str());
    vector<float> vals;
    while (!ifs.eof()) {
        float s = 0;
        ifs >> s;
        vals.push_back(s);
    }
    ifs.close();
    utils::Check(vals.size() == data.shape_.Size(), "FileInitializer: parameter file error.");

    mshadow::Tensor<xpu, 2, float> mat = data.FlatTo2D();
    index_t idx = 0;
    for (index_t i = 0; i < mat.size(0); ++i) {
      for (index_t j = 0; j < mat.size(1); ++j) {
        mat[i][j] = vals[idx];
        idx += 1;
      }
    }
  }
  
  string file_path;
};
}  // namespace initializer
}  // namespace textnet
#endif  // TEXTNET_CONSTANT_INIT_INL_HPP_

