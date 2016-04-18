#ifndef TEXTNET_LAYER_IMAGE_LAYER_INL_HPP_
#define TEXTNET_LAYER_IMAGE_LAYER_INL_HPP_

#include <iostream>
#include <fstream>
#include <sstream>

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class ImageLayer : public Layer<xpu>{
 public:
  ImageLayer(LayerType type) { this->layer_type = type; }
  virtual ~ImageLayer(void) {}
  
  virtual int BottomNodeNum() { return 0; }
  virtual int TopNodeNum() { return 2; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want
	this->defaults["shuffle"] = SettingV(false);
	this->defaults["scale"] = SettingV(1.0f);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["data_file"] = SettingV();
    this->defaults["batch_size"] = SettingV();
    this->defaults["height"] = SettingV();
	this->defaults["width"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ImageLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ImageLayer:top size problem.");

    data_file = setting["data_file"].sVal();
    batch_size = setting["batch_size"].iVal();
	shuffle = setting["shuffle"].bVal();
	height = setting["height"].iVal();
	width = setting["width"].iVal();
	scale = setting["scale"].fVal();
    
    ReadImageData(); // Only for mnist
    
    line_ptr = 0;
  }
  
  void ReadImageData() {
    utils::Printf("Open data file: %s\n", data_file.c_str());    
    std::vector<std::string> lines;
    std::ifstream fin(data_file.c_str());
    std::string s;
    utils::Check(fin.is_open(), "Open data file problem.");
    while (!fin.eof()) {
      std::getline(fin, s);
      if (s.empty()) break;
      lines.push_back(s);
    }
    fin.close();
    
    line_count = lines.size();
    data_set.Resize(mshadow::Shape3(line_count, height, width));
    length_set.Resize(mshadow::Shape2(line_count, 2));
    label_set.Resize(mshadow::Shape1(line_count), 0);
    data_set = -1;
	length_set = 0;
    
    utils::Printf("Line count in file: %d\n", line_count);

    std::istringstream iss;
    for (int i = 0; i < line_count; ++i) {
      iss.clear();
        iss.seekg(0, iss.beg);
      iss.str(lines[i]);
      iss >> label_set[i];
	  length_set[i][1] = width;
	  length_set[i][2] = height;
      for (int j = 0; j < height; ++j) {
		for (int k = 0; k < width; ++k) {
		  iss >> data_set[i][j][k];
		}
      }
    }

	data_set = data_set * scale;
    
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ImageLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ImageLayer:top size problem.");
	
	utils::Check(batch_size > 0, "batch_size <= 0");

    top[0]->Resize(batch_size, 1, height, width, batch_size, 2, true);
    top[1]->Resize(batch_size, 1, 1, 1, true);
	
	if (show_info) {
		top[0]->PrintShape("top0");
		top[1]->PrintShape("top1");
	}
  }
  
  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top0_data = top[0]->data;
    mshadow::Tensor<xpu, 2> top0_length = top[0]->length;
    mshadow::Tensor<xpu, 1> top1_data = top[1]->data_d1();
    for (int i = 0; i < batch_size; ++i) {
      if (shuffle) {
        line_ptr = rand() % line_count;
      } 
      top0_data[i][0] = F<op::identity>(data_set[line_ptr]);
	  top0_length[i] = F<op::identity>(length_set[line_ptr]);
      top1_data[i] = label_set[line_ptr];
      line_ptr = (line_ptr + 1) % line_count;
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    
  }
  
 protected:
  std::string data_file;
  int batch_size;
  bool shuffle;
  int height;
  int width;
  float scale;

  mshadow::TensorContainer<xpu, 3> data_set;
  mshadow::TensorContainer<xpu, 2> length_set;
  mshadow::TensorContainer<xpu, 1> label_set;
  int line_count;
  int line_ptr;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_IMAGE_LAYER_INL_HPP_

