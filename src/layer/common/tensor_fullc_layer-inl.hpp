#ifndef TEXTNET_LAYER_TENSOR_FULLC_LAYER_INL_HPP_
#define TEXTNET_LAYER_TENSOR_FULLC_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class TensorFullConnectLayer : public Layer<xpu> {
 public:
  TensorFullConnectLayer(LayerType type) { this->layer_type = type; }
  virtual ~TensorFullConnectLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 3; }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  
  virtual void Require() {
    this->defaults["mode"] = SettingV("t1w1b0"); // string value, 't*w*b*', * is 1 or 0
    // default value, just set the value you want
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["num_hidden"] = SettingV();
    this->defaults["t_filler"] = SettingV();
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["t_updater"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(), "TensorFullConnectionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "TensorFullConnectionLayer:top size problem.");
                            
    this->param_file = setting["param_file"].sVal();
    std::string mode = setting["mode"].sVal();
    if (mode[1] == '1') {
      is_t = true;
    } else if (mode[1] == '0') {
      is_t = false;
    } else {
      utils::Check(false, "TensorFullConnectionLayer: mode error.");
    }
    if (mode[3] == '1') {
      is_w = true;
    } else if (mode[3] == '0') {
      is_w = false;
    } else {
      utils::Check(false, "TensorFullConnectionLayer: mode error.");
    }
    if (mode[5] == '1') {
      is_b = true;
    } else if (mode[5] == '0') {
      is_b = false;
    } else {
      utils::Check(false, "TensorFullConnectionLayer: mode error.");
    }

    d_hidden = setting["num_hidden"].iVal();

    Tensor2D bottom_data = bottom[0]->data_d2();
    d_input = bottom_data.size(1);

    this->params.resize(3);
    this->params[0].Resize(d_hidden, d_input, d_input, 1, true);
    this->params[1].Resize(d_hidden, d_input, 1, 1, true);
    this->params[2].Resize(d_hidden, 1, 1, 1, true);
    
    std::map<std::string, SettingV> &t_setting = *setting["t_filler"].mVal();
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(t_setting["init_type"].iVal(),
          t_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
          w_setting, this->prnd_);
    this->params[2].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), 
          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    this->params[2].Init();
    
    std::map<std::string, SettingV> &t_updater = *setting["t_updater"].mVal();
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(t_updater["updater_type"].iVal(),
          t_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
          w_updater, this->prnd_);
    this->params[2].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
          b_updater, this->prnd_);
    if (!this->param_file.empty()) {
      this->LoadParams();
    }
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "TensorFullConnectionLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "TensorFullConnectionLayer:top size problem.");
    
    Tensor2D bottom_data = bottom[0]->data_d2();
    
    batch_size = bottom_data.size(0); 
    out_product.Resize(batch_size, d_input, d_input, 1, true);
    top[0]->Resize(batch_size, d_hidden, 1, 1, true);

	if (show_info) {
		bottom[0]->PrintShape("bottom0");
		top[0]->PrintShape("top0");
	}
  }
  
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor2D bottom_data = bottom[0]->data_d2();
    Tensor2D top_data = top[0]->data_d2();

    top_data = 0.f;
    if (is_t) {
      Tensor3D out_prod = out_product.data_d3();
      for (index_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        Tensor2D one_example = bottom_data.Slice(batch_idx, batch_idx+1);
        out_prod[batch_idx] = dot(one_example.T(), one_example);
      }
      Tensor2D t = this->params[0].data_d2();
      Tensor2D input = out_product.data_d2();
      top_data += dot(input, t.T());
    }
    if (is_w) {
      top_data += dot(bottom_data, this->params[1].data_d2().T());
    }
    if (is_b) {
      top_data += repmat(this->params[2].data_d1(), batch_size);
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    Tensor2D top_diff = top[0]->diff_d2();
    Tensor2D bottom_data = bottom[0]->data_d2();
    Tensor2D bottom_diff = bottom[0]->diff_d2();

    if (is_t) {
      this->params[0].diff_d2() += dot(top_diff.T(), out_product.data_d2());
      out_product.diff_d2() = dot(top_diff, this->params[0].data_d2());
      Tensor3D out_prod = out_product.diff_d3();
      for (index_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        Tensor2D data = bottom_data.Slice(batch_idx, batch_idx+1);
        Tensor2D diff = bottom_diff.Slice(batch_idx, batch_idx+1);
        diff += dot(data, out_prod[batch_idx]);
        diff += dot(data, out_prod[batch_idx].T());
      }
    }
    if (is_w) {
      this->params[1].diff_d2() += dot(top_diff.T(), bottom_data);
      bottom_diff += dot(top_diff, this->params[1].data_d2());
    }
    if (is_b) {
      this->params[2].diff_d1() += sum_rows(top_diff);
    }
  }

 protected:
  /*! \brief random number generator */
  int d_input, d_hidden, batch_size;
  bool is_t, is_w, is_b;
  Node<xpu> out_product; 
};
}  // namespace layer
}  // namespace textnet
#endif 
