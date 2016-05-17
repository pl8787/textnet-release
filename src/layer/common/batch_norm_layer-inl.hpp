#ifndef TEXTNET_LAYER_BATCH_NORM_LAYER_INL_HPP_
#define TEXTNET_LAYER_BATCH_NORM_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class BatchNormLayer : public Layer<xpu> {
 public:
  BatchNormLayer(LayerType type) { this->layer_type = type; }
  virtual ~BatchNormLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 5; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["eps"] = SettingV(1e-5f);
    this->defaults["moving_average_fraction"] = SettingV(0.99f);
    this->defaults["ignore_len"] = SettingV(true);

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["w_filler"] = SettingV();
    this->defaults["b_filler"] = SettingV();
    this->defaults["w_updater"] = SettingV();
    this->defaults["b_updater"] = SettingV();
    
    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchNormLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchNormLayer:top size problem.");
                            
    moving_average_fraction = setting["moving_average_fraction"].fVal();
    eps = setting["eps"].fVal();
    ignore_len = setting["ignore_len"].bVal();

    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    nbatch = shape_in[0];
    feat_count = shape_in[1];
    feat_size = shape_in[0] * shape_in[2] * shape_in[3];
    bias_correction_factor = feat_size > 1 ? (float)feat_size/(feat_size-1) : 1;

    this->params.resize(5);
    this->params[0].Resize(feat_count, 1, 1, 1, true);
    this->params[1].Resize(feat_count, 1, 1, 1, true);
    this->params[2].Resize(feat_count, 1, 1, 1, true);
    this->params[3].Resize(feat_count, 1, 1, 1, true);
    this->params[4].Resize(1, 1, 1, 1, true);

    this->params[2].data = 0.0f;
    this->params[3].data = 0.0f;
    this->params[4].data = 0.0f;
    
    std::map<std::string, SettingV> &w_setting = *setting["w_filler"].mVal();
    std::map<std::string, SettingV> &b_setting = *setting["b_filler"].mVal();
    this->params[0].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(w_setting["init_type"].iVal(),
          w_setting, this->prnd_);
    this->params[1].initializer_ = 
        initializer::CreateInitializer<xpu, 4>(b_setting["init_type"].iVal(), 
          b_setting, this->prnd_);
    this->params[0].Init();
    this->params[1].Init();
    
    std::map<std::string, SettingV> &w_updater = *setting["w_updater"].mVal();
    std::map<std::string, SettingV> &b_updater = *setting["b_updater"].mVal();
    this->params[0].updater_ = 
        updater::CreateUpdater<xpu, 4>(w_updater["updater_type"].iVal(),
          w_updater, this->prnd_);
    this->params[1].updater_ = 
        updater::CreateUpdater<xpu, 4>(b_updater["updater_type"].iVal(),
          b_updater, this->prnd_);
    
    // Ignore setting parameter 2 and 3
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
                       bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(),
                  "BatchNormLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "BatchNormLayer:top size problem.");
    
    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;

    nbatch = shape_in[0];
    feat_count = shape_in[1];
    feat_size = shape_in[0] * shape_in[2] * shape_in[3];
    bias_correction_factor = feat_size > 1 ? (float)feat_size/(feat_size-1) : 1;

    running_mean_var_.Resize(feat_count, 1, 1, 1, 1, 1, true);
    diff_mean_.Resize(feat_count, 1, 1, 1, 1, 1, true);

    feat_size_sum_multiplier_.Resize(mshadow::Shape2(feat_size, 1));
    feat_size_sum_multiplier_ = 1.0f;

    bottom_mat_.Resize(shape_in[1], shape_in[0], shape_in[2], shape_in[3], 1, 1, true);
    top_mat_.Resize(shape_in[1], shape_in[0], shape_in[2], shape_in[3], 1, 1, true);

    top[0]->Resize(bottom[0]->data.shape_, bottom[0]->length.shape_, true);

    if (show_info) {
        bottom[0]->PrintShape("bottom0");
        top[0]->PrintShape("top0");
    }
  }

  virtual void CheckReshape(const std::vector<Node<xpu>*> &bottom,
                            const std::vector<Node<xpu>*> &top) {
    // Check for reshape
    bool need_reshape = false;
    if (nbatch != bottom[0]->data.size(0)) {
        need_reshape = true;
    }

    // Do reshape 
    if (need_reshape) {
        this->Reshape(bottom, top);
    }
  }
  
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

  void do_arrange(mshadow::Tensor<xpu, 4> &from_tensor, mshadow::Tensor<xpu, 4> &to_matrix, 
          mshadow::Tensor<xpu, 2> &length, int nbatch, int feat_count, int feat_size) {
    int v_idx = 0;
    for (int c = 0; c < feat_count; ++c) {
      for (int i = 0; i < nbatch; ++i) {
        if (length.shape_[1] == 1){
          for (int x = 0; x < length[i][0]; ++x) {
            to_matrix[c][v_idx++][0][0] = from_tensor[i][c][0][x];
          }
        } else if (length.shape_[1] == 2) {
          for (int y = 0; y < length[i][0]; ++y) {
            for (int x = 0; x < length[i][1]; ++x) {
              to_matrix[c][v_idx++][0][0] = from_tensor[i][c][y][x];
            }
          }
        } else {
          utils::Check(false, "In BatchNormLayer: bottom length size error");
        }
      }
    }
    utils::Check(v_idx==feat_size, "In BatchNormLayer: v_idx=%d, feat_size=%d.", v_idx, feat_size);
  }

  void undo_arrange(mshadow::Tensor<xpu, 4> &from_matrix, mshadow::Tensor<xpu, 4> &to_tensor, 
          mshadow::Tensor<xpu, 2> &length, int nbatch, int feat_count, int feat_size, bool do_plus=false) {
    int v_idx = 0;
    for (int c = 0; c < feat_count; ++c) {
      for (int i = 0; i < nbatch; ++i) {
        if (length.shape_[1] == 1){
          for (int x = 0; x < length[i][0]; ++x) {
            if (do_plus) {
              to_tensor[i][c][0][x] += from_matrix[c][v_idx++][0][0] ;
            } else {
              to_tensor[i][c][0][x] = from_matrix[c][v_idx++][0][0] ;
            }
          }
        } else if (length.shape_[1] == 2) {
          for (int y = 0; y < length[i][0]; ++y) {
            for (int x = 0; x < length[i][1]; ++x) {
              if (do_plus) {
                to_tensor[i][c][y][x] += from_matrix[c][v_idx++][0][0];
              } else {
                to_tensor[i][c][y][x] = from_matrix[c][v_idx++][0][0];
              }
            }
          }
        } else {
          utils::Check(false, "In BatchNormLayer: bottom length size error");
        }
      }
    }
    utils::Check(v_idx==feat_size, "In BatchNormLayer: v_idx=%d, feat_size=%d.", v_idx, feat_size);
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;

    mshadow::Tensor<xpu, 1> gamma = this->params[0].data_d1();
    mshadow::Tensor<xpu, 1> beta = this->params[1].data_d1();

    mshadow::Tensor<xpu, 2> running_mean_d2 = running_mean_var_.data_d2();
    mshadow::Tensor<xpu, 1> running_mean_d1 = running_mean_var_.data_d1();
    mshadow::Tensor<xpu, 2> running_var_d2 = running_mean_var_.diff_d2();
    mshadow::Tensor<xpu, 1> running_var_d1 = running_mean_var_.diff_d1();

    mshadow::Tensor<xpu, 2> stored_mean_d2 = this->params[2].data_d2();
    mshadow::Tensor<xpu, 1> stored_mean_d1 = this->params[2].data_d1();
    mshadow::Tensor<xpu, 2> stored_var_d2 = this->params[3].data_d2();
    mshadow::Tensor<xpu, 1> stored_var_d1 = this->params[3].data_d1();

    mshadow::Tensor<xpu, 1> scale = this->params[4].data_d1();

    top_len = F<op::identity>(bottom_len);
    
    if (ignore_len) {
      bottom_mat_.data = swapaxis<1, 0>(bottom_data);
    } else {
      // Calculate feat size
      feat_size = 0;
      for (int i = 0; i < nbatch; ++i) {
        if (bottom_len.shape_[1] == 1){
          feat_size += bottom_len[i][0];
        } else if (bottom_len.shape_[1] == 2) {
          feat_size += bottom_len[i][0] * bottom_len[i][1];
        } else {
          utils::Check(false, "In BatchNormLayer: bottom length size error");
        }
      }

      // Reshape
      feat_size_sum_multiplier_.Resize(mshadow::Shape2(feat_size, 1));
      feat_size_sum_multiplier_ = 1.0f;

      bottom_mat_.Resize(feat_count, feat_size, 1, 1, 1, 1, true);
      top_mat_.Resize(feat_count, feat_size, 1, 1, 1, 1, true);

      // Rearrage values
      do_arrange(bottom_data, bottom_mat_.data, bottom_len, nbatch, feat_count, feat_size);
    }

    mshadow::Tensor<xpu, 2> bottom_mat_data = bottom_mat_.data_d2();
    mshadow::Tensor<xpu, 2> bottom_mat_data2 = bottom_mat_.diff_d2();
    
    use_stored_param = this->phrase_type == kTest;
    if (use_stored_param) {
      running_mean_d2 = stored_mean_d2 / scale[0];
      running_var_d2 = stored_var_d2 / scale[0];
    } else {
      // Compute running mean
      running_mean_d2 = dot(bottom_mat_data, feat_size_sum_multiplier_);
      running_mean_d2 /= feat_size;
      // Compute running var
      bottom_mat_data2 = bottom_mat_data * bottom_mat_data;
      running_var_d2 = dot(bottom_mat_data2, feat_size_sum_multiplier_);
      running_var_d2 /= feat_size;
      running_var_d2 -= running_mean_d2 * running_mean_d2;

      // Update stored params
      scale *= moving_average_fraction;
      scale += 1.0f;
      stored_mean_d1 = moving_average_fraction * stored_mean_d1 + running_mean_d1;
      stored_var_d1 = moving_average_fraction * stored_var_d1 + bias_correction_factor * running_var_d1; 
    }

    // Normalize Var
    running_var_d1 += eps;
    running_var_d1 = F<op::square_root>(running_var_d1);
    // Compute output
    mshadow::Tensor<xpu, 2> top_mat_data = top_mat_.data_d2();
    // top_mat_data = F<op::identity>(bottom_mat_data);

    for (int i = 0; i < feat_count; ++i) {
      for (int j = 0; j < feat_size; ++j) {
        bottom_mat_data[i][j] = (bottom_mat_data[i][j] - running_mean_d1[i]) / running_var_d1[i];
        top_mat_data[i][j] = gamma[i] * bottom_mat_data[i][j] + beta[i];
      }
    }
    // top_mat_data -= repmat(running_mean_d1, feat_size);
    // top_mat_data /= repmat(running_var_d1, feat_size);

    // top_mat_data *= repmat(gamma, feat_size);
    // top_mat_data += repmat(beta, feat_size);

    if (ignore_len) {
      top_data = swapaxis<1, 0>(top_mat_.data);
    } else {
      //Rearrange values
      undo_arrange(top_mat_.data, top_data, bottom_len, nbatch, feat_count, feat_size);
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 2> bottom_len = bottom[0]->length;
    mshadow::Tensor<xpu, 2> top_len = top[0]->length;

    mshadow::Tensor<xpu, 1> gamma = this->params[0].data_d1();
    mshadow::Tensor<xpu, 1> beta = this->params[1].data_d1();
    mshadow::Tensor<xpu, 2> gamma_diff = this->params[0].diff_d2();
    mshadow::Tensor<xpu, 2> beta_diff = this->params[1].diff_d2();

    mshadow::Tensor<xpu, 2> diff_mean_d2 = diff_mean_.data_d2();
    mshadow::Tensor<xpu, 1> diff_mean_d1 = diff_mean_.data_d1();
    mshadow::Tensor<xpu, 2> diff_y_mean_d2 = diff_mean_.diff_d2();
    mshadow::Tensor<xpu, 1> diff_y_mean_d1 = diff_mean_.diff_d1();

    if (ignore_len) {
      top_mat_.diff = swapaxis<1, 0>(top_diff);
    } else {
      // Rearrage values
      do_arrange(top_diff, top_mat_.diff, bottom_len, nbatch, feat_count, feat_size);
    }
    mshadow::Tensor<xpu, 2> top_mat_diff = top_mat_.diff_d2();

    mshadow::Tensor<xpu, 2> bottom_mat_data = bottom_mat_.data_d2();
    mshadow::Tensor<xpu, 2> temp_ = bottom_mat_.diff_d2();

    // Compute grad of gamma
    temp_ = bottom_mat_data * top_mat_diff;
    gamma_diff = dot(temp_, feat_size_sum_multiplier_);

    // Compute grad of beta
    beta_diff = dot(top_mat_diff, feat_size_sum_multiplier_);

    // Compute diff mean
    diff_mean_d2 = beta_diff / feat_size;

    // Compute diff*y mean = gamma_diff
    diff_y_mean_d2 = gamma_diff / feat_size;
    
    mshadow::Tensor<xpu, 2> bottom_mat_diff = bottom_mat_.diff_d2();
    mshadow::Tensor<xpu, 1> running_var_d1 = running_mean_var_.diff_d1();

    // PrintTensor("top_mat_diff", top_mat_diff);
    // PrintTensor("diff_mean_d1", diff_mean_d1);
    // PrintTensor("diff_y_mean_d1", diff_y_mean_d1);
    // PrintTensor("running_var_d1", running_var_d1);
    // PrintTensor("gamma", gamma);

    for (int i = 0; i < feat_count; ++i) {
      for (int j = 0; j < feat_size; ++j) {
        bottom_mat_diff[i][j] = (top_mat_diff[i][j] - diff_mean_d1[i] - diff_y_mean_d1[i] * bottom_mat_data[i][j]) * gamma[i] / running_var_d1[i]; 
      }
    }

    if (ignore_len) {
      bottom_diff += swapaxis<1, 0>(bottom_mat_.diff);
    } else {
      //Rearrange values
      undo_arrange(bottom_mat_.diff, bottom_diff, bottom_len, nbatch, feat_count, feat_size, true);
    }
  }

 protected:
  /*! \brief random number generator */
  Node<xpu> bottom_mat_;
  Node<xpu> top_mat_;
  Node<xpu> running_mean_var_;
  Node<xpu> diff_mean_;
  mshadow::TensorContainer<xpu, 2> feat_size_sum_multiplier_;
  float moving_average_fraction;
  float bias_correction_factor;
  float eps;
  int nbatch;
  int feat_count;
  int feat_size;
  bool use_stored_param;
  bool ignore_len;

};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_BATCH_NORM_LAYER_INL_HPP_
