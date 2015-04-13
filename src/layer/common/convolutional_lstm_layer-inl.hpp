#ifndef TEXTNET_LAYER_CONVOLUTIONAL_LSTM_LAYER_INL_HPP_
#define TEXTNET_LAYER_CONVOLUTIONAL_LSTM_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"
#include "../../utils/utils.h"

namespace textnet {
namespace layer {

template<typename xpu>
class ConvolutionalLstmLayer : public Layer<xpu> {
 public:
  ConvolutionalLstmLayer(LayerType type) { this->layer_type = type; }
  virtual ~ConvolutionalLstmLayer(void) {}
  
  virtual int BottomNodeNum() { return 3; } // l2r_lstm word r2l_lstm
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 2; }
  
  virtual void Require() {
    // default value, just set the value you want
    this->defaults["no_bias"] = SettingV(false);
    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["num_hidden"] = SettingV();
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
    
    utils::Check(bottom.size() == BottomNodeNum(), "ConvolutionalLstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ConvolutionalLstmLayer:top size problem.");
                            
    num_hidden = setting["num_hidden"].iVal();
    no_bias = setting["no_bias"].bVal();

    int rep_dim_l2r  = bottom[0]->data.size(3);
    int rep_dim_word = bottom[1]->data.size(3);
    int rep_dim_r2l  = bottom[2]->data.size(3);
    num_input = rep_dim_l2r + rep_dim_word + rep_dim_r2l;

    this->params.resize(2);
    this->params[0].Resize(num_hidden, num_input, 1, 1, true);
    this->params[1].Resize(num_hidden, 1, 1, 1, true);
    
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
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "ConvolutionalLstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "ConvolutionalLstmLayer:top size problem.");
    utils::Check(bottom[0]->data.size(0) == bottom[1]->data.size(0), "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[0]->data.size(0) == bottom[2]->data.size(0), "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[0]->data.size(1) == bottom[1]->data.size(1), "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[0]->data.size(1) == bottom[2]->data.size(1), "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[0]->data.size(2) == bottom[1]->data.size(2), "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[0]->data.size(2) == bottom[2]->data.size(2), "ConvolutionalLstmLayer:bottom size problem.");
    
    int batch_size = bottom[0]->data.size(0);
    int num_seq = bottom[0]->data.size(1);
    int length = bottom[0]->data.size(2);
    
    top[0]->Resize(batch_size, num_seq, length, num_hidden, true);
    concat_input_data.Resize(mshadow::Shape4(batch_size, num_seq, length, num_input));
    concat_input_diff.Resize(mshadow::Shape4(batch_size, num_seq, length, num_input));
    concat_input_data = 0.f;
    concat_input_diff = 0.f;

	bottom[0]->PrintShape("ConvolutionalLstm: bottom_0");
	bottom[1]->PrintShape("ConvolutionalLstm: bottom_1");
	bottom[2]->PrintShape("ConvolutionalLstm: bottom_2");
	top[0]->PrintShape("ConvolutionalLstm: top_0");
  }

  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 1> Tensor1D;

  virtual void Concat(Tensor2D l2r_rep, Tensor2D word_rep, Tensor2D r2l_rep, Tensor2D cc_rep) {
    utils::Assert((l2r_rep.size(1)+word_rep.size(1)+r2l_rep.size(1)) == cc_rep.size(1), "ConvolutionalLstmLayer:size problem.");

    int dim_left = l2r_rep.size(1);
    int dim_middle = word_rep.size(1);
    int dim_right = r2l_rep.size(1);

    int begin = 0, end = l2r_rep.size(0);
    cc_rep = 0.; // init
    // for (int batch_idx = 0; batch_idx < l2r_rep.size(0); ++batch_idx) {
    for (int word_idx = begin; word_idx < end; ++word_idx) {
        Tensor1D row_l2r, row_word, row_r2l, row_cc;

        row_cc = cc_rep[word_idx].Slice(0, dim_left);
        if (word_idx != begin) {
            row_l2r  = l2r_rep[word_idx-1];
            row_cc = mshadow::expr::F<op::identity>(row_l2r);
        }

        row_cc = cc_rep[word_idx].Slice(dim_left, dim_left+dim_middle);
        row_word = word_rep[word_idx];
        row_cc = mshadow::expr::F<op::identity>(row_word);

        row_cc = cc_rep[word_idx].Slice(dim_left+dim_middle, dim_left+dim_middle+dim_right);
        if (word_idx != end-1) {
            row_r2l  = r2l_rep[word_idx+1];
            row_cc = mshadow::expr::F<op::identity>(row_r2l);
        }
    }
    // }
  }
  virtual void Split(Tensor2D cc_diff, Tensor2D l2r_diff, Tensor2D word_diff, Tensor2D r2l_diff) {
    utils::Assert((l2r_diff.size(1)+word_diff.size(1)+r2l_diff.size(1)) == cc_diff.size(1), "ConvolutionalLstmLayer:size problem.");

    int dim_left = l2r_diff.size(1);
    int dim_middle = word_diff.size(1);
    int dim_right = r2l_diff.size(1);

    // for (int batch_idx = 0; batch_idx < l2r_diff.size(0); ++batch_idx) {
    int begin = 0, end = cc_diff.size(0);

    for (int word_idx = begin; word_idx < end; ++word_idx) {
        Tensor1D row_l2r, row_word, row_r2l, row_cc;

        row_cc = cc_diff[word_idx].Slice(0, dim_left);
        if (word_idx != begin) {
            row_l2r = l2r_diff[word_idx-1];
            row_l2r += row_cc;
        }

        row_cc = cc_diff[word_idx].Slice(dim_left, dim_left+dim_middle);
        row_word = word_diff[word_idx];
        row_word += row_cc;

        row_cc = cc_diff[word_idx].Slice(dim_left+dim_middle, dim_left+dim_middle+dim_right);
        if (word_idx != end-1) {
            row_r2l  = r2l_diff[word_idx+1];
            row_r2l += row_cc;
        }
    }
    // }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

    top[0]->data = 0.f; // init
    top[0]->length = F<op::identity>(bottom[0]->length); // init

    for (index_t batch_idx = 0; batch_idx < concat_input_data.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < concat_input_data.size(1); ++seq_idx) {
        int len = bottom[0]->length[batch_idx][seq_idx];
        utils::Assert(len >= 0, "ConvolutionalLstmLayer: sequence length error.");
        Tensor2D input, output;
        input = concat_input_data[batch_idx][seq_idx].Slice(0,len);
        output = top[0]->data[batch_idx][seq_idx].Slice(0,len);

        Concat(bottom[0]->data[batch_idx][seq_idx].Slice(0,len), 
               bottom[1]->data[batch_idx][seq_idx].Slice(0,len), 
               bottom[2]->data[batch_idx][seq_idx].Slice(0,len), 
               input);

        output = dot(input, this->params[0].data_d2().T());
        if (!no_bias) {
            output += repmat(this->params[1].data_d1(), len);
        }
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    
    for (index_t batch_idx = 0; batch_idx < top[0]->data.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < top[0]->data.size(1); ++seq_idx) {
        int len = bottom[0]->length[batch_idx][seq_idx];
        utils::Assert(len >= 0, "ConvolutionalLstmLayer: sequence length error.");
        Tensor2D in_data, in_diff, out_diff;
        in_data = concat_input_data[batch_idx][seq_idx].Slice(0,len);
        in_diff = concat_input_diff[batch_idx][seq_idx].Slice(0,len);
        out_diff = top[0]->diff[batch_idx][seq_idx].Slice(0,len);

        in_diff = dot(out_diff, this->params[0].data_d2());
        this->params[0].diff_d2() += dot(out_diff.T(), in_data);
        if (!no_bias) {
           this->params[1].diff_d1() += sum_rows(out_diff);
        }

        Split(in_diff, 
              bottom[0]->diff[batch_idx][seq_idx].Slice(0,len),
              bottom[1]->diff[batch_idx][seq_idx].Slice(0,len),
              bottom[2]->diff[batch_idx][seq_idx].Slice(0,len));
      }
    }
  }

 public:
  /*! \brief random number generator */
  int num_input;
  int num_hidden;
  bool no_bias;

  mshadow::TensorContainer<xpu, 4> concat_input_data, concat_input_diff;

};
}  // namespace layer
}  // namespace textnet
#endif 
