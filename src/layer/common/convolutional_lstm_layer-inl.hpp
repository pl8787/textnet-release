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
    this->defaults["pad_value"] = SettingV((float)(NAN));
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
    
    utils::Check(bottom.size() == BottomNodeNum(),
                  "ConvolutionalLstmLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(),
                  "ConvolutionalLstmLayer:top size problem.");
                            
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
    utils::Check(bottom[0]->data.size(1) == 1, "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[1]->data.size(1) == 1, "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[2]->data.size(1) == 1, "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[0]->data.size(2) == bottom[1]->data.size(2), "ConvolutionalLstmLayer:bottom size problem.");
    utils::Check(bottom[0]->data.size(2) == bottom[2]->data.size(2), "ConvolutionalLstmLayer:bottom size problem.");
    
    int batch_size = bottom[0]->data.size(0);
    int length = bottom[0]->data.size(2);
    
    top[0]->Resize(batch_size, 1, length, num_hidden, true);
    concat_input_data->Resize(batch_size, 1, lenght, num_input, true);
    concat_input_data->Resize(batch_size, 1, length, num_input, true);

	bottom[0]->PrintShape("ConvolutionalLstm: bottom_0");
	bottom[1]->PrintShape("ConvolutionalLstm: bottom_1");
	bottom[2]->PrintShape("ConvolutionalLstm: bottom_2");
	top[0]->PrintShape("ConvolutionalLstm: top_0");
  }

  typedef mshadow::Tensor<xpu, 4> Tensor4D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 1> Tensor1D;

  virtual void Concat(Tensor4D l2r_rep, Tensor4D word_rep, Tensor4D r2l_rep, Tensor4D cc_rep) {
    utils::Assert(l2r_rep.size(0) == cc_rep.size(0), "ConvolutionalLstmLayer:size problem.");
    utils::Assert(l2r_rep.size(2) == cc_rep.size(2), "ConvolutionalLstmLayer:size problem.");
    utils::Assert((l2r_rep.size(3)+word_rep.size(3)+r2l_rep.size(3)) == cc_rep.size(3), "ConvolutionalLstmLayer:size problem.");

    int dim_left = l2r_rep.size(3);
    int dim_middle = word_rep.size(3);
    int dim_right = r2l_rep.size(3);

    cc_rep = pad_value; // init
    for (int batch_idx = 0; batch_idx < l2r_rep.size(0); ++batch_idx) {
        int begin = 0; end = 0;
        LocateBeginEnd(l2r_rep, begin, end);
#if DEBUG
        int begin_1 = 0; end_1 = 0;
        LocateBeginEnd(word_rep, begin_1, end_1);
        int begin_2 = 0; end_2 = 0;
        LocateBeginEnd(r2l_rep, begin_2, end_2);
        utils::Assert(begin == begin_1 && begin == begin_2, "");
        utils::Assert(end == end_1 && end == end_2, "");
#endif
        for (int word_idx = begin; word_idx < end; ++word_idx) {
            Tensor1D row_l2r, row_word, row_r2l, row_cc;

            row_cc = cc_rep[batch_idx][0][word_idx].Slice(0, dim_left);
            if (wor_idx == begin) {
                row_cc = 0.f; // left context rep of begining
            } else {
                row_l2r  = l2r_rep[batch_idx][0][word_idx-1];
                row_cc = mshadow::expr::F<op::identity>(row_l2r);
            }

            row_cc = cc_rep[batch_idx][0][word_idx].Slice(dim_left, dim_left+dim_middle);
            row_word = word_rep[batch_idx][0][word_idx];
            row_cc = mshadow::expr::F<op::identity>(row_word);

            row_cc = cc_rep[batch_idx][0][word_idx].Slice(dim_left+dim_middle, dim_left+dim_middle+dim_right);
            if (word_idx == end-1) {
                row_cc = 0.f // right context rep of ending
            } else {
                row_r2l  = r2l_rep[batch_idx][0][word_idx+1];
                row_cc = mshadow::expr::F<op::identity>(row_r2l);
            }
        }
    }
  }
  virtual void Split(Tensor4D cc_rep, Tensor4D l2r_rep, Tensor4D word_rep, Tensor4D r2l_rep) {
    utils::Assert(l2r_rep.size(0) == cc_rep.size(0), "ConvolutionalLstmLayer:size problem.");
    utils::Assert(l2r_rep.size(2) == cc_rep.size(2), "ConvolutionalLstmLayer:size problem.");
    utils::Assert((l2r_rep.size(3)+word_rep.size(3)+r2l_rep.size(3)) == cc_rep.size(3), "ConvolutionalLstmLayer:size problem.");

    int dim_left = l2r_rep.size(3);
    int dim_middle = word_rep.size(3);
    int dim_right = r2l_rep.size(3);

    l2r_rep  = pad_value; // init
    word_rep = pad_value; // init
    r2l_rep  = pad_value; // init

    for (int batch_idx = 0; batch_idx < l2r_rep.size(0); ++batch_idx) {
        int begin = 0; end = 0;
        LocateBeginEnd(cc_rep, begin, end);

        l2r_rep[batch_idx][0][end-1] = 0.f;
        r2l_rep[batch_idx][0][begin] = 0.f;
        for (int word_idx = begin; word_idx < end; ++word_idx) {
            Tensor1D row_l2r, row_word, row_r2l, row_cc;

            row_cc = cc_rep[batch_idx][0][word_idx].Slice(0, dim_left);
            if (wor_idx != begin) {
                row_l2r  = l2r_rep[batch_idx][0][word_idx-1];
                row_l2r = mshadow::expr::F<op::identity>(row_cc);
            }

            row_cc = cc_rep[batch_idx][0][word_idx].Slice(dim_left, dim_left+dim_middle);
            row_word = word_rep[batch_idx][0][word_idx];
            row_word = mshadow::expr::F<op::identity>(row_cc);

            row_cc = cc_rep[batch_idx][0][word_idx].Slice(dim_left+dim_middle, dim_left+dim_middle+dim_right);
            if (word_idx != end-1) {
                row_r2l  = r2l_rep[batch_idx][0][word_idx+1];
                row_r2l = mshadow::expr::F<op::identity>(row_cc);
            }
        }
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;

    concat(bottom[0]->data, bottom[1]->data, bottom[2]->data, concat_input_data);

    for (int batch_idx = 0; batch_idx < concat_input_data.size(0); ++batch_idx) {
        Tensor 2D input, output;
        input = concat_input_data[batch_idx][0];
        output = top[0]->data[batch_idx][0];

        int begin = 0, end = 0;
        LocateBeginEnd(input, begin, end);

        output.Slice(begin, end) = dot(input.Slice(begin, end), this->params[0].T());
        if (!no_bias) {
            output.Slice(begin, end) += repmat(this->params[1].data_d1(), end - begin);
        }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    
    // init
    this->params[0].diff = 0.f;
    this->params[1].diff = 0.f;
    bottom[0]->diff = pad_value;
    bottom[1]->diff = pad_value;
    bottom[2]->diff = pad_value;

    for (int batch_idx = 0; batch_idx < top[0]->data.size(0); ++batch_idx) {
        Tensor2D in_data, in_diff, out_diff;
        int begin = 0, end = 0;
        LocateBeginEnd(top[0]->data[batch_idx][0], begin, end);
        
        in_data = concat_input_data[batch_idx][0]
        in_diff = concat_input_diff[batch_idx][0]
        out_diff = top[0]->diff[batch_idx][0];

        out_diff.Slice(begin, end) = dot(out_diff.Slice(begin, end), this->params[0].data_d2());
        this->params[0].diff_d2() += dot(out_diff.Slice(begin,end).T(), in_data.Slice(begin,end));
        if (!no_bias) {
            this->params[1].diff_d1() += sum_rows(out_diff.Slice(begin,end));
        }
    }
    split(concat_input_diff, bottom[0]->diff, bottom[1]->diff, bottom[2]->diff);
  }

 protected:
  /*! \brief random number generator */
  int num_input;
  int num_hidden;
  bool no_bias;
  float pad_value;

  mshadow::TensorContainer<xpu, 4> concat_input_data, concat_input_diff;

};
}  // namespace layer
}  // namespace textnet
#endif 
