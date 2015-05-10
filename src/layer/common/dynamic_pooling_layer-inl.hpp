#ifndef TEXTNET_LAYER_DYNAMIC_POOLING_LAYER_INL_HPP_
#define TEXTNET_LAYER_DYNAMIC_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class DynamicPoolingLayer : public Layer<xpu>{
 public:
  DynamicPoolingLayer(LayerType type) { this->layer_type = type; }
  virtual ~DynamicPoolingLayer(void) {}
  
  virtual int BottomNodeNum() { return 3; } // matrix node, l_seq for length, r_seq for length
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["row"] = SettingV();
    this->defaults["col"] = SettingV();

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    row = setting["row"].iVal();
    col = setting["col"].iVal();
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    utils::Check(bottom.size() == BottomNodeNum(), "DynamicPoolingLayer: bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "DynamicPoolingLayer: top size problem.");

    mshadow::Shape<4> shape_out  = bottom[0]->data.shape_;
    shape_out[2] = row;
    shape_out[3] = col;
    top[0]->Resize(shape_out, true);
    pos_row.Resize(shape_out, true);
    pos_col.Resize(shape_out, true);

    bottom[0]->PrintShape("bottom0");
    bottom[1]->PrintShape("bottom1");
    bottom[2]->PrintShape("bottom2");
    top[0]->PrintShape("top0");
  }

  typedef mshadow::Tensor<xpu,2> Tensor2D;
  typedef mshadow::Tensor<xpu,2,int> Tensor2DInt;

  // select top 1 position from a chunk
  void pooling_one_chunk(Tensor2D t, 
                         int input_row, int input_col, 
                         int begin_row, int end_row, 
                         int begin_col, int end_col, 
                         int &max_row,  int &max_col) {
    max_row = max_col = -1;
    int max_val = -100000000.f;
    for (int row_idx = begin_row; row_idx < end_row; ++row_idx) {
      for (int col_idx = begin_col; col_idx < end_col; ++col_idx) {
        int real_row_idx = row_idx % input_row;
        int real_col_idx = col_idx % input_col;
        if (t[real_row_idx][real_col_idx] > max_val) {
          max_val = t[real_row_idx][real_col_idx];
          max_row = row_idx;
          max_col = col_idx;
        }
      }
    }
  }
  // void duplicate_by_row(Tensor2D &t, int ori_row, int dst_row) {
  //   int max_row = t.size(0);
  //   utils::Check(ori_row < dst_row && dst_row <= max_row, "DynamicPoolingLayer: duplicate error.");
  //   for (int row_idx = ori_row; row_idx < dst_row; ++row_idx) {
  //     int src_row_idx = (row_idx-ori_row) % ori_row;
  //     for (int col_idx = 0; col_idx < t.size(1); ++col_idx) {
  //       t[row_idx][col_idx] = t[src_row_idx][col_idx];
  //     }
  //   }
  // }
  // void duplicate_by_col(Tensor2D &t, int ori_col, int dst_col) {
  //   int max_col = t.size(0);
  //   utils::Check(ori_col < dst_col && dst_col <= max_col, "DynamicPoolingLayer: duplicate error.");
  //   for (int col_idx = ori_col; col_idx < dst_col; ++col_idx) {
  //     int src_col_idx = (col_idx-ori_col) % ori_col;
  //     for (int row_idx = 0; row_idx < t.size(0); ++row_idx) {
  //       t[row_idx][col_idx] = t[row_idx][src_col_idx];
  //     }
  //   }
  // }
  void dynamic_split(int input_row, int pool_row, vector<int> &pos) {
    pos.clear();
    int pad_input_row = input_row < pool_row ? pool_row : input_row;
    int margin = pad_input_row / pool_row;
    int mod    = pad_input_row % pool_row;
    pos.push_back(0);
    for (size_t i = 0; i < pool_row; ++i) {
      if (i < (pool_row-mod)) { 
        pos.push_back(pos[pos.size()-1]+margin);
      } else {
        pos.push_back(pos[pos.size()-1]+margin+1);
      }
    }
    
    utils::Check(pos[pos.size()-1] == pad_input_row, "DynamicPoolingLayer: split error.");
    
    for (size_t i = 1; i < pos.size(); ++i) {
      utils::Check(pos[i-1] < pos[i], "DynamicPoolingLayer: split error.");
      utils::Check((pos[i] - pos[i-1]) <= ((pad_input_row-1)/pool_row) + 1, "DynamicPoolingLayer: split error.");
    }
  }

  void pooling_one_matrix(Tensor2D t_in, Tensor2D t_out,
                          int input_row,  int input_col,
                          int pool_row,   int pool_col,
                          Tensor2DInt row_pos, Tensor2DInt col_pos) {
    utils::Check(t_out.size(0) == pool_row && t_out.size(1) == pool_col, "DynamicPoolingLayer: size error.");
    utils::Check(t_in.size(0) >= input_row && t_in.size(1) >= input_col, "DynamicPoolingLayer: size error.");
    vector<int> begin_pos_row, begin_pos_col;
    dynamic_split(input_row, pool_row, begin_pos_row);
    dynamic_split(input_col, pool_col, begin_pos_col);

    for (int i = 0; i < pool_row; ++i) {
      for (int j = 0; j < pool_col; ++j) {
        int max_row = -1; 
        int max_col = -1;
        pooling_one_chunk(t_in,
                          input_row, input_col,
                          begin_pos_row[i], begin_pos_row[i+1],
                          begin_pos_col[j], begin_pos_col[j+1],
                          max_row, max_col);
        int real_pos_row = max_row % input_row;
        int real_pos_col = max_col % input_col;
        t_out[i][j] = t_in[real_pos_row][real_pos_col];
        row_pos[i][j] = real_pos_row;
        col_pos[i][j] = real_pos_col;
      }
    }
  }

  void unpooling_one_matrix(Tensor2D t_in, Tensor2D t_out,
                            int pool_row,  int pool_col,
                            Tensor2DInt row_pos, Tensor2DInt col_pos) {
    for (int i = 0; i < pool_row; ++i) {
      for (int j = 0; j < pool_col; ++j) {
        int real_pos_row = row_pos[i][j];
        int real_pos_col = col_pos[i][j];
        t_in[real_pos_row][real_pos_col] += t_out[i][j];
      }
    }
  }

  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len_l = bottom[1]->length;
    mshadow::Tensor<xpu, 2> bottom_len_r = bottom[2]->length;
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;

    top_data = 0;
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < bottom_data.size(1); ++seq_idx) {
        pooling_one_matrix(bottom_data[batch_idx][seq_idx], top_data[batch_idx][seq_idx],
                           bottom_len_l[batch_idx][seq_idx], bottom_len_r[batch_idx][seq_idx],
                           row, col,
                           pos_row[batch_idx][seq_idx], pos_col[batch_idx][seq_idx]);
      }
    }
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_diff  = bottom[0]->diff;
    mshadow::Tensor<xpu, 4> top_diff     = top[0]->diff;

    for (index_t batch_idx = 0; batch_idx < bottom_diff.size(0); ++batch_idx) {
      for (index_t seq_idx = 0; seq_idx < bottom_diff.size(1); ++seq_idx) {
        unpooling_one_matrix(bottom_diff[batch_idx][seq_idx], top_diff[batch_idx][seq_idx],
                             row, col,
                             pos_row[batch_idx][seq_idx], pos_col[batch_idx][seq_idx]);
        
      }
    }
  }
 protected:
  mshadow::TensorContainer<xpu, 4, int> pos_row;
  mshadow::TensorContainer<xpu, 4, int> pos_col;
  int row, col;
};
}  // namespace layer
}  // namespace textnet
#endif  // LAYER_WHOLEPOOLING_LAYER_INL_HPP_

