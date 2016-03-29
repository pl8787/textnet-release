#ifndef TEXTNET_LAYER_WHOLE_POOLING_2D_LAYER_INL_HPP_
#define TEXTNET_LAYER_WHOLE_POOLING_2D_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "../layer.h"
#include "../op.h"

namespace textnet {
namespace layer {

template<typename xpu>
class WholePooling2dLayer : public Layer<xpu>{
 public:
  WholePooling2dLayer(LayerType type) { this->layer_type = type; }
  virtual ~WholePooling2dLayer(void) {}
  
  virtual int BottomNodeNum() { return 1; }
  virtual int TopNodeNum() { return 1; }
  virtual int ParamNodeNum() { return 0; }
  
  virtual void Require() {
    // default value, just set the value you want

    // require value, set to SettingV(),
    // it will force custom to set in config
    this->defaults["pool_type"] = SettingV(); // last, ave, sum, first

    Layer<xpu>::Require();
  }
  
  virtual void SetupLayer(std::map<std::string, SettingV> &setting,
                          const std::vector<Node<xpu>*> &bottom,
                          const std::vector<Node<xpu>*> &top,
                          mshadow::Random<xpu> *prnd) {
    Layer<xpu>::SetupLayer(setting, bottom, top, prnd);
    pool_type = setting["pool_type"].sVal();
  }
  
  virtual void Reshape(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top,
					   bool show_info = false) {
    utils::Check(bottom.size() == BottomNodeNum(), "WholePooling2dLayer:bottom size problem."); 
    utils::Check(top.size() == TopNodeNum(), "WholePooling2dLayer:top size problem.");

    mshadow::Shape<4> shape_in = bottom[0]->data.shape_;
    mshadow::Shape<4> shape_out = mshadow::Shape4(shape_in[0], 1, 1, shape_in[3]);
    top[0]->Resize(shape_out, true);
    pos.Resize(shape_out, -1);

	if (show_info) {
	  bottom[0]->PrintShape("bottom0");
	  top[0]->PrintShape("top0");
	}
  }

  typedef mshadow::Tensor<xpu, 1> Tensor1D;
  typedef mshadow::Tensor<xpu, 2> Tensor2D;
  typedef mshadow::Tensor<xpu, 3> Tensor3D;
  typedef mshadow::Tensor<xpu, 2, int> Tensor2DInt;
  typedef mshadow::Tensor<xpu, 3, int> Tensor3DInt;
  typedef mshadow::Tensor<xpu, 4> Tensor4D;

  // in (x, y, d), out (1, d)
  void wholeAvePooling(Tensor3D in, Tensor2D out) {
    Tensor2D in_d2 = Tensor2D(in.dptr_, mshadow::Shape2(in.size(0)*in.size(1), in.size(2)));
    utils::Check(in_d2.size(1) == out.size(1) && out.size(0) == 1, "WholePooling2dLayer:pooling io size error");
    out[0] = sum_rows(in_d2);
    out /= float(in_d2.size(0));
  }
  // in (1, d), out (x, y, d)
  void wholeUnAvePooling(Tensor2D in, Tensor3D out) {
    Tensor2D out_d2 = Tensor2D(out.dptr_, mshadow::Shape2(out.size(0)*out.size(1), out.size(2)));
    utils::Check(in.size(1) == out_d2.size(1) && in.size(0) == 1, "WholePooling2dLayer:pooling io size error");
    out_d2 += repmat(in[0] / (float)(out_d2.size(0)), out_d2.size(0));
  }
  // in (x, y, d), out (1, d)
  void wholeSumPooling(Tensor3D in, Tensor2D out) {
    Tensor2D in_d2 = Tensor2D(in.dptr_, mshadow::Shape2(in.size(0)*in.size(1), in.size(2)));
    utils::Check(in_d2.size(1) == out.size(1) && out.size(0) == 1, "WholePooling2dLayer:pooling io size error");
    out[0] = sum_rows(in_d2);
  }
  // in (1, d), out (x, y, d)
  void wholeUnSumPooling(Tensor2D in, Tensor3D out) {
    Tensor2D out_d2 = Tensor2D(out.dptr_, mshadow::Shape2(out.size(0)*out.size(1), out.size(2)));
    utils::Check(in.size(1) == out_d2.size(1) && in.size(0) == 1, "WholePooling2dLayer:pooling io size error");
    out_d2 += repmat(in[0], out_d2.size(0));
  }

  // in (x, y, d), out (1, d)
  void wholeFirstPooling(Tensor3D in, Tensor2D out) {
    utils::Check(in.size(2) == out.size(1) && out.size(0) == 1, "WholePooling2dLayer:pooling io size error");
    out[0] = mshadow::expr::F<op::identity>(in[0][0]);
  }
  // in (1, d), out (x, y, d)
  void wholeUnFirstPooling(Tensor2D in, Tensor3D out) {
    utils::Check(in.size(1) == out.size(2) && in.size(0) == 1, "WholePooling2dLayer:pooling io size error");
    out[0][0] += in[0];
  }
  // in (x, y, d), out (1, d)
  void wholeLastPooling(Tensor3D in, Tensor2D out) {
    utils::Check(in.size(2) == out.size(1) && out.size(0) == 1, "WholePooling2dLayer:pooling io size error");
    out[0] = mshadow::expr::F<op::identity>(in[in.size(0)-1][in.size(1)-1]);
  }
  // in (1, d), out (x, y, d)
  void wholeUnLastPooling(Tensor2D in, Tensor3D out) {
    utils::Check(in.size(1) == out.size(2) && in.size(0) == 1, "WholePooling2dLayer:pooling io size error");
    out[out.size(0)-1][out.size(1)-1] += in[0];
  }

  // to do
  // void wholeMaxPooling(Tensor2D in, Tensor2DInt pos, Tensor2D out) {
  //   utils::Check(in.size(1) == out.size(1) && out.size(0) == 1, "WholePooling2dLayer:pooling io size error");
  //   utils::Check(in.size(1) == pos.size(1) && pos.size(0) == 1, "WholePooling2dLayer:pooling io size error");
  //   out = -1000000;
  //   for (index_t col = 0; col < in.size(1); ++col) {
  //     for (index_t row = 0; row < in.size(0); ++row) {
  //       if (in[row][col] > out[0][col]) {
  //           out[0][col] = in[row][col];
  //           pos[0][col] = (int)(row);
  //       }
  //     }
  //   }
  // }
  
  // to do
  // void wholeUnMaxPooling(Tensor2D in, Tensor2DInt pos, Tensor2D out) {
  //   utils::Check(in.size(1) == out.size(1) && in.size(0) == 1, "WholePooling2dLayer:pooling io size error");
  //   utils::Check(in.size(1) == pos.size(1) && pos.size(0) == 1, "WholePooling2dLayer:pooling io size error");
  //   for (index_t col = 0; col < in.size(1); ++col) {
  //     int row = pos[0][col];
  //     out[row][col] += in[0][col];
  //   }
  // }

  void checkNan(float *p, int l) {
    for (int i = 0; i < l; ++i) {
      assert(!isnan(p[i]));
    }
  }

  // note: here we use += other than =
  void sub_tensor_add_equal(Tensor3D in, int x_len, int y_len, Tensor3D out) {
    // utils::Check(in.size(0) >= x_len && in.size(1) >= y_len, "WholePooling2dLayer: length error");
    // utils::Check(out.size(0) == x_len && out.size(1) == y_len, "WholePooling2dLayer: length error");
    for (int i = 0; i < x_len; ++i) {
      for (int j = 0; j < y_len; ++j) {
        // out[i][j] += mshadow::expr::F<op::identity>(in[i][j]);
        out[i][j] += in[i][j];
      }
    }
  }
  // void backward_copy(Tensor3D in, int x_len, int y_len, Tensor3D out) {
  //   utils::Check(out.size(0) >= x_len && out.size(1) >= y_len, "WholePooling2dLayer: length error");
  //   utils::Check(in.size(0) == x_len && in.size(1) == y_len, "WholePooling2dLayer: length error");
  //   for (int i = 0; i < x_len; ++i) {
  //     for (int j = 0; j < y_len; ++j) {
  //       out[i][j] = mshadow::expr::F<op::identity>(in[i][j]);
  //     }
  //   }
  // }

  // bottom_data: (batch_size, x_len, y_len, d_hidden)
  virtual void Forward(const std::vector<Node<xpu>*> &bottom,
                       const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    utils::Check(bottom_len.size(0) == bottom_data.size(0), "WholePooling2dLayer: sequence length error.");
    utils::Check(bottom_len.size(1) == 2, "WholePooling2dLayer: sequence length error.");
    mshadow::Tensor<xpu, 4> top_data = top[0]->data;
	mshadow::Tensor<xpu, 2> top_len  = top[0]->length;

    // conv var len to static len, no need to forward length info
    
    top_data = 0;
    top_len = -1;

    int d_hidden = bottom_data.size(3);
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
	  // top_len[batch_idx][seq_idx] = 1; // not forward length for a fixed length layer
      int x_len = bottom_len[batch_idx][0];
      int y_len = bottom_len[batch_idx][1];
      utils::Check(x_len > 0, "WholePooling2dLayer: sequence length error.");
      utils::Check(y_len > 0, "WholePooling2dLayer: sequence length error.");

      mshadow::TensorContainer<xpu, 3, float> sub_tensor(mshadow::Shape3(x_len, y_len, d_hidden));
      sub_tensor = 0.f;
      sub_tensor_add_equal(bottom_data[batch_idx], x_len, y_len, sub_tensor);
      if (pool_type == "ave") {
        // if (begin == end) continue;
        wholeAvePooling(sub_tensor, top_data[batch_idx][0]);
      } else if (pool_type == "sum") {
        // if (begin == end) continue;
        wholeSumPooling(sub_tensor, top_data[batch_idx][0]);
      } else if (pool_type == "first") {
        // if (begin == end) continue;
        wholeFirstPooling(sub_tensor, top_data[batch_idx][0]);
      } else if (pool_type == "last") {
        // if (begin == end) continue;
        wholeLastPooling(sub_tensor, top_data[batch_idx][0]);
      } else {
        utils::Check(false, "WholePoolLayer: pool type error.");
      }
    }
    // checkNan(top_data.dptr_, top_data.size(0)*top_data.size(1)*top_data.size(2)*top_data.size(3));
  }
  
  virtual void Backprop(const std::vector<Node<xpu>*> &bottom,
                        const std::vector<Node<xpu>*> &top) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> top_diff = top[0]->diff;
    mshadow::Tensor<xpu, 4> bottom_data = bottom[0]->data;
    mshadow::Tensor<xpu, 2> bottom_len  = bottom[0]->length;
    mshadow::Tensor<xpu, 4> bottom_diff = bottom[0]->diff;

    int d_hidden = bottom_data.size(3);
    for (index_t batch_idx = 0; batch_idx < bottom_data.size(0); ++batch_idx) {
      // int begin = 0, end = bottom_len[batch_idx][seq_idx]; 
      int x_len = bottom_len[batch_idx][0];
      int y_len = bottom_len[batch_idx][1];
      utils::Check(x_len > 0, "WholePooling2dLayer: sequence length error.");
      utils::Check(y_len > 0, "WholePooling2dLayer: sequence length error.");

      mshadow::TensorContainer<xpu, 3, float> sub_tensor(mshadow::Shape3(x_len, y_len, d_hidden));
      sub_tensor = 0.f;
      if (this->prop_error[0]) {
        if (pool_type == "ave") {
          // if (begin == end) continue;
          wholeUnAvePooling(top_diff[batch_idx][0], sub_tensor);
        } else if (pool_type == "sum") {
          // if (begin == end) continue;
          wholeUnSumPooling(top_diff[batch_idx][0], sub_tensor);
        } else if (pool_type == "first") {
          // if (begin == end) continue;
          wholeUnFirstPooling(top_diff[batch_idx][0], sub_tensor);
        } else if (pool_type == "last") {
          // if (begin == end) continue;
          wholeUnLastPooling(top_diff[batch_idx][0], sub_tensor);
        } else {
          utils::Check(false, "WholePoolLayer: pool type error.");
        }
      }
      sub_tensor_add_equal(sub_tensor, x_len, y_len, bottom_diff[batch_idx]);
    }
  }
 protected:
  mshadow::TensorContainer<xpu, 4, int> pos;
  std::string pool_type;
};
}  // namespace layer
}  // namespace textnet
#endif

