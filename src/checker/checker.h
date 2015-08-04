#ifndef TEXTNET_CHECKER_CHECKER_H_
#define TEXTNET_CHECKER_CHECKER_H_

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <mshadow/tensor.h>
#include "../global.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../layer/layer.h"

/*! \brief namespace of textnet */
namespace textnet {
/*! \brief namespace of layer defintiion */
namespace checker {

// Add a L2 norm loss on the top

template<typename xpu>
class Checker {
 public:
  Checker(void) {}
  ~Checker(void) {}
  
  struct error {
    MSHADOW_XINLINE static real_t Map(real_t x, real_t minv, real_t maxv) {
      if (x < minv) return x;
      if (x > maxv) return x;
      return 0.0f;
    }
  };  
  
  void PrintTensor(const char * name, mshadow::Tensor<cpu, 4> x) {
	using namespace std;
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
    cout << endl;
  }
  
  void SetupChecker(std::map<std::string, SettingV> &setting, mshadow::Random<cpu>* prnd) {
    prnd_ = prnd;
    range_min = setting["range_min"].fVal();
    range_max = setting["range_max"].fVal();
    delta = setting["delta"].fVal();
  }
  
  float GetLoss(std::vector<layer::Node<xpu>*> tops) {
    float loss = 0.0;
    for (int i = 0; i < tops.size(); ++i) {
      tops[i]->data *= tops[i]->data;
      for (int j = 0; j < tops[i]->data.shape_.Size(); ++j) {
        loss += tops[i]->data.dptr_[j];
      }
    }
    return loss/2.0f;
  }
  
  bool CheckError(layer::Layer<xpu>* layer, 
                  std::vector<layer::Node<xpu>*> bottoms,
                  std::vector<layer::Node<xpu>*> tops) {
    using namespace mshadow::expr;
    using namespace std;

	utils::Check(layer->BottomNodeNum() > 0, 
			"CheckError: This layer does not has input, so remove this check.");
	utils::Check(layer->TopNodeNum() > 0, 
			"CheckError: This layer does not has output, so remove this check.");

    std::vector<mshadow::TensorContainer<xpu, 4> > calculate_error;
    std::vector<mshadow::TensorContainer<xpu, 4> > estimate_error;
    
    // Fill the top diff
    // Because the loss is L2 norm, the top diff is simply copy of top data
    layer->Forward(bottoms, tops);
    layer->ClearDiff(bottoms, tops);
    for (int i = 0; i < bottoms.size(); ++i) {
      bottoms[i]->diff = 0.;
    }
    for (int i = 0; i < tops.size(); ++i) {
      tops[i]->diff = F<mshadow::op::identity>(tops[i]->data);
    }
    
    // Calculate by Backprop
    layer->Backprop(bottoms, tops);
    for (int i = 0; i < bottoms.size(); ++i) {
      calculate_error.push_back(mshadow::TensorContainer<xpu, 4>(bottoms[i]->diff.shape_));
      calculate_error[i] = F<mshadow::op::identity>(bottoms[i]->diff);
      
      estimate_error.push_back(mshadow::TensorContainer<xpu, 4>(bottoms[i]->diff.shape_, 0.0));
    }
    
    // Estimate by delta
    for (int i = 0; i < bottoms.size(); ++i) {
	  if (!layer->prop_error[i]) continue;
      for (int j = 0; j < bottoms[i]->data.shape_.Size(); ++j) {
        float positive_act = 0.0f;
        float negative_act = 0.0f;

        // Add delta to bottoms[i][j]
        bottoms[i]->data.dptr_[j] += delta;
        layer->Forward(bottoms, tops);
        positive_act = GetLoss(tops);
        
        // Minus delta to bottoms[i][j]
        bottoms[i]->data.dptr_[j] -= 2*delta;
        layer->Forward(bottoms, tops);
        negative_act = GetLoss(tops);
        
        estimate_error[i].dptr_[j] = (positive_act - negative_act) / (2*delta);
		
		bottoms[i]->data.dptr_[j] += delta;
      }
    }
    
	float scale = 0.0f;

    for (int i = 0; i < bottoms.size(); ++i) {
      for (int j = 0; j < calculate_error[i].shape_.Size(); ++j) {
		scale = max( max(fabs(calculate_error[i].dptr_[j]), fabs(estimate_error[i].dptr_[j])), 1.0f);
        if (calculate_error[i].dptr_[j] - estimate_error[i].dptr_[j] < range_min*scale ||
            calculate_error[i].dptr_[j] - estimate_error[i].dptr_[j] > range_max*scale  ) {
          cout << "E [" << i << ", " << j << "] = " << calculate_error[i].dptr_[j] << " | " << estimate_error[i].dptr_[j] << endl;
        }
        cout << "I [" << i << ", " << j << "] = " << calculate_error[i].dptr_[j] << " | " << estimate_error[i].dptr_[j] << endl;
      }
    }

    return true;
  }
  
  bool CheckGrad(layer::Layer<xpu>* layer, 
                 std::vector<layer::Node<xpu>*> bottoms, 
                 std::vector<layer::Node<xpu>*> tops) {
    using namespace mshadow::expr;
    using namespace std;
    std::vector<mshadow::TensorContainer<xpu, 4> > calculate_grad;
    std::vector<mshadow::TensorContainer<xpu, 4> > estimate_grad;

	utils::Check(layer->ParamNodeNum() > 0, 
			"CheckGrad: This layer does not has parameters, so remove this check.");
    
    std::vector<layer::Node<xpu> > &params = layer->GetParams();
    
    // Fill the top diff
    // Because the loss is L2 norm, the top diff is simply copy of top data
    layer->Forward(bottoms, tops);
    layer->ClearDiff(bottoms, tops);
    for (int i = 0; i < bottoms.size(); ++i) {
      bottoms[i]->diff = 0.;
    }
    for (int i = 0; i < tops.size(); ++i) {
      tops[i]->diff = F<mshadow::op::identity>(tops[i]->data);
    }
    
    // Calculate by Backprop
    layer->Backprop(bottoms, tops);
    
    for (int i = 0; i < params.size(); ++i) {
      calculate_grad.push_back(mshadow::TensorContainer<xpu, 4>(params[i].diff.shape_));
      calculate_grad[i] = F<mshadow::op::identity>(params[i].diff);
      
      estimate_grad.push_back(mshadow::TensorContainer<xpu, 4>(params[i].diff.shape_, 0.0));
    }
    
    // Estimate by delta
    for (int i = 0; i < params.size(); ++i) {
	  if (!layer->prop_grad[i]) continue;
      for (int j = 0; j < params[i].data.shape_.Size(); ++j) {
        float positive_act = 0.0f;
        float negative_act = 0.0f;

        // Add delta to bottoms[i][j]
        params[i].data.dptr_[j] += delta;
        layer->Forward(bottoms, tops);
        positive_act = GetLoss(tops);
        
        // Minus delta to bottoms[i][j]
        params[i].data.dptr_[j] -= 2*delta;
        layer->Forward(bottoms, tops);
        negative_act = GetLoss(tops);
        
        estimate_grad[i].dptr_[j] = (positive_act - negative_act) / (2*delta);

		params[i].data.dptr_[j] += delta;
      }
    }
    
	float scale = 0.0f;
    for (int i = 0; i < bottoms.size(); ++i) {
      for (int j = 0; j < calculate_grad[i].shape_.Size(); ++j) {
		scale = max( max(fabs(calculate_grad[i].dptr_[j]), fabs(estimate_grad[i].dptr_[j])), 1.0f);
        if (calculate_grad[i].dptr_[j] - estimate_grad[i].dptr_[j] < range_min*scale ||
            calculate_grad[i].dptr_[j] - estimate_grad[i].dptr_[j] > range_max*scale  ) {
          cout << "E [" << i << ", " << j << "] = " << calculate_grad[i].dptr_[j] << " | " << estimate_grad[i].dptr_[j] << endl;
        }
        cout << "I [" << i << ", " << j << "] = " << calculate_grad[i].dptr_[j] << " | " << estimate_grad[i].dptr_[j] << endl;
      }
    }

    return true;
  }
  
 protected:
  mshadow::Random<xpu>* prnd_;
  float range_min;
  float range_max;
  float delta;  
};

template<typename xpu>
Checker<xpu>* CreateChecker();

}  // namespace checker
}  // namespace textnet
#endif  // TEXTNET_CHECKER_CHECKER_H_
