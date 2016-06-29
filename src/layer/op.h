#ifndef TEXTNET_LAYER_OP_H_
#define TEXTNET_LAYER_OP_H_
#pragma once

#include <mshadow/tensor.h>
#include <iostream>

extern float SIGMOID_MAX_INPUT;
extern int SIGMOID_TABLE_SIZE;
extern float *p_sigmoid_lookup_table;
extern float TANH_MAX_INPUT;
extern int TANH_TABLE_SIZE;
extern float *p_tanh_lookup_table;
extern float EXP_MAX_INPUT;
extern int EXP_TABLE_SIZE;
extern float *p_exp_lookup_table;

namespace textnet {
/*! \brief operations for ActivationLayer */
namespace op {
/*! \brief Linear Operation */
struct identity {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a;
  }
};

struct identity_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f;
  }
};

struct orc_exp {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return std::exp(a);
  }
};

/*! \brief sigmoid unit */
struct sigmoid_lookup {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    if (a >= SIGMOID_MAX_INPUT) {
      return 1.f;
    } else if (a <= -SIGMOID_MAX_INPUT) {
      return 0.f;
    }
    
    int pos = ((a+SIGMOID_MAX_INPUT)/(2*SIGMOID_MAX_INPUT))*SIGMOID_TABLE_SIZE;
    if (pos >= SIGMOID_TABLE_SIZE) {
      pos = SIGMOID_TABLE_SIZE-1;
    }
    if (pos < 0) {
      pos = 0; 
    }
    return p_sigmoid_lookup_table[pos];
  }
};

/*! \brief sigmoid unit */
struct tanh_lookup {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    if (a >= TANH_MAX_INPUT) {
      return 1.f;
    } else if (a <= -TANH_MAX_INPUT) {
      return -1.f;
    }
    
    // NOTICE: float accuracy may cause problem here
    int pos = ((a+TANH_MAX_INPUT)/(2*TANH_MAX_INPUT))*TANH_TABLE_SIZE;
    if (pos >= TANH_TABLE_SIZE) {
      pos = TANH_TABLE_SIZE-1;
    }
    if (pos < 0) {
      pos = 0; 
    }
    return p_tanh_lookup_table[pos];
  }
};

/*! \brief exp unit */
struct exp_lookup {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    if (a >= EXP_MAX_INPUT) {
      return 1e8f;
    } else if (a <= -EXP_MAX_INPUT) {
      return 1e-9f;
    }
    
    // NOTICE: float accuracy may cause problem here
    int pos = ((a+EXP_MAX_INPUT)/(2*EXP_MAX_INPUT))*EXP_TABLE_SIZE;
    if (pos >= EXP_TABLE_SIZE) {
      pos = EXP_TABLE_SIZE-1;
    }
    if (pos < 0) {
      pos = 0; 
    }
    return p_exp_lookup_table[pos];
  }
};

/*! \brief sigmoid unit */
struct sigmoid {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};
struct sigmoid_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * (1.0f - a);
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    using namespace std;
    return max(a, 0.0f);
  }
};
struct relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : 0.0f;
  }
};

/*! \brief Leaky ReLU Operation */
struct xelu {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a > 0 ? a : a / b;
  }
};

struct xelu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a > 0 ? 1 : 1.0f / b;
  }
};

struct tanh {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return tanhf( a );
  }
};

struct tanh_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f - a * a;
  }
};


struct square {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * a;
  }
};

// orc
struct pow_3{
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * a * a;
  }
};



/*! \brief used for generate Bernoulli mask */
struct threshold {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return a < b ? 1.0f : 0.0f;
  }
};

/*! \brief used for generate element of power */
struct power {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    return powf( a, b );
  }
};

/*!\ \brief used for generate element sqrt */
struct square_root {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return sqrt(a);
  }
};

}  // namespace op
}  // namespace textnet
#endif // TEXTNET_LAYER_OP_H
