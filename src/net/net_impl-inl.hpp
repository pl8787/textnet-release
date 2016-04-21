#ifndef TEXTNET_NET_IMPL_INL_HPP_
#define TEXTNET_NET_IMPL_INL_HPP_
#pragma once

#include "./net.h"
#include "./train_valid_net-inl.hpp"
#include "./train_valid_test_net-inl.hpp"
#include "./test_net-inl.hpp"
#include "./multi_train_valid_test_net-inl.hpp"

namespace textnet {
namespace net {
template<typename xpu>
INet* CreateNet_(NetType type) {
  switch(type) {
    case kTrainValid: return new TrainValidNet<xpu>();
    case kTrainValidTest: return new TrainValidTestNet<xpu>();
	// case kCrossValid: return new CrossValidNet<xpu>();
	case kTestOnly: return new TestNet<xpu>();
    case kMultiTrainValidTest: return new MultiTrainValidTestNet<xpu>();
    default: utils::Error("unknown net type id : \"%d\"", type); return NULL;
  }
}

}  // namespace net
}  // namespace textnet
#endif
