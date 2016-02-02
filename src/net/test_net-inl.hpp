#ifndef TEXTNET_NET_TEST_NET_INL_HPP_
#define TEXTNET_NET_TEST_NET_INL_HPP_

#include <mshadow/tensor.h>
#include "./net.h"

using namespace std;

namespace textnet {
namespace net {

template<typename xpu>
class TestNet : public Net<xpu>{
 public:
  TestNet() { 
	  this->net_type = kTestOnly; 
	  need_activation = false;
  }

  TestNet(int per_file_iter_, int max_iter_, vector<string> node_names_, string file_prefix_, string tag_) {
	  this->net_type = kTestOnly;
	  need_activation = true;
	  per_file_iter = per_file_iter_;
	  max_iter = max_iter_;
	  node_names = node_names_;
	  file_prefix = file_prefix_;
      tag = tag_;
  }

  virtual ~TestNet(void) {}
  
  virtual void Start() {

	utils::Check(this->nets.count(tag),
			"No [%s] tag in config.", tag.c_str());
    if (!need_activation) {
		this->TestAll(tag, 0);
	} else {
		for (int test_iter = 0; test_iter < max_iter; ++test_iter) {
			string file_name = file_prefix + "." + int2str(test_iter);
			this->SaveModelActivation(tag, node_names, per_file_iter, file_name);
		}
	}	

  } 

  bool need_activation;
  int per_file_iter;
  int max_iter;
  vector<string> node_names;
  string file_prefix;
  string tag;
};
}  // namespace net
}  // namespace textnet
#endif  // NET_TEST_NET_INL_HPP_

