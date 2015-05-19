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

  TestNet(int batch_size_, int max_iter_, vector<string> node_names_, string file_prefix_) {
	  this->net_type = kTestOnly;
	  need_activation = true;
	  batch_size = batch_size_;
	  max_iter = max_iter_;
	  node_names = node_names_;
	  file_prefix = file_prefix_;
  }

  virtual ~TestNet(void) {}
  
  virtual void Start() {

	utils::Check(this->nets.count("Test"),
			"No [Test] tag in config.");
    if (!need_activation) {
		this->TestAll("Test", 0);
	} else {
		for (int test_iter = 0; test_iter < max_iter; ++test_iter) {
			string file_name = file_prefix + "." + int2str(test_iter);
			this->SaveModelActivation("Test", node_names, batch_size, file_name);
		}
	}	

  } 

  bool need_activation;
  int batch_size;
  int max_iter;
  vector<string> node_names;
  string file_prefix;
};
}  // namespace net
}  // namespace textnet
#endif  // NET_TEST_NET_INL_HPP_

