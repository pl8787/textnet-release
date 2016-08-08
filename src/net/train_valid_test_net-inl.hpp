#ifndef TEXTNET_NET_TRAIN_VALID_TEST_NET_INL_HPP_
#define TEXTNET_NET_TRAIN_VALID_TEST_NET_INL_HPP_

#include <mshadow/tensor.h>
#include "./net.h"
#include <ctime>

namespace textnet {
namespace net {

template<typename xpu>
class TrainValidTestNet : public Net<xpu>{
 public:
  TrainValidTestNet() { this->net_type = kTrainValidTest; }
  virtual ~TrainValidTestNet(void) {}
  
  virtual void Start() {

	utils::Check(this->nets.count("Train"), 
			"No [Train] tag in config.");
	utils::Check(this->nets.count("Valid"),
			"No [Valid] tag in config.");
	utils::Check(this->nets.count("Test"),
			"No [Test] tag in config.");
    

    time_t begin = 0, end = 0;
    time(&begin);
	for (int iter = 0; iter < this->max_iters["Train"]; ++iter) {
		if (iter != 0 || (iter == 0 && this->model_save_initial)) {
            this->SaveModel(iter);

            this->SaveModelActivation(iter);
		}
		if (iter != 0 || (iter == 0 && this->model_test_initial)) {
		    if (this->display_interval["Valid"] > 0 && iter % this->display_interval["Valid"] == 0) {
                time(&end);
                cout << "valid display interval: " << end-begin << "s." << endl;
			    this->TestAll("Valid", iter);
                time(&begin);
		    }	

		    if (this->display_interval["Test"] > 0 && iter % this->display_interval["Test"] == 0) {
			    this->TestAll("Test", iter);
	            utils::ShowMemoryUse();
#if TIME_DEBUG
                this->PrintClock("Train");
#endif
	    	}	
		}
		this->TrainOneStep("Train");

		if (this->display_interval["Train"] > 0 && iter % this->display_interval["Train"] == 0) {
			this->TrainDisplay("Train", iter);
		}
	}

	this->SaveModel(this->max_iters["Train"], this->model_save_last);
  } 
};
}  // namespace net
}  // namespace textnet
#endif  // NET_TRAIN_VALID_TEST_NET_INL_HPP_

