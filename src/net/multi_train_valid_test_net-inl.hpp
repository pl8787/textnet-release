#ifndef TEXTNET_NET_MULTI_TRAIN_VALID_TEST_NET_INL_HPP_
#define TEXTNET_NET_MULTI_TRAIN_VALID_TEST_NET_INL_HPP_

#include <mshadow/tensor.h>
#include "./net.h"
#include <ctime>

namespace textnet {
namespace net {

template<typename xpu>
class MultiTrainValidTestNet : public Net<xpu>{
 public:
  MultiTrainValidTestNet() { this->net_type = kMultiTrainValidTest; }
  virtual ~MultiTrainValidTestNet(void) {}
  
  vector<string> train_tags;
  vector<string> valid_tags;
  vector<string> test_tags;

  virtual void Start() {

    for (auto it = this->nets.begin(); it != this->nets.end(); ++it) {
        string tag = it->first;
        if (tag.find("Train") == 0) {
            train_tags.push_back(tag);
        } else if (tag.find("Valid") == 0) {
            valid_tags.push_back(tag);
        } else if (tag.find("Test") == 0) {
            test_tags.push_back(tag);
        } else {
            utils::Check(false, "Tag name invalid: %s.", tag.c_str());
        }
    }

    utils::Printf("Train tags: \n");
    for (int i = 0; i < (int)train_tags.size(); ++i) {
        utils::Printf("[%s]\n", train_tags[i].c_str());
    }
    utils::Printf("Valid tags: \n");
    for (int i = 0; i < (int)valid_tags.size(); ++i) {
        utils::Printf("[%s]\n", valid_tags[i].c_str());
    }
    utils::Printf("Test tags: \n");
    for (int i = 0; i < (int)test_tags.size(); ++i) {
        utils::Printf("[%s]\n", test_tags[i].c_str());
    }

	utils::Check(train_tags.size() > 0, 
			"No [Train] tag in config.");
	utils::Check(valid_tags.size() > 0,
			"No [Valid] tag in config.");
	utils::Check(test_tags.size() > 0,
			"No [Test] tag in config.");
    
	for (int iter = 0; iter < this->max_iters[train_tags[0]]; ++iter) {
		if (iter != 0 || (iter == 0 && this->model_save_initial)) {
            this->SaveModel(iter);

            this->SaveModelActivation(iter);
		}

		if (iter != 0 || (iter == 0 && this->model_test_initial)) {
            for (int i = 0; i < (int)valid_tags.size(); ++i) {
                string tag = valid_tags[i];
		        if (this->display_interval[tag] > 0 && iter % this->display_interval[tag] == 0) {
			        this->TestAll(tag, iter);
		        }
            }    

            for (int i = 0; i < (int)test_tags.size(); ++i) {
                string tag = test_tags[i];
		        if (this->display_interval[tag] > 0 && iter % this->display_interval[tag] == 0) {
			        this->TestAll(tag, iter);
	    	    }	
            }

	        utils::ShowMemoryUse();
		}

        for (int i = 0; i < (int)train_tags.size(); ++i) {
            string tag = train_tags[i];
		    this->TrainOneStep(tag);
		    if (this->display_interval[tag] > 0 && iter % this->display_interval[tag] == 0) {
			    this->TrainDisplay(tag, iter);
		    }
        }
	}

	this->SaveModel(this->max_iters[train_tags[0]], this->model_save_last);
  } 
};
}  // namespace net
}  // namespace textnet
#endif  // NET_MULTI_TRAIN_VALID_TEST_NET_INL_HPP_

