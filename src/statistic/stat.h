#ifndef _STAT_H_
#define _STAT_H_

// #include "../net/net.h" 

namespace textnet {
namespace stat {

using namespace std;
using namespace layer;
using namespace mshadow;
using namespace net;

class TensorStater {
public:
    float ave, max;

    TensorStater(void) {
      clear();
    }

    void clear() {
      ave = 0.f;
      max = -1000000000.f;
    }

    void statTensor(mshadow::Tensor<cpu, 4> &t) {
      int num = t.size(0) * t.size(1) * t.size(2) * t.size(3);
      clear();
      for (int i = 0; i < num; ++i) {
        float x = t.dptr_[i];
        if (x < 0.f) {
          x = -x;
        }

        ave += x;
        if (max < x) {
          max = x;
        }
      }
      ave /= float(num);
    }

    static  TensorStater aggregate(vector<TensorStater> &stats) {
      TensorStater all;
      if (stats.empty()) return all;

      for (int i = 0; i < stats.size(); ++i) {
        all.ave += stats[i].ave;
        all.max += stats[i].max;
      }
      all.ave /= float(stats.size());
      all.max /= float(stats.size());
      return all;
    }
};

template<typename xpu>
class Net<xpu>;

template<typename xpu>
class Stater {
public:
    // using namespace net;
    Stater(net::Net<xpu> *net_, string tag_, PhraseType phrase_, int n_batch_ = 10) {
        net = net_;
        tag = tag_;
        n_batch = n_batch_;
        phrase = phrase_;

        int n_layer = net->nets[tag].size();
        layer_params.resize(n_layer);
        layer_tops.resize(n_layer);
        if (phrase == kTrain) {
          layer_param_diffs.resize(n_layer);
          layer_diffs.resize(n_layer);
        }
    }
    void stat(void) {
        statParams();
        statTopsDiffs();
        print();
    }

private:

    void statParams(void) {
      for (int i = 0; i < net->nets[tag].size(); ++i) {
        if (layer_params[i].empty()) {
          layer_params[i].resize(net->nets[tag][i]->ParamNodeNum());
        }
        for (int j = 0; j < layer_params[i].size(); ++j) {
          layer_params[i][j].statTensor(net->nets[tag][i]->params[j].data);
        }
      }
    }

    vector<Node *> getTopNodes(Layer<xpu> *layer) {
        return net->top_vecs[layer->layer_idx];
    }
    vector<Node *> getBottomNodes(Layer<xpu> *layer) {
        return net->bottom_vecs[layer->layer_idx];
    }

    void statTopsDiffs(void) {
      vector<Layer<xpu> *> layers = net->nets[tag];
      vector<vector<vector<TensorStater> > > all_datas(layers.size());       // layer, top_node, batch
      vector<vector<vector<TensorStater> > > all_data_diffs(layers.size());  // layer, top_node, batch
      vector<vector<vector<TensorStater> > > all_param_diffs(layers.size()); // layer, param, batch


      net->SetPhrase(tag, kTrain);
      for (int batch_idx = 0; batch_idx < n_batch; ++batch_idx) {
        net->Forward(tag);
        if (phrase == kTrain) {
          net->Backprop(tag);
        }

        for (int i = 0; i < layers.size(); ++i) {
          if (all_datas[i].empty()) {
            all_datas[i].resize(layers[i].TopNodeNum());
          }

          if (phrase == kTrain && all_data_diffs[i].empty()) {
            all_data_diffs[i].resize(layers[i].TopNodeNum());
          }

          if (phrase == kTrain && all_param_diffs[i].empty()) {
            all_param_diffs[i].resize(layers[i].ParamNum());
          }

          vector<Node<xpu> *> topNodes = getTopNodes(layers[i]);
          for (int j = 0; j < topNodes.size(); j++) {
            if (all_datas[i][j].empty()) {
                all_datas[i][j].resize(n_batch);
            }
            all_datas[i][j][batch_idx].statTensor(topNodes[j].data);
            if (phrase == kTrain) {
              if (all_data_diffs[i][j].empty()) {
                all_data_diffs[i][j].resize(n_batch);
              }
              all_data_diffs[i][j][batch_idx].statTensor(topNodes[j].diff);
            }
          }
          for (int j = 0; j < layers[i]->ParamNum(); ++j) {
            if (phrase == kTrain) {
              if (all_param_diffs[i][j].empty()) {
                all_param_diffs[i][j].resize(n_batch);
              }
              all_param_diffs[i][j][batch_idx].statTensor(layer[i].params[j].diff);
            }
          }
        }
      }
        
      for (int i = 0; i < layers.size(); ++i) {
        if (layer_datas[i].empty()) {
          layer_datas[i].resize(layers[i]->TopNodeNum());
        }
        for (int j = 0; j < layer_datas[i].size(); ++j) {
          layer_datas[i][j] = TensorStater::aggregate(all_datas[i][j]);
        }
      }

      if (phrase == kTrain) {
        for (int i = 0; i < layers.size(); ++i) {
          if (layer_data_diffs[i].empty()) {
            layer_data_diffs[i].resize(layers[i]->TopNodeNum());
          }
          for (int j = 0; j < layer_data_diffs[i].size(); ++j) {
            layer_data_diffs[i][j] = TensorStater::aggregate(all_data_diffs[i][j]);
          }
        }

        for (int i = 0; i < layers.size(); ++i) {
          if (layer_param_diffs[i].empty()) {
            layer_param_diffs[i].resize(layers[i]->ParamNum());
          }
          for (int j = 0; j < layer_param_diffs[i].size(); ++j) {
            layer_param_diffs[i][j] = TensorStater::aggregate(all_param_diffs[i][j]);
          }
        }
      }
    }
    void print(void) {
      vector<Layer<xpu> *> layers = net->nets[tag];
      for (int i = 0; i < layers.size(); ++i) {
        print("Statistics of layer %d (%s)\n", layers->layer_idx, layers->layer_name);
        for (int j = 0; j < layer_params[i].size(); ++j) {
          print("param_datas[%d]:%12f,%12f\n", j, layer_params[i][j].ave, layer_params[i][j].max);
        }
        for (int j = 0; j < layer_param_diffs[i].size(); ++j) {
          print("param_diffs[%d]:%12f,%12f\n", j, layer_param_diffs[i][j].ave, layer_param_diffs[i][j].max);
        }
        for (int j = 0; j < layer_datas[i].size(); ++j) {
          print("top_datas[%d]:%12f,%12f\n", j, layer_datas[i][j].ave, layer_datas[i][j].max);
        }
        for (int j = 0; j < layer_data_diffs[i].size(); ++j) {
          print("top_diffs[%d]:%12f,%12f\n", j, layer_data_diffs[i][j].ave, layer_data_diffs[i][j].max);
        }
      }
    }

private:
    vector<vector<TensorStater> > layer_params;
    vector<vector<TensorStater> > layer_param_diffs;
    vector<vector<TensorStater> > layer_datas;
    vector<vector<TensorStater> > layer_data_diffs;
    
    int n_batch;
    Net<xpu> *net;
    std::string tag;
    PhraseType phrase;
};
}
}

#endif
