Model Configuration File Description (JSON)
====
We use JSON as a protocol of our model. We describe each sections meaning below.

Net Name
====
- net_name : set the name of the net.
- log: the log file for redirecting all screen outputs, default unsetted.
- cross_validation : the # of folds if cross validation is used, default unsetted. If setted, all data file for all folds must be splited advanced and named as ($FILE_PREFIX).($i), where FILE_PREFIX is setted in input layers and i is the index of the fold.

```json
"net_name" : "simple_net"
"log" : "log.simple_net"
"cross_validation" : 10
```

Net Configuration Section
====
- tag : the name of tag
- max_iters : the maximum iterations count
- display_interval : the interval of display this net result
- save_interval : the interval of saving this net model
- save_name : the path and prefix name of the saving model file
- out_nodes : the output node name list

Set all tagged networks. Here we set ```Train``` and ```Valid``` networks.

```json
"net_config" : [
  {
    "tag" : "Train",
    "max_iters" : 10000,
    "display_interval": 1,
    "save_interval": 1000,
    "save_name": "weights/matching_net_arc2",
    "out_nodes" : ["loss"]
  },
  {
    "tag" : "Valid",
    "max_iters" : 34,
    "display_interval" : 100,
    "save_interval" : 0,
    "save_name" : "",
    "out_nodes" : ["loss", "acc"]
  }
]
```

Global Section
====
In this section, we define some repeat parameters using in each layers, namely **place holder**. 

For example:

```json
"global" : {
  "w_updater_ph" : {
    "w_updater" : {
       "decay" : 0.01,
       "lr" : 0.01,
       "updater_type" : "SGD"
    }
  }
}
```

Here ```w_updater_ph``` is an arbitrary name of a **place holder**. When it occurs in any layer, we will replace it with:
```json
"w_updater" : {
   "decay" : 0.01,
   "lr" : 0.01,
   "updater_type" : "SGD"
}
```

For example, in layer *Embedding* we write:
```json
{
   "bottom_nodes" : [ "data" ],
   "layer_idx" : 1,
   "layer_name" : "embedding",
   "layer_type" : 21,
   "setting" : {
      "embedding_file" : "wikicorp_50_msr.txt",
      "feat_size" : 50,
      "w_filler" : {
         "init_type" : 2,
         "range" : 0.01
      },
->    "w_updater_ph" : 0,    <-
      "word_count" : 14727
   },
   "top_nodes" : [ "embed" ]
}
```

Model Save Section
====
In this section, we configure how to save intermediate models and node activations.

- save_model: configure how to save model parameters
  - save_interval: the interval of batches for saving a model
  -	file_prefix: the prefix of the model file which will be subfixed by the iter id
- save_activation: config how to save node activations, this is a list value for saving different tags
  - tag: the tag of the net for saving
  - save_interval: the interval of batches for saving activations
  -	file_prefix: the prefix of the model file which will be subfixed by the iter id
  - save_iter_num: the # of batches for saving
  - save_nodes: the node names for saving, default all nodes of the net.
  

Layers Section
====
In this section, we list all layers we use as a list. 

- layer_name : the name of this layer
- layer_idx : the index of this layer, set to any number you want
- layer_type : the type of this layer, string or int
- bottom_nodes : a list of bottom node name
- top_nodes : a list of top node name
- tag : the tag name list, identify which net contain this layer. If null, all net use this layer.
- tag_mode : two type **share** or **new**
  - share : share this layer with all net
  - new : create a new layer for each net
- setting : a map specified by different layers

### Parameter Sharing
To cope with sharing parameters between layers, we can use these configuration below:
- share : a list of parameters for share.
  - param_id : the parameter id in current layer
  - source_layer_name : the source layer name
  - source_param_id : the source parameter id in source layer
      
for example:
```json
"share" : [
  {
    "param_id" : 0,
    "source_layer_name" : "conv11",
    "source_param_id" : 0
  }
]
```

### Parameter Values
To cope with the protocol with have parameter values, we use these symbols below:
- param : the parameter section
  - shape : the shape of the parameter
  - value : the value of the parameter
  
for example:
```json
"param" : [
  {
     "shape" : [ 14727, 50, 1, 1 ],
     "value" : [
        -0.3041162,
        0.4885388,
        -0.1863326,
        ...
        0.789396,
        1.03154
     ]
  }
]
```

Simple Network Example
====
Finally, let's list a simple example here:

![Image of Simple Net](simple.png)

```json
{
   "global" : {
      "w_updater_ph" : {
        "w_updater" : {
           "decay" : 0.01,
           "lr" : 0.01,
           "updater_type" : "SGD"
        }
      },
      "wb_updater_ph" : {
        "b_updater" : {
           "decay" : 0.01,
           "lr" : 0.01,
           "updater_type" : "SGD"
        },
        "w_updater" : {
           "decay" : 0.01,
           "lr" : 0.01,
           "updater_type" : "SGD"
        }
      }
   },
   
   "net_name" : "simple_net",
   "net_config" : [
      {
          "tag" : "Train",
          "max_iters" : 10000,
          "display_interval": 1,
          "save_interval": 1000,
          "save_name": "weights/matching_net_arc2",
          "out_nodes" : ["loss"]
      },
      {
          "tag" : "Valid",
          "max_iters" : 34,
          "display_interval" : 100,
          "save_interval" : 0,
          "save_name" : "",
          "out_nodes" : ["loss", "acc"]
      },
      {
          "tag" : "Test",
          "max_iters" : 34,
          "display_interval" : 100,
          "save_interval" : 0,
          "save_name" : "",
          "out_nodes" : ["loss", "acc"]
      }
   ],

   "layers" : [
      {
         "bottom_nodes" : null,
         "layer_idx" : 0,
         "layer_name" : "textdata",
         "layer_type" : 71,
         "setting" : {
            "batch_size" : 50,
            "data_file" : "msr_paraphrase_local_train_wid_dup.txt",
            "max_doc_len" : 31,
            "min_doc_len" : 5
         },
         "top_nodes" : [ "data", "label" ],
         "tag" : ["Train"]
      },
      {
         "bottom_nodes" : null,
         "layer_idx" : 0,
         "layer_name" : "textdata",
         "layer_type" : 71,
         "setting" : {
            "batch_size" : 50,
            "data_file" : "msr_paraphrase_local_valid_wid.txt",
            "max_doc_len" : 31,
            "min_doc_len" : 5
         },
         "top_nodes" : [ "data", "label" ],
         "tag" : ["Valid"]
      },
      {
         "bottom_nodes" : null,
         "layer_idx" : 0,
         "layer_name" : "textdata",
         "layer_type" : 71,
         "setting" : {
            "batch_size" : 50,
            "data_file" : "msr_paraphrase_test_wid.txt",
            "max_doc_len" : 31,
            "min_doc_len" : 5
         },
         "top_nodes" : [ "data", "label" ],
         "tag" : ["Test"]
      },
      {
         "bottom_nodes" : [ "data" ],
         "layer_idx" : 1,
         "layer_name" : "embedding",
         "layer_type" : 21,
         "setting" : {
            "embedding_file" : "wikicorp_50_msr.txt",
            "feat_size" : 50,
            "w_filler" : {
               "init_type" : 2,
               "range" : 0.01
            },
            "w_updater_ph" : 0,
            "word_count" : 14727
         },
         "top_nodes" : [ "embed" ]
      },
      {
         "bottom_nodes" : [ "embed" ],
         "layer_idx" : 2,
         "layer_name" : "split",
         "layer_type" : 20,
         "setting" : null,
         "top_nodes" : [ "splt1", "splt2" ]
      },
      {
         "bottom_nodes" : [ "splt1", "splt2" ],
         "layer_idx" : 5,
         "layer_name" : "match",
         "layer_type" : 23,
         "setting" : null,
         "top_nodes" : [ "cross" ]
      },
      {
         "bottom_nodes" : [ "cross" ],
         "layer_idx" : 6,
         "layer_name" : "maxpool1",
         "layer_type" : 15,
         "setting" : {
            "kernel_x" : 2,
            "kernel_y" : 2,
            "stride" : 2
         },
         "top_nodes" : [ "pool1" ]
      },
      {
         "bottom_nodes" : [ "pool1" ],
         "layer_idx" : 7,
         "layer_name" : "relu1",
         "layer_type" : 1,
         "setting" : null,
         "top_nodes" : [ "relu1" ]
      },
      {
         "bottom_nodes" : [ "relu1" ],
         "layer_idx" : 14,
         "layer_name" : "fc1",
         "layer_type" : 11,
         "setting" : {
            "b_filler" : {
               "init_type" : 0
            },
            "no_bias" : false,
            "num_hidden" : 512,
            "w_filler" : {
               "init_type" : 3,
               "sigma" : 0.005
            },
            "wb_updater_ph" : 0
         },
         "top_nodes" : [ "fc1" ]
      },
      {
         "bottom_nodes" : [ "fc1", "label" ],
         "layer_idx" : 18,
         "layer_name" : "softmax",
         "layer_type" : 51,
         "setting" : {
            "delta" : 1
         },
         "top_nodes" : [ "loss" ]
      },
      {
         "bottom_nodes" : [ "fc1", "label" ],
         "layer_idx" : 18,
         "layer_name" : "accuracy",
         "layer_type" : 56,
         "setting" : {
            "topk" : 1
         },
         "top_nodes" : [ "acc" ],
         "tag" : ["Valid", "Test"]
      }
   ],
   "save_activation": [
    {   
      "file_prefix": "./model/train", 
      "save_interval": 500, 
      "save_iter_num": 20, 
      "tag": "Train",
      "save_nodes" : ["x", "y", "acc", "loss", "softmax"]
    },  
    {   
      "file_prefix": "./model/valid", 
      "save_interval": 500, 
      "save_iter_num": 20, 
      "tag": "Valid",
      "save_nodes" : ["x", "y", "acc", "loss", "softmax"]
    },  
    {   
      "file_prefix": "./model/test", 
      "save_interval": 500, 
      "save_iter_num": 20, 
      "tag": "Test",
      "save_nodes" : ["x", "y", "acc", "loss", "softmax"]
    }   
  ],  
  "save_model": {
    "file_prefix": "./model/model", 
    "save_interval": 500 
  }
   
}
```