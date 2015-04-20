Model Configuration File Description (JSON)
====
We use JSON as a protocol of our model. We describe each sections meaning below.

Net Name
====
Set the name of the net.

```json
"net_name" : "simple_net"
```

Net Configuration Section
====
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
In this section, we define some repeat parameters using in each layers, namely *place holder*. 

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

Here ```w_updater_ph``` is an arbitrary name of a *place holder*. When it occurs in any layer, we will replace it with:
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


Layers Section
====
In this section, we list all layers we use as a list. 

- layer_name : the name of this layer
- layer_idx : the index of this layer, set to any number you want
- layer_type : the type of this layer, string or int
- bottom_nodes : a list of bottom node name
- top_nodes : a list of top node name
- tag : the tag name list, identify which net contain this layer. If null, all net use this layer.
- tag_mode : two type *share* or *new*
  - share : share this layer with all net
  - new : create a new layer for each net
- setting : a map specified by different layers

To cope with sharing parameters between layers we can use these configuration below:
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
   ]
   
}
```