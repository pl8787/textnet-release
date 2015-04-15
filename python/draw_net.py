"""
Textnet network visualization: draw the Network json file.

NOTE: this requires pydot>=1.0.
"""
import json
import pydot
import argparse

# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record', 'fillcolor': '#6495ED',
         'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record', 'fillcolor': '#90EE90',
         'style': 'filled'}
BLOB_STYLE = {'shape': 'octagon', 'fillcolor': '#E0E0E0',
        'style': 'filled'}
        
Layer2Name = {
  0: 'UnkonwnLayer',
  # Activation Layer 1-10
  1: 'RectifiedLinear',
  2: 'Sigmoid', 
  3: 'Tanh', 

  # Common Layer 11-50
  11: 'FullConnect', 
  12: 'Flatten', 
  13: 'Dropout', 
  14: 'Conv', 
  15: 'MaxPooling', 
  16: 'SumPooling', 
  17: 'AvgPooling', 
  18: 'Concat', 
  19: 'ChConcat', 
  20: 'Split', 
  21: 'Embedding', 
  22: 'Cross', 
  23: 'Match', 
  24: 'Lstm', 
  25: 'WholePooling', 
  26: 'ConvolutionalLstm', 
  27: 'Recurrent', 
  28: 'SequenceDimReduction', 
  29: 'ConvLstmSplit', 

  # Loss Layer 51-70
  51: 'Softmax', 
  52: 'L2Loss', 
  53: 'MultiLogistic', 
  54: 'HingeLoss', 
  55: 'PairHingeLoss', 
  56: 'Accuracy', 

  # Input Layer 71-
  71: 'TextData', 
  72: 'SequenceClassificationData', 
  73: 'NextBasketData', 
}

def parse_args():
  """Parse input arguments
  """

  parser = argparse.ArgumentParser(description='Draw a network graph')

  parser.add_argument('input_net_json_file',
                      help='Input network json file')
  parser.add_argument('output_image_file',
                      help='Output image file')
  parser.add_argument('--details',
                      help='Need details of each layer.',
                      default='True')
  parser.add_argument('--rankdir',
                      help=('One of TB (top-bottom, i.e., vertical), '
                            'RL (right-left, i.e., horizontal), or another'
                            'valid dot option; see'
                            'http://www.graphviz.org/doc/info/attrs.html#k:rankdir'
                            '(default: LR)'),
                      default='LR')

  args = parser.parse_args()
  return args

def get_layer_type_name(layer_name):
  if type(layer_name) == int:
    return Layer2Name[layer_name]
  else:
    return str(layer_name)

def determine_edge_label_by_layertype(layer, layertype):
  """Define edge label based on layer type
  """

  edge_label = '""'

  return edge_label


def determine_node_label_by_layertype(layer, layertype, rankdir, need_details = True):
  """Define node label based on layer type
  """

  if rankdir in ('TB', 'BT'):
    # If graph orientation is vertical, horizontal space is free and
    # vertical space is not; separate words with spaces
    separator = ' '
  else:
    # If graph orientation is horizontal, vertical space is free and
    # horizontal space is not; separate words with newlines
    separator = '\n'
    
  node_label = '%s [%s] %s' % \
                (layer["layer_name"],
                 get_layer_type_name(layer["layer_type"]), 
                 separator)
  
  if need_details:  
    if layer["setting"] == None:
      layer["setting"] = {}

    node_label += separator.join(['%s: %s' % p for p in layer['setting'].items()])
    
  node_label = '"%s"' % node_label
  
  return node_label


def choose_color_by_layertype(layertype):
  """Define colors for nodes based on the layer type
  """
  color = '#6495ED'  # Default
  if layertype == 'Conv':
      color = '#FF5050'
  elif layertype == 'Embedding':
      color = '#FF9900'
  elif layertype == 'InnerProduct':
      color = '#CC33FF'
  return color


def get_pydot_graph(text_net, rankdir, label_edges=True, need_details=True):
  pydot_graph = pydot.Dot(text_net["net_name"], graph_type='digraph', rankdir=rankdir)
  pydot_nodes = {}
  pydot_edges = []
  for layer in text_net["layers"]:
    name = layer["layer_name"]
    # TODO - need convert to layer type name
    layertype = get_layer_type_name(layer["layer_type"])
    node_label = determine_node_label_by_layertype(layer, layertype, rankdir, need_details)
    
    # set None to 0 list
    if layer["bottom_nodes"] == None: 
      layer["bottom_nodes"] = []
    if layer["top_nodes"] == None:
      layer["top_nodes"] = []
    
    if (len(layer["bottom_nodes"]) == 1 and len(layer["top_nodes"]) == 1 and
        layer["bottom_nodes"][0] == layer["top_nodes"][0]):
      # We have an in-place neuron layer.
      pydot_nodes[name + '_' + layertype] = pydot.Node(
          node_label, **NEURON_LAYER_STYLE)
    else:
      layer_style = LAYER_STYLE_DEFAULT
      layer_style['fillcolor'] = choose_color_by_layertype(layertype)
      pydot_nodes[name + '_' + layertype] = pydot.Node(
          node_label, **layer_style)
    for bottom_node in layer["bottom_nodes"]:
      pydot_nodes[bottom_node + '_blob'] = pydot.Node(
        '%s' % (bottom_node), **BLOB_STYLE)
      edge_label = '""'
      pydot_edges.append({'src': bottom_node + '_blob',
                          'dst': name + '_' + layertype,
                          'label': edge_label})
    for top_node in layer["top_nodes"]:
      pydot_nodes[top_node + '_blob'] = pydot.Node(
        '%s' % (top_node))
      if label_edges:
        edge_label = determine_edge_label_by_layertype(layer, layertype)
      else:
        edge_label = '""'
      pydot_edges.append({'src': name + '_' + layertype,
                          'dst': top_node + '_blob',
                          'label': edge_label})
  # Now, add the nodes and edges to the graph.
  for node in pydot_nodes.values():
    pydot_graph.add_node(node)
  for edge in pydot_edges:
    pydot_graph.add_edge(
        pydot.Edge(pydot_nodes[edge['src']], pydot_nodes[edge['dst']],
                   label=edge['label']))
  return pydot_graph

def draw_net(text_net, rankdir, ext='png', need_details=True):
  return get_pydot_graph(text_net, rankdir, False, need_details).create(format=ext)

def draw_net_to_file(text_net, filename, rankdir='LR', need_details=True):
  ext = filename[filename.rfind('.')+1:]
  with open(filename, 'wb') as fid:
    fid.write(draw_net(text_net, rankdir, ext, need_details))

if __name__ == "__main__":
  args = parse_args()
  text_net = json.loads(open(args.input_net_json_file).read())
  print('Drawing net to %s' % args.output_image_file)
  need_details = True
  if args.details.lower() == 'true':
    need_details = True
  elif args.details.lower() == 'false':
    need_details = False
  draw_net_to_file(text_net, args.output_image_file, args.rankdir, need_details)

