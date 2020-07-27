# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Building blocks for Graph Networks.

This module contains elementary building blocks of graph networks:

  - `broadcast_{field_1}_to_{field_2}` propagates the features from `field_1`
    onto the relevant elements of `field_2`;

  - `{field_1}To{field_2}Aggregator` propagates and then reduces the features
    from `field_1` onto the relevant elements of `field_2`;

  - the `EdgeBlock`, `NodeBlock` and `GlobalBlock` are elementary graph networks
    that only update the edges (resp. the nodes, the globals) of their input
    graph (as described in https://arxiv.org/abs/1806.01261).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import _base
from graph_nets import graphs
from graph_nets import utils_tf

import tensorflow as tf


NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE


def _validate_graph(graph, mandatory_fields, additional_message=None):
  for field in mandatory_fields:
    if getattr(graph, field) is None:
      message = "`{}` field cannot be None".format(field)
      if additional_message:
        message += " " + format(additional_message)
      message += "."
      raise ValueError(message)


def _validate_broadcasted_graph(graph, from_field, to_field):
  additional_message = "when broadcasting {} to {}".format(from_field, to_field)
  _validate_graph(graph, [from_field, to_field], additional_message)


def _get_static_num_nodes(graph):
  """Returns the static total number of nodes in a batch or None."""
  return None if graph.nodes is None else graph.nodes.shape.as_list()[0]


def _get_static_num_edges(graph):
  """Returns the static total number of edges in a batch or None."""
  return None if graph.senders is None else graph.senders.shape.as_list()[0]


def broadcast_globals_to_edges(graph, name="broadcast_globals_to_edges",
                               num_edges_hint=None):
  """Broadcasts the global features to the edges of a graph.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s, with globals features of
      shape `[n_graphs] + global_shape`, and `N_EDGE` field of shape
      `[n_graphs]`.
    name: (string, optional) A name for the operation.
    num_edges_hint: Integer indicating the total number of edges, if known.

  Returns:
    A tensor of shape `[n_edges] + global_shape`, where
    `n_edges = sum(graph.n_edge)`. The i-th element of this tensor is given by
    `globals[j]`, where j is the index of the graph the i-th edge belongs to
    (i.e. is such that
    `sum_{k < j} graphs.n_edge[k] <= i < sum_{k <= j} graphs.n_edge[k]`).

  Raises:
    ValueError: If either `graph.globals` or `graph.n_edge` is `None`.
  """
  _validate_broadcasted_graph(graph, GLOBALS, N_EDGE)
  with tf.name_scope(name):
    return utils_tf.repeat(graph.globals, graph.n_edge, axis=0,
                           sum_repeats_hint=num_edges_hint)


def broadcast_globals_to_nodes(graph, name="broadcast_globals_to_nodes",
                               num_nodes_hint=None):
  """Broadcasts the global features to the nodes of a graph.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s, with globals features of
      shape `[n_graphs] + global_shape`, and `N_NODE` field of shape
      `[n_graphs]`.
    name: (string, optional) A name for the operation.
    num_nodes_hint: Integer indicating the total number of nodes, if known.

  Returns:
    A tensor of shape `[n_nodes] + global_shape`, where
    `n_nodes = sum(graph.n_node)`. The i-th element of this tensor is given by
    `globals[j]`, where j is the index of the graph the i-th node belongs to
    (i.e. is such that
    `sum_{k < j} graphs.n_node[k] <= i < sum_{k <= j} graphs.n_node[k]`).

  Raises:
    ValueError: If either `graph.globals` or `graph.n_node` is `None`.
  """
  _validate_broadcasted_graph(graph, GLOBALS, N_NODE)
  with tf.name_scope(name):
    return utils_tf.repeat(graph.globals, graph.n_node, axis=0,
                           sum_repeats_hint=num_nodes_hint)


def broadcast_sender_nodes_to_edges(
    graph, name="broadcast_sender_nodes_to_edges"):
  """Broadcasts the node features to the edges they are sending into.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s, with nodes features of
      shape `[n_nodes] + node_shape`, and `senders` field of shape
      `[n_edges]`.
    name: (string, optional) A name for the operation.

  Returns:
    A tensor of shape `[n_edges] + node_shape`, where
    `n_edges = sum(graph.n_edge)`. The i-th element is given by
    `graph.nodes[graph.senders[i]]`.

  Raises:
    ValueError: If either `graph.nodes` or `graph.senders` is `None`.
  """
  _validate_broadcasted_graph(graph, NODES, SENDERS)
  with tf.name_scope(name):
    return tf.gather(graph.nodes, graph.senders)


def broadcast_receiver_nodes_to_edges(
    graph, name="broadcast_receiver_nodes_to_edges"):
  """Broadcasts the node features to the edges they are receiving from.

  Args:
    graph: A `graphs.GraphsTuple` containing `Tensor`s, with nodes features of
      shape `[n_nodes] + node_shape`, and receivers of shape `[n_edges]`.
    name: (string, optional) A name for the operation.

  Returns:
    A tensor of shape `[n_edges] + node_shape`, where
    `n_edges = sum(graph.n_edge)`. The i-th element is given by
    `graph.nodes[graph.receivers[i]]`.

  Raises:
    ValueError: If either `graph.nodes` or `graph.receivers` is `None`.
  """
  _validate_broadcasted_graph(graph, NODES, RECEIVERS)
  with tf.name_scope(name):
    return tf.gather(graph.nodes, graph.receivers)


class EdgesToGlobalsAggregator(_base.AbstractModule):
  """Aggregates all edges into globals."""

  def __init__(self, reducer, name="edges_to_globals_aggregator"):
    """Initializes the EdgesToGlobalsAggregator module.

    The reducer is used for combining per-edge features (one set of edge
    feature vectors per graph) to give per-graph features (one feature
    vector per graph). The reducer should take a `Tensor` of edge features, a
    `Tensor` of segment indices, and a number of graphs. It should be invariant
    under permutation of edge features within each graph.

    Examples of compatible reducers are:
    * tf.math.unsorted_segment_sum
    * tf.math.unsorted_segment_mean
    * tf.math.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-edge features to individual
        per-graph features.
      name: The module name.
    """
    super(EdgesToGlobalsAggregator, self).__init__(name=name)
    self._reducer = reducer

  def _build(self, graph):
    _validate_graph(graph, (EDGES,),
                    additional_message="when aggregating from edges.")
    num_graphs = utils_tf.get_num_graphs(graph)
    graph_index = tf.range(num_graphs)
    indices = utils_tf.repeat(graph_index, graph.n_edge, axis=0,
                              sum_repeats_hint=_get_static_num_edges(graph))
    return self._reducer(graph.edges, indices, num_graphs)


class NodesToGlobalsAggregator(_base.AbstractModule):
  """Aggregates all nodes into globals."""

  def __init__(self, reducer, name="nodes_to_globals_aggregator"):
    """Initializes the NodesToGlobalsAggregator module.

    The reducer is used for combining per-node features (one set of node
    feature vectors per graph) to give per-graph features (one feature
    vector per graph). The reducer should take a `Tensor` of node features, a
    `Tensor` of segment indices, and a number of graphs. It should be invariant
    under permutation of node features within each graph.

    Examples of compatible reducers are:
    * tf.math.unsorted_segment_sum
    * tf.math.unsorted_segment_mean
    * tf.math.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-node features to individual
        per-graph features.
      name: The module name.
    """
    super(NodesToGlobalsAggregator, self).__init__(name=name)
    self._reducer = reducer

  def _build(self, graph):
    _validate_graph(graph, (NODES,),
                    additional_message="when aggregating from nodes.")
    num_graphs = utils_tf.get_num_graphs(graph)
    graph_index = tf.range(num_graphs)
    indices = utils_tf.repeat(graph_index, graph.n_node, axis=0,
                              sum_repeats_hint=_get_static_num_nodes(graph))
    return self._reducer(graph.nodes, indices, num_graphs)


class _EdgesToNodesAggregator(_base.AbstractModule):
  """Agregates sent or received edges into the corresponding nodes."""

  def __init__(self, reducer, use_sent_edges=False,
               name="edges_to_nodes_aggregator"):
    super(_EdgesToNodesAggregator, self).__init__(name=name)
    self._reducer = reducer
    self._use_sent_edges = use_sent_edges

  def _build(self, graph):
    _validate_graph(graph, (EDGES, SENDERS, RECEIVERS,),
                    additional_message="when aggregating from edges.")
    # If the number of nodes are known at graph construction time (based on the
    # shape) then use that value to make the model compatible with XLA/TPU.
    if graph.nodes is not None and graph.nodes.shape.as_list()[0] is not None:
      num_nodes = graph.nodes.shape.as_list()[0]
    else:
      num_nodes = tf.reduce_sum(graph.n_node)
    indices = graph.senders if self._use_sent_edges else graph.receivers
    return self._reducer(graph.edges, indices, num_nodes)


class SentEdgesToNodesAggregator(_EdgesToNodesAggregator):
  """Agregates sent edges into the corresponding sender nodes."""

  def __init__(self, reducer, name="sent_edges_to_nodes_aggregator"):
    """Constructor.

    The reducer is used for combining per-edge features (one set of edge
    feature vectors per node) to give per-node features (one feature
    vector per node). The reducer should take a `Tensor` of edge features, a
    `Tensor` of segment indices, and a number of nodes. It should be invariant
    under permutation of edge features within each segment.

    Examples of compatible reducers are:
    * tf.math.unsorted_segment_sum
    * tf.math.unsorted_segment_mean
    * tf.math.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-edge features to individual
        per-node features.
      name: The module name.
    """
    super(SentEdgesToNodesAggregator, self).__init__(
        use_sent_edges=True,
        reducer=reducer,
        name=name)


class ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator):
  """Agregates received edges into the corresponding receiver nodes."""

  def __init__(self, reducer, name="received_edges_to_nodes_aggregator"):
    """Constructor.

    The reducer is used for combining per-edge features (one set of edge
    feature vectors per node) to give per-node features (one feature
    vector per node). The reducer should take a `Tensor` of edge features, a
    `Tensor` of segment indices, and a number of nodes. It should be invariant
    under permutation of edge features within each segment.

    Examples of compatible reducers are:
    * tf.math.unsorted_segment_sum
    * tf.math.unsorted_segment_mean
    * tf.math.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-edge features to individual
        per-node features.
      name: The module name.
    """
    super(ReceivedEdgesToNodesAggregator, self).__init__(
        use_sent_edges=False, reducer=reducer, name=name)


def _unsorted_segment_reduction_or_zero(reducer, values, indices, num_groups):
  """Common code for unsorted_segment_{min,max}_or_zero (below)."""
  reduced = reducer(values, indices, num_groups)
  present_indices = tf.math.unsorted_segment_max(
      tf.ones_like(indices, dtype=reduced.dtype), indices, num_groups)
  present_indices = tf.clip_by_value(present_indices, 0, 1)
  present_indices = tf.reshape(
      present_indices, [num_groups] + [1] * (reduced.shape.ndims - 1))
  reduced *= present_indices
  return reduced


def unsorted_segment_min_or_zero(values, indices, num_groups,
                                 name="unsorted_segment_min_or_zero"):
  """Aggregates information using elementwise min.

  Segments with no elements are given a "min" of zero instead of the most
  positive finite value possible (which is what `tf.math.unsorted_segment_min`
  would do).

  Args:
    values: A `Tensor` of per-element features.
    indices: A 1-D `Tensor` whose length is equal to `values`' first dimension.
    num_groups: A `Tensor`.
    name: (string, optional) A name for the operation.

  Returns:
    A `Tensor` of the same type as `values`.
  """
  with tf.name_scope(name):
    return _unsorted_segment_reduction_or_zero(
        tf.math.unsorted_segment_min, values, indices, num_groups)


def unsorted_segment_max_or_zero(values, indices, num_groups,
                                 name="unsorted_segment_max_or_zero"):
  """Aggregates information using elementwise max.

  Segments with no elements are given a "max" of zero instead of the most
  negative finite value possible (which is what `tf.math.unsorted_segment_max`
  would do).

  Args:
    values: A `Tensor` of per-element features.
    indices: A 1-D `Tensor` whose length is equal to `values`' first dimension.
    num_groups: A `Tensor`.
    name: (string, optional) A name for the operation.

  Returns:
    A `Tensor` of the same type as `values`.
  """
  with tf.name_scope(name):
    return _unsorted_segment_reduction_or_zero(
        tf.math.unsorted_segment_max, values, indices, num_groups)


class EdgeBlock(_base.AbstractModule):
  """Edge block.

  A block that updates the features of each edge in a batch of graphs based on
  (a subset of) the previous edge features, the features of the adjacent nodes,
  and the global features of the corresponding graph.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(self,
               edge_model_fn,
               use_edges=True,
               use_receiver_nodes=True,
               use_sender_nodes=True,
               use_globals=True,
               name="edge_block"):
    """Initializes the EdgeBlock module.

    Args:
      edge_model_fn: A callable that will be called in the variable scope of
        this EdgeBlock and should return a Sonnet module (or equivalent
        callable) to be used as the edge model. The returned module should take
        a `Tensor` (of concatenated input features for each edge) and return a
        `Tensor` (of output features for each edge). Typically, this module
        would input and output `Tensor`s of rank 2, but it may also be input or
        output larger ranks. See the `_build` method documentation for more
        details on the acceptable inputs to this module in that case.
      use_edges: (bool, default=True). Whether to condition on edge attributes.
      use_receiver_nodes: (bool, default=True). Whether to condition on receiver
        node attributes.
      use_sender_nodes: (bool, default=True). Whether to condition on sender
        node attributes.
      use_globals: (bool, default=True). Whether to condition on global
        attributes.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """
    super(EdgeBlock, self).__init__(name=name)

    if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
      raise ValueError("At least one of use_edges, use_sender_nodes, "
                       "use_receiver_nodes or use_globals must be True.")

    self._use_edges = use_edges
    self._use_receiver_nodes = use_receiver_nodes
    self._use_sender_nodes = use_sender_nodes
    self._use_globals = use_globals

    with self._enter_variable_scope():
      self._edge_model = edge_model_fn()

  def _collect_features(self, graph):
    edges_to_collect = []

    if self._use_edges:
      _validate_graph(graph, (EDGES,), "when use_edges == True")
      edges_to_collect.append(graph.edges)

    if self._use_receiver_nodes:
      edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))

    if self._use_sender_nodes:
      edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))

    if self._use_globals:
      num_edges_hint = _get_static_num_edges(graph)
      edges_to_collect.append(
          broadcast_globals_to_edges(graph, num_edges_hint=num_edges_hint))

    collected_edges = tf.concat(edges_to_collect, axis=-1)
    return collected_edges

  def _build(self, graph, **kwargs):
    """Connects the edge block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        features (if `use_edges` is `True`), individual nodes features (if
        `use_receiver_nodes` or `use_sender_nodes` is `True`) and per graph
        globals (if `use_globals` is `True`) should be concatenable on the last
        axis.
      **kwargs: Optional keyword arguments to pass to the Sonnet module.

    Returns:
      An output `graphs.GraphsTuple` with updated edges.

    Raises:
      ValueError: If `graph` does not have non-`None` receivers and senders, or
        if `graph` has `None` fields incompatible with the selected `use_edges`,
        `use_receiver_nodes`, `use_sender_nodes`, or `use_globals` options.
    """
    _validate_graph(
        graph, (SENDERS, RECEIVERS, N_EDGE), " when using an EdgeBlock")

    collected_edges = self._collect_features(graph)
    updated_edges = self._edge_model(collected_edges, **kwargs)
    return graph.replace(edges=updated_edges)


class NodeBlock(_base.AbstractModule):
  """Node block.

  A block that updates the features of each node in batch of graphs based on
  (a subset of) the previous node features, the aggregated features of the
  adjacent edges, and the global features of the corresponding graph.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(self,
               node_model_fn,
               use_received_edges=True,
               use_sent_edges=False,
               use_nodes=True,
               use_globals=True,
               received_edges_reducer=tf.math.unsorted_segment_sum,
               sent_edges_reducer=tf.math.unsorted_segment_sum,
               name="node_block"):
    """Initializes the NodeBlock module.

    Args:
      node_model_fn: A callable that will be called in the variable scope of
        this NodeBlock and should return a Sonnet module (or equivalent
        callable) to be used as the node model. The returned module should take
        a `Tensor` (of concatenated input features for each node) and return a
        `Tensor` (of output features for each node). Typically, this module
        would input and output `Tensor`s of rank 2, but it may also be input or
        output larger ranks. See the `_build` method documentation for more
        details on the acceptable inputs to this module in that case.
      use_received_edges: (bool, default=True) Whether to condition on
        aggregated edges received by each node.
      use_sent_edges: (bool, default=False) Whether to condition on aggregated
        edges sent by each node.
      use_nodes: (bool, default=True) Whether to condition on node attributes.
      use_globals: (bool, default=True) Whether to condition on global
        attributes.
      received_edges_reducer: Reduction to be used when aggregating received
        edges. This should be a callable whose signature matches
        `tf.math.unsorted_segment_sum`.
      sent_edges_reducer: Reduction to be used when aggregating sent edges.
        This should be a callable whose signature matches
        `tf.math.unsorted_segment_sum`.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """

    super(NodeBlock, self).__init__(name=name)

    if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
      raise ValueError("At least one of use_received_edges, use_sent_edges, "
                       "use_nodes or use_globals must be True.")

    self._use_received_edges = use_received_edges
    self._use_sent_edges = use_sent_edges
    self._use_nodes = use_nodes
    self._use_globals = use_globals

    with self._enter_variable_scope():
      self._node_model = node_model_fn()
      if self._use_received_edges:
        if received_edges_reducer is None:
          raise ValueError(
              "If `use_received_edges==True`, `received_edges_reducer` "
              "should not be None.")
        self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(
            received_edges_reducer)
      if self._use_sent_edges:
        if sent_edges_reducer is None:
          raise ValueError(
              "If `use_sent_edges==True`, `sent_edges_reducer` "
              "should not be None.")
        self._sent_edges_aggregator = SentEdgesToNodesAggregator(
            sent_edges_reducer)

  def _collect_features(self, graph):
    nodes_to_collect = []

    if self._use_received_edges:
      nodes_to_collect.append(self._received_edges_aggregator(graph))

    if self._use_sent_edges:
      nodes_to_collect.append(self._sent_edges_aggregator(graph))

    if self._use_nodes:
      _validate_graph(graph, (NODES,), "when use_nodes == True")
      nodes_to_collect.append(graph.nodes)

    if self._use_globals:
      # The hint will be an integer if the graph has node features and the total
      # number of nodes is known at tensorflow graph definition time, or None
      # otherwise.
      num_nodes_hint = _get_static_num_nodes(graph)
      nodes_to_collect.append(
          broadcast_globals_to_nodes(graph, num_nodes_hint=num_nodes_hint))

    collected_nodes = tf.concat(nodes_to_collect, axis=-1)
    return collected_nodes

  def _build(self, graph, **kwargs):
    """Connects the node block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        features (if `use_received_edges` or `use_sent_edges` is `True`),
        individual nodes features (if `use_nodes` is True) and per graph globals
        (if `use_globals` is `True`) should be concatenable on the last axis.
      **kwargs: Optional keyword arguments to pass to the Sonnet module.

    Returns:
      An output `graphs.GraphsTuple` with updated nodes.
    """

    collected_nodes = self._collect_features(graph)
    updated_nodes = self._node_model(collected_nodes, **kwargs)
    return graph.replace(nodes=updated_nodes)


class GlobalBlock(_base.AbstractModule):
  """Global block.

  A block that updates the global features of each graph in a batch based on
  (a subset of) the previous global features, the aggregated features of the
  edges of the graph, and the aggregated features of the nodes of the graph.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(self,
               global_model_fn,
               use_edges=True,
               use_nodes=True,
               use_globals=True,
               nodes_reducer=tf.math.unsorted_segment_sum,
               edges_reducer=tf.math.unsorted_segment_sum,
               name="global_block"):
    """Initializes the GlobalBlock module.

    Args:
      global_model_fn: A callable that will be called in the variable scope of
        this GlobalBlock and should return a Sonnet module (or equivalent
        callable) to be used as the global model. The returned module should
        take a `Tensor` (of concatenated input features) and return a `Tensor`
        (the global output features). Typically, this module would input and
        output `Tensor`s of rank 2, but it may also input or output larger
        ranks. See the `_build` method documentation for more details on the
        acceptable inputs to this module in that case.
      use_edges: (bool, default=True) Whether to condition on aggregated edges.
      use_nodes: (bool, default=True) Whether to condition on node attributes.
      use_globals: (bool, default=True) Whether to condition on global
        attributes.
      nodes_reducer: Reduction to be used when aggregating nodes. This should
        be a callable whose signature matches tf.math.unsorted_segment_sum.
      edges_reducer: Reduction to be used when aggregating edges. This should
        be a callable whose signature matches tf.math.unsorted_segment_sum.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """

    super(GlobalBlock, self).__init__(name=name)

    if not (use_nodes or use_edges or use_globals):
      raise ValueError("At least one of use_edges, "
                       "use_nodes or use_globals must be True.")

    self._use_edges = use_edges
    self._use_nodes = use_nodes
    self._use_globals = use_globals

    with self._enter_variable_scope():
      self._global_model = global_model_fn()
      if self._use_edges:
        if edges_reducer is None:
          raise ValueError(
              "If `use_edges==True`, `edges_reducer` should not be None.")
        self._edges_aggregator = EdgesToGlobalsAggregator(
            edges_reducer)
      if self._use_nodes:
        if nodes_reducer is None:
          raise ValueError(
              "If `use_nodes==True`, `nodes_reducer` should not be None.")
        self._nodes_aggregator = NodesToGlobalsAggregator(
            nodes_reducer)

  def _collect_features(self, graph):
    globals_to_collect = []

    if self._use_edges:
      _validate_graph(graph, (EDGES,), "when use_edges == True")
      globals_to_collect.append(self._edges_aggregator(graph))

    if self._use_nodes:
      _validate_graph(graph, (NODES,), "when use_nodes == True")
      globals_to_collect.append(self._nodes_aggregator(graph))

    if self._use_globals:
      _validate_graph(graph, (GLOBALS,), "when use_globals == True")
      globals_to_collect.append(graph.globals)

    collected_globals = tf.concat(globals_to_collect, axis=-1)
    return collected_globals

  def _build(self, graph, **kwargs):
    """Connects the global block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        (if `use_edges` is `True`), individual nodes (if `use_nodes` is True)
        and per graph globals (if `use_globals` is `True`) should be
        concatenable on the last axis.
      **kwargs: Optional keyword arguments to pass to the Sonnet module.

    Returns:
      An output `graphs.GraphsTuple` with updated globals.
    """
    collected_globals = self._collect_features(graph)
    updated_globals = self._global_model(collected_globals, **kwargs)
    return graph.replace(globals=updated_globals)


class NeighborhoodAggregator(_base.AbstractModule):
  """Agregates sender or receiver features into the corresponding nodes."""

  def __init__(self, reducer, to_senders=False,
               name="neighborhood_aggregator"):
    """Constructor.

    The reducer is used for combining one set of sender/receiver
    feature vectors per receiver/sender to give per-receiver/sender features
    (one feature vector per receiver/sender). The reducer should take a `Tensor`
    of sender/receiver features, a `Tensor` of segment indices, and a number of
    nodes. It should be invariant under permutation of sender/receiver features
    within each segment.

    Examples of compatible reducers are:
    * tf.math.unsorted_segment_sum
    * tf.math.unsorted_segment_mean
    * tf.math.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero

    Args:
      reducer: A function for reducing sets of per-sender features to individual
        per-receiver features.
      name: The module name.
    """
    super(NeighborhoodAggregator, self).__init__(name=name)
    self._reducer = reducer
    self._to_senders = to_senders

  def _build(self, graph):
    _validate_graph(graph, (EDGES, SENDERS, RECEIVERS,),
                    additional_message="when aggregating from node features.")
    # If the number of nodes are known at graph construction time (based on the
    # shape) then use that value to make the model compatible with XLA/TPU.
    if graph.nodes is not None and graph.nodes.shape.as_list()[0] is not None:
      num_nodes = graph.nodes.shape.as_list()[0]
    else:
      num_nodes = tf.reduce_sum(graph.n_node)
    indices = graph.senders if self._to_senders else graph.receivers
    broadcast = broadcast_receiver_nodes_to_edges if self._to_senders else broadcast_sender_nodes_to_edges
    return self._reducer(broadcast(graph), indices, num_nodes)


class RecurrentEdgeBlock(EdgeBlock):
  """Recurrent Edge block.

  A block that updates the features of each edge in a batch of graphs based on
  (a subset of) the previous edge features, the previous recurrent state,
  the features of the adjacent nodes, and the global features of the corresponding graph.
  The updating must uses a recurrent Sonnet model.
  """

  def __init__(self,
               edge_recurrent_model_fn,
               use_edges=True,
               use_receiver_nodes=True,
               use_sender_nodes=True,
               use_globals=True,
               name="recurrent edge_block"):
    """Initializes the RecurrentEdgeBlock module.

    Args:
      edge_recurrent_model_fn: A callable that will be called in the variable scope of
        this RecurrentEdgeBlock and should return a Sonnet recurrent module (or equivalent
        callable) to be used as the edge model. The returned recurrent module should take
        two `Tensors` (one of concatenated input features for each edge and other
        of previous recurrent state) and return a tuple of two `Tensors` (one of output
        features for each edge and other of next recurrent state). See the `_build` method
        documentation for more details on the acceptable inputs to this module in that case.
      use_edges: (bool, default=True). Whether to condition on edge attributes.
      use_receiver_nodes: (bool, default=True). Whether to condition on receiver
        node attributes.
      use_sender_nodes: (bool, default=True). Whether to condition on sender
        node attributes.
      use_globals: (bool, default=True). Whether to condition on global
        attributes.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """
    super(RecurrentEdgeBlock, self).__init__(edge_model_fn=edge_recurrent_model_fn,
                                             use_edges=use_edges,
                                             use_receiver_nodes=use_receiver_nodes,
                                             use_sender_nodes=use_sender_nodes,
                                             use_globals=use_globals,
                                             name=name)

  def initial_state(self, batch_size, **kwargs):
    """Constructs an initial state for the recurrent model.

    Args:
      batch_size: An int or an integral scalar `Tensor` representing batch size.
      **kwargs: Optional keyword arguments.
    Returns:
      Arbitrarily nested initial state for the recurrent model.
    """
    self._edge_model.initial_state(batch_size, **kwargs)

  def _build(self, graph, prev_state, **kwargs):
    """Connects the recurrent edge block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        features (if `use_edges` is `True`), individual nodes features (if
        `use_receiver_nodes` or `use_sender_nodes` is `True`) and per graph
        globals (if `use_globals` is `True`) should be concatenable on the last
        axis.
    prev_state: A previous recurrent state.
    **kwargs: Optional keyword arguments to pass to the Sonnet module.

    Returns:
      A tuple with two elements, an output `graphs.GraphsTuple` with updated edges and
      a next recurrent state.

    Raises:
      ValueError: If `graph` does not have non-`None` receivers, senders and prev_state,
         or if `graph` has `None` fields incompatible with the selected `use_edges`,
        `use_receiver_nodes`, `use_sender_nodes`, or `use_globals` options.
    """
    _validate_graph(
        graph, (SENDERS, RECEIVERS, N_EDGE), " when using an RecurrentEdgeBlock")

    collected_edges = self._collect_features(graph)
    updated_edges, next_state = self._edge_model(collected_edges, prev_state, **kwargs)
    return graph.replace(edges=updated_edges), next_state


class RecurrentNodeBlock(NodeBlock):
  """Recurrent Node block.

  A block that updates the features of each node in batch of graphs based on
  (a subset of) the previous node features, the previous recurrent state,
  the aggregated features of the adjacent edges, and the global features of
  the corresponding graph. The updating must uses a recurrent Sonnet model.
  """

  def __init__(self,
               node_recurrent_model_fn,
               use_received_edges=True,
               use_sent_edges=False,
               use_nodes=True,
               use_globals=True,
               aggregator_model_fn=NeighborhoodAggregator,
               neighborhood_reducer=tf.math.unsorted_segment_sum,
               received_edges_reducer=tf.math.unsorted_segment_sum,
               sent_edges_reducer=tf.math.unsorted_segment_sum,
               name="node_block"):
    """Initializes the RecurrentNodeBlock module.

    Args:
      node_recurrent_model_fn: A callable that will be called in the variable scope of
        this RecurrentNodeBlock and should return a Sonnet recurrent module (or equivalent
        callable) to be used as the node model. The returned recurrent module should take
        two `Tensors` (one of concatenated input features for each node and other
        of previous recurrent state) and return a tuple of two `Tensors` (one of output
        features for each node and other of next recurrent state). See the `_build` method
        documentation for more details on the acceptable inputs to this module in that case.
      use_received_edges: (bool, default=True) Whether to condition on
        aggregated edges received by each node.
      use_sent_edges: (bool, default=False) Whether to condition on aggregated
        edges sent by each node.
      use_nodes: (bool, default=True) Whether to condition on node attributes.
      use_globals: (bool, default=True) Whether to condition on global
        attributes.
      aggregator_model_fn: A callable that will be called in the variable scope
        of this RecurrentNodeBlock and should return a aggregator model that takes a
        `graphs.GraphsTuple` and return the reduction for each node neighborhood. The
        contruction shoud accepting a reducer, and a boolean `to_sender` which determines
        if how neighboor a neighboor should be defined from sender to receiver or from
        receiver to sender. The default aggregator model is a simple summation among
        neighborhood.
      neighborhood_reducer: Reduction to be used when aggregating neighborhood
        features. This should be a callable whose signature matches
        `tf.math.unsorted_segment_sum`.
      received_edges_reducer: Reduction to be used when aggregating received
        edges. This should be a callable whose signature matches
        `tf.math.unsorted_segment_sum`.
      sent_edges_reducer: Reduction to be used when aggregating sent edges.
        This should be a callable whose signature matches
        `tf.math.unsorted_segment_sum`.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """

    super(RecurrentNodeBlock, self).__init__(node_model_fn=node_recurrent_model_fn,
                                    use_received_edges=use_received_edges,
                                    use_sent_edges=use_sent_edges,
                                    use_nodes=use_nodes,
                                    use_globals=use_globals,
                                    received_edges_reducer=tf.math.unsorted_segment_sum,
                                    sent_edges_reducer=tf.math.unsorted_segment_sum,
                                    name=name)

    with self._enter_variable_scope():
      self._aggregator_model = aggregator_model_fn(neighborhood_reducer,
                                                   to_sender=False if self._use_received_edges else True)

  def initial_state(self, batch_size, **kwargs):
    """Constructs an initial state for the recurrent model.

    Args:
      batch_size: An int or an integral scalar `Tensor` representing batch size.
      **kwargs: Optional keyword arguments.
    Returns:
      Arbitrarily nested initial state for the recurrent model.
    """
    self._node_model.initial_state(batch_size, **kwargs)


  def _build(self, graph, prev_state, **kwargs):
    """Connects the recurrent node block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        features (if `use_received_edges` or `use_sent_edges` is `True`),
        individual nodes features (if `use_nodes` is True) and per graph globals
        (if `use_globals` is `True`) should be concatenable on the last axis.
      prev_state: A previous recurrent state.
      **kwargs: Optional keyword arguments to pass to the Sonnet module.

    Returns:
      An output `graphs.GraphsTuple` with updated nodes.
    """

    collected_nodes = self._collect_features(graph)
    aggregated_nodes = self._aggregator_model(graph.replace(nodes=collected_nodes))
    updated_nodes, next_state = self._node_model(aggregated_nodes, prev_state, **kwargs)
    return graph.replace(nodes=updated_nodes), next_state


class RecurrentGlobalBlock(GlobalBlock):
  """Global block.

  A block that updates the global features of each graph in a batch based on
  (a subset of) the previous global features, the previous recurrent state,
  the aggregated features of the edges of the graph, and the aggregated features
  of the nodes of the graph. The updating must uses a recurrent Sonnet model.
  """

  def __init__(self,
               global_recurrent_model_fn,
               use_edges=True,
               use_nodes=True,
               use_globals=True,
               nodes_reducer=tf.math.unsorted_segment_sum,
               edges_reducer=tf.math.unsorted_segment_sum,
               name="recurrent_global_block"):
    """Initializes the RecurrentGlobalBlock module.
        this RecurrentNodeBlock and should return a Sonnet recurrent module (or equivalent
        callable) to be used as the node model. The returned recurrent module should take
        two `Tensors` (one of concatenated input features for each node and other
        of previous recurrent state) and return a tuple of two `Tensors` (one of output
        features for each node and other of next recurrent state). See the `_build` method
        documentation for more details on the acceptable inputs to this module in that case.

    Args:
      global_recurrent_model_fn: A callable that will be called in the variable scope of
        this RecurrentGlobalBlock and should return a Sonnet recurrent module
        (or equivalent callable) to be used as the global model. The returned recurrent
        module should take two `Tensors` (one of concatenated input features for each
        node and other of previous recurrent state) and return a tuple of two `Tensors`
        (one of output features for each node and other of next recurrent state).
         See the `_build` method documentation for more details on the acceptable inputs
         to this module in that case.
      use_edges: (bool, default=True) Whether to condition on aggregated edges.
      use_nodes: (bool, default=True) Whether to condition on node attributes.
      use_globals: (bool, default=True) Whether to condition on global
        attributes.
      nodes_reducer: Reduction to be used when aggregating nodes. This should
        be a callable whose signature matches tf.math.unsorted_segment_sum.
      edges_reducer: Reduction to be used when aggregating edges. This should
        be a callable whose signature matches tf.math.unsorted_segment_sum.
      name: The module name.

    Raises:
      ValueError: When fields that are required are missing.
    """

    super(GlobalBlock, self).__init__(global_model_fn=global_recurrent_model_fn,
                                      use_edges=use_edges,
                                      use_nodes=use_nodes,
                                      use_globals=use_globals,
                                      nodes_reducer=nodes_reducer,
                                      edges_reducer=edges_reducer,
                                      name="recurrent_global_block")

  def initial_state(self, batch_size, **kwargs):
    """Constructs an initial state for the recurrent model.

    Args:
      batch_size: An int or an integral scalar `Tensor` representing batch size.
      **kwargs: Optional keyword arguments.
    Returns:
      Arbitrarily nested initial state for the recurrent model.
    """
    self._node_model.initial_state(batch_size, **kwargs)

  def _build(self, graph, prev_state, **kwargs):
    """Connects the recurrent global block.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, whose individual edges
        (if `use_edges` is `True`), individual nodes (if `use_nodes` is True)
        and per graph globals (if `use_globals` is `True`) should be
        concatenable on the last axis.
    prev_state: A previous recurrent state.
    **kwargs: Optional keyword arguments to pass to the Sonnet module.

    Returns:
      An output `graphs.GraphsTuple` with updated globals.
    """
    collected_globals = self._collect_features(graph)
    updated_globals, next_state = self._global_model(collected_globals, prev_state, **kwargs)
    return graph.replace(globals=updated_globals), next_state

