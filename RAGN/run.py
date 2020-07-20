import os
import time

import numpy as np
import pandas as pd
import networkx as nx
import sonnet as snt
import tensorflow as tf
import matplotlib.pyplot as plt

import pytop
from graph_nets import graphs
from graph_nets import blocks
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

SEED = 2
NUM_LAYERS = 3
LATENT_SIZE = 24

def create_placeholders(batch_generator):
    input_graphs, target_graphs, _ = next(batch_generator)
    input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs)

    dtype = tf.as_dtype(utils_np.networkxs_to_graphs_tuple(target_graphs).edges.dtype)
    weight_ph = tf.placeholder(dtype, name="loss_weights")
    is_training_ph = tf.placeholder(tf.bool, name="training_flag")
    return input_ph, target_ph, weight_ph, is_training_ph

def create_feed_dict(batch_tuple, is_training, weights, input_ph, target_ph, is_training_ph, weight_ph):
    inputs, targets, pos = batch_tuple
    # inputs, targets, pos = next(batch_generator)
    input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets)

    if weights[0] != 1 or weights[1] != 1:
        batch_weights = np.ones(target_graphs.edges.shape[0])
        target_args = np.argmax(target_graphs.edges, axis=-1)
        batch_weights[target_args == 0] *= weights[0]
        batch_weights[target_args == 1] *= weights[1]
    else:
        batch_weights = 1

    feed_dict = {input_ph: input_graphs, target_ph: target_graphs, is_training_ph: is_training, weight_ph: batch_weights}
    return feed_dict, pos

def compute_accuracy(target, output, distribution=False):
    acc_all = []
    solved_all = []
    acc_true_all = []
    acc_false_all = []
    solved_true_all = []
    solved_false_all = []

    tg_dict = utils_np.graphs_tuple_to_data_dicts(target)
    out_dict = utils_np.graphs_tuple_to_data_dicts(output)
    for tg_graph, out_graph in zip(tg_dict, out_dict):
        expect = np.argmax(tg_graph["edges"], axis=-1)
        predict = np.argmax(out_graph["edges"], axis=-1)
        true_mask = np.ma.masked_equal(expect, 1).mask
        false_mask = np.ma.masked_equal(expect, 0).mask

        acc = (expect == predict)
        acc_true = acc[true_mask]
        acc_false = acc[false_mask]

        solved = np.all(acc)
        solved_true = np.all(acc_true)
        solved_false = np.all(acc_false)

        acc_all.append(np.mean(acc))
        acc_true_all.append(np.mean(acc_true))
        acc_false_all.append(np.mean(acc_false))

        solved_all.append(solved)
        solved_true_all.append(solved_true)
        solved_false_all.append(solved_false)
    acc_all = np.stack(acc_all)
    acc_true_all = np.stack(acc_true_all)
    acc_false_all = np.stack(acc_false_all)

    solved_all = np.stack(solved_all)
    solved_true_all = np.stack(solved_true_all)
    solved_false_all = np.stack(solved_false_all)
    if not distribution:
        acc_all = np.mean(acc_all)
        acc_true_all = np.mean(acc_true_all)
        acc_false_all = np.mean(acc_false_all)

        solved_all = np.mean(solved_all)
        solved_true_all = np.mean(solved_true_all)
        solved_false_all = np.mean(solved_false_all)
    return acc_all, solved_all, acc_true_all, solved_true_all, acc_false_all, solved_false_all

def get_generator_path_metrics(inputs, targets, outputs):
    num_attempts = 3
    out_dicts = utils_np.graphs_tuple_to_data_dicts(outputs)
    in_dicts = utils_np.graphs_tuple_to_data_dicts(inputs)
    tg_dicts = utils_np.graphs_tuple_to_data_dicts(targets)

    def softmax_prob(x):  # pylint: disable=redefined-outer-name
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    n_graphs = len(tg_dicts)
    for tg_graph, out_graph, in_graph, idx_graph in zip(tg_dicts, out_dicts, in_dicts, range(n_graphs)):
        n_node = out_graph["n_node"]
        tg_graph_dist = tg_graph["nodes"][:,0]
        tg_graph_hops = tg_graph["nodes"][:,1]
        out_graph_dist = np.zeros_like(tg_graph_dist)
        out_graph_hops = np.zeros_like(tg_graph_dist)
        end_node = np.argwhere(tg_graph_dist == 0).reshape(1)[0]
        for node in range(n_node):
            hops = 0
            strength = 0
            start = node
            sender = None
            reachable = True
            path = np.zeros(n_node, dtype=int)
            while start != end_node:
                path[start] += 1
                start_edges_idx = np.argwhere(out_graph["senders"] == start).reshape(-1,)
                receivers = out_graph["receivers"][start_edges_idx]
                start_edges = out_graph["edges"][start_edges_idx]
                edges_prob = softmax_prob(start_edges)
                routing_links = edges_prob[:, 0] < edges_prob[:, 1]

                if path[start] > num_attempts:
                    if end_node in receivers:
                        edge_forward_idx = np.argwhere(receivers == end_node).reshape(-1,)[0]
                        routing_links = np.ones_like(routing_links, dtype=bool)
                        sender = start
                        start = end_node
                    else:
                        reachable = False
                        break
                else:
                    if not np.any(routing_links):
                        routing_links = ~routing_links
                        edges_idx_sort = np.argsort(edges_prob[:, 0] - edges_prob[:, 1])[::-1]
                    else:
                        edges_idx_sort = np.argsort(edges_prob[routing_links][:, 1])

                    if path[start] <= len(edges_idx_sort):
                        edge_forward_idx = edges_idx_sort[-path[start]]
                    else:
                        edge_forward_idx = edges_idx_sort[-1]

                    sender = start
                    start = receivers[routing_links][edge_forward_idx]
                hops += 1
                strength += in_graph["edges"][start_edges_idx][routing_links][edge_forward_idx][0]
            if reachable:
                out_graph_dist[node] = strength
                out_graph_hops[node] = hops
        out_graph_hops = np.delete(out_graph_hops, end_node)
        out_graph_dist = np.delete(out_graph_dist, end_node)
        tg_graph_hops = np.delete(tg_graph_hops, end_node)
        tg_graph_dist = np.delete(tg_graph_dist, end_node)
        idx_non_zero = np.flatnonzero(out_graph_hops)
        unreachable_p =  1 - idx_non_zero.size / out_graph_dist.size
        if idx_non_zero.size > 0:
            diff_dist = (tg_graph_dist[idx_non_zero] / out_graph_dist[idx_non_zero])
            diff_hops = (tg_graph_hops[idx_non_zero] / out_graph_hops[idx_non_zero])
            yield (diff_dist, diff_hops, unreachable_p)
        else:
            yield (None, None, unreachable_p)

def aggregator_path_metrics(inputs, targets, outputs, distribution=False):
    n_graphs = targets.n_node.size
    idx_graph = 0
    none_idx = []
    hist_hops = []
    hist_dist = []
    batch_max_dist_diff = np.zeros(n_graphs)
    batch_min_dist_diff = np.zeros(n_graphs)
    batch_avg_dist_diff = np.zeros(n_graphs)
    batch_max_hops_diff = np.zeros(n_graphs)
    batch_min_hops_diff = np.zeros(n_graphs)
    batch_avg_hops_diff = np.zeros(n_graphs)
    batch_unreachable_p = np.zeros(n_graphs)
    metrics_graph_generator = get_generator_path_metrics(inputs, targets, outputs)
    for diff_dist, diff_hops, unreachable_p in metrics_graph_generator:
        batch_unreachable_p[idx_graph] = unreachable_p

        if np.any(diff_dist == None):
            none_idx.append(idx_graph)
        else:
            batch_max_dist_diff[idx_graph] = np.max(diff_dist)
            batch_min_dist_diff[idx_graph] = np.min(diff_dist)
            batch_avg_dist_diff[idx_graph] = np.mean(diff_dist)
            batch_max_hops_diff[idx_graph] = np.max(diff_hops)
            batch_min_hops_diff[idx_graph] = np.min(diff_hops)
            batch_avg_hops_diff[idx_graph] = np.mean(diff_hops)
            if distribution:
                hist_hops.append(diff_hops)
                hist_dist.append(diff_dist)
        idx_graph += 1
    batch_max_dist_diff = np.delete(batch_max_dist_diff, none_idx)
    batch_min_dist_diff = np.delete(batch_min_dist_diff, none_idx)
    batch_avg_dist_diff = np.delete(batch_avg_dist_diff, none_idx)
    batch_max_hops_diff = np.delete(batch_max_hops_diff, none_idx)
    batch_min_hops_diff = np.delete(batch_min_hops_diff, none_idx)
    batch_avg_hops_diff = np.delete(batch_avg_hops_diff, none_idx)
    if not distribution:
        return dict(avg_batch_max_dist_diff=np.mean(batch_max_dist_diff) if batch_max_dist_diff.size else np.infty,
                    avg_batch_min_dist_diff=np.mean(batch_min_dist_diff) if batch_min_dist_diff.size else np.infty,
                    avg_batch_avg_dist_diff=np.mean(batch_avg_dist_diff) if batch_avg_dist_diff.size else np.infty,
                    avg_batch_max_hops_diff=np.mean(batch_max_hops_diff) if batch_max_hops_diff.size else np.infty,
                    avg_batch_min_hops_diff=np.mean(batch_min_hops_diff) if batch_min_hops_diff.size else np.infty,
                    avg_batch_avg_hops_diff=np.mean(batch_avg_hops_diff) if batch_avg_hops_diff.size else np.infty,
                    max_batch_unreachable_p=np.max(batch_unreachable_p),
                    min_batch_unreachable_p=np.min(batch_unreachable_p),
                    avg_batch_unreachable_p=np.mean(batch_unreachable_p))
    else:
        return {"percentage of unreachable paths":batch_unreachable_p, "difference of hops":np.concatenate(hist_hops), "difference of strength":np.concatenate(hist_dist)}

def create_loss_ops(target_op, output_ops, weight):
    loss_ops = [
        tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges, weights=weight)
        for output_op in output_ops
    ]
    return loss_ops

def make_all_runnable_in_session(*args):
    return [utils_tf.make_runnable_in_session(a) for a in args]

class LeakyReluMLP(snt.AbstractModule):
    def __init__(self,
                 hidden_size,
                 n_layers,
                 name="LeakyReluMLP"):
        super(LeakyReluMLP, self).__init__(name=name)
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        with self._enter_variable_scope():
            self._linear_layers = []
            for _ in range(self._n_layers - 1):
                self._linear_layers.append(snt.Linear(int(np.floor(self._hidden_size * .7))))
            self._linear_layers.append(snt.Linear(self._hidden_size))

    def _build(self, inputs, is_training):
        outputs_op = inputs
        for linear in self._linear_layers:
            outputs_op = linear(outputs_op)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=0.2)
        return outputs_op


class LeakyReluNormMLP(snt.AbstractModule):
    def __init__(self,
                 hidden_size,
                 n_layers,
                 dropou=0.75,
                 name="LeakyReluNormMLP"):
        super(LeakyReluNormMLP, self).__init__(name=name)
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        with self._enter_variable_scope():
            self._linear_layers = []
            self._bn_layers = []
            for _ in range(self._n_layers - 1):
                self._linear_layers.append(snt.Linear(int(np.floor(self._hidden_size * .7))))
                self._bn_layers.append(snt.BatchNorm())
            self._linear_layers.append(snt.Linear(self._hidden_size))
            self._bn_layers.append(snt.BatchNorm())

    def _build(self, inputs, is_training):
        outputs_op = inputs
        for linear, bn in zip(self._linear_layers, self._bn_layers):
            outputs_op = linear(outputs_op)
            outputs_op = bn(outputs_op, is_training=is_training, test_local_stats=True)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=0.2)
        return outputs_op

class LeakyReluNormGRU(snt.AbstractModule):
    def __init__(self,
                 hidden_size,
                 recurrent_dropout=0.75,
                 name="LeakyReluNormGRU"):
        super(LeakyReluNormGRU, self).__init__(name=name)
        self._hidden_size = hidden_size
        with self._enter_variable_scope():
            self._dropout_gru, self._gru = snt.lstm_with_recurrent_dropout(self._hidden_size, keep_prob=recurrent_dropout)
            # self._gru = snt.GRU(self._hidden_size)
            # self._dropout_gru = snt.python.modules.gated_rnn.RecurrentDropoutWrapper(self._gru, recurrent_dropout)
            self._batch_norm = snt.BatchNorm()

    def get_initial_state(self, batch_size, dtype=tf.float64):
        return self._dropout_gru.initial_state(batch_size, dtype=dtype)

    def _build(self, inputs, prev_states, is_training):
        def true_fn():
            return self._dropout_gru(inputs, prev_states)

        def false_fn():
            o, ns = self._gru(inputs, prev_states[0])
            ns = (ns, [tf.ones_like(ns, name="FoolMask")])
            return o, ns

        outputs_op, next_states = tf.cond(is_training, true_fn=true_fn, false_fn=false_fn)
        outputs_op = self._batch_norm(outputs_op, is_training=is_training, test_local_stats=True)
        outputs_op = tf.nn.leaky_relu(outputs_op, alpha=0.2)
        return outputs_op, next_states

def make_gru_model(size=LATENT_SIZE):
    return LeakyReluNormGRU(size)

def make_mlp_model(size=LATENT_SIZE, n_layers=NUM_LAYERS, model=LeakyReluNormMLP):
    return model(size, n_layers)

class LocalRoutingNetwork(snt.AbstractModule):
    def __init__(self,
                 output_size,
                 model_fn=make_mlp_model,
                 n_heads=3,
                 name="LocalRoutingNetwork"):
        super(LocalRoutingNetwork, self).__init__(name=name)
        self._multihead_models = []
        with self._enter_variable_scope():
            self._routing_layer = snt.Linear(output_size)
            self._final_node_model = model_fn(model=LeakyReluMLP)
            self._final_query_model = model_fn(model=LeakyReluMLP)
            for _ in range(n_heads):
                self._multihead_models.append( [model_fn(model=LeakyReluMLP),
                                                model_fn(size=12, model=LeakyReluMLP),
                                                model_fn(size=12, model=LeakyReluMLP),
                                                model_fn(size=8, model=LeakyReluMLP)] )

    def _build(self, inputs, **kwargs):
        queries = utils_tf.repeat(inputs.globals, inputs.n_edge)
        senders_feature = tf.gather(inputs.nodes, inputs.senders)
        receivers_feature = tf.gather(inputs.nodes, inputs.receivers)
        edge_rec_pair = tf.concat([inputs.edges, receivers_feature], -1)

        multihead_routing = []
        for query_model, sender_model, edge_rec_model, multi_model in self._multihead_models:
            enc_queries = query_model(queries, **kwargs)
            enc_senders = sender_model(senders_feature, **kwargs)
            enc_edge_rec = edge_rec_model(edge_rec_pair, **kwargs)
            att_op = tf.reduce_sum(
              tf.multiply(tf.concat([enc_senders, enc_edge_rec], -1), enc_queries),
              -1,
              keepdims=True
            )
            attention_input = tf.nn.leaky_relu(att_op, alpha=0.2)
            attentions = self._unsorted_segment_softmax(attention_input, inputs.senders, tf.reduce_sum(inputs.n_node))

            multhead_op1 = multi_model(edge_rec_pair, **kwargs)
            multhead_op2 = tf.multiply(attentions, multhead_op1)
            multhead_op3 = tf.unsorted_segment_sum(multhead_op2, inputs.senders, tf.reduce_sum(inputs.n_node))
            multihead_routing.append(multhead_op2)
        node_attention_feature = tf.concat(multihead_routing, -1)
        final_features = self._final_node_model(tf.concat(
          [tf.gather(node_attention_feature, inputs.senders), inputs.edges], -1), **kwargs)
        final_queries = self._final_query_model(queries, **kwargs)
        output_edges = self._routing_layer(tf.multiply(final_features, final_queries))
        return inputs.replace(edges=output_edges)

    def _unsorted_segment_softmax(self, x, idx, n_idx):
        op1 = tf.exp(x)
        op2 = tf.unsorted_segment_sum(op1, idx, n_idx)
        op3 = tf.gather(op2, idx)
        op4 = tf.divide(op1, op3)
        return op4

class MLPGraphIndependent(snt.AbstractModule):
    def __init__(self, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=make_mlp_model,
                node_model_fn=make_mlp_model,
                global_model_fn=None)

    def _build(self, inputs, **kwargs):
        return self._network(inputs, **kwargs)

class GraphGatedNonLocalNetwork(snt.AbstractModule):
    def __init__(self,
                 gate_recurrent_model_fn=make_gru_model,
                 bias_shape=[LATENT_SIZE * 3],
                 reducer=tf.unsorted_segment_sum,
                 name="GraphGatedNonLocalNetwork"):
        super(GraphGatedNonLocalNetwork, self).__init__(name=name)

        with self._enter_variable_scope():
            self._edge_block = blocks.GatedEdgeBlock(
                gate_recurrent_model_fn=gate_recurrent_model_fn,
                use_edges=True,
                use_receiver_nodes=True,
                use_sender_nodes=True,
                use_globals=False
            )
            self._node_block = blocks.GatedNodeBlock(
                gate_recurrent_model_fn=gate_recurrent_model_fn,
                bias_shape=bias_shape,
                use_received_edges=True,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False
            )

    def reset_state(self, edge_batch_size,  node_batch_size, edge_state=None, node_state=None):
        self._edge_block.reset_state(edge_batch_size, state=edge_state)
        self._node_block.reset_state(node_batch_size, state=node_state)

    def _build(self, graph, **kwargs):
        return self._node_block(self._edge_block(graph, **kwargs), **kwargs)

class EncodeProcessDecode(snt.AbstractModule):
    def __init__(self, edge_output_size, name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        with self._enter_variable_scope():
            self._encoder = MLPGraphIndependent()
            self._core = GraphGatedNonLocalNetwork()
            #self._decoder = MLPGraphIndependent()
            self._lookup = LocalRoutingNetwork(edge_output_size)

    def _build(self, input_op, num_processing_steps, is_training):
        latent0 = self._encoder(input_op, is_training=is_training)
        latent = latent0
        output_ops = []
        node_batch_size = tf.reduce_sum(latent.n_node)
        edge_batch_size = tf.reduce_sum(latent.n_edge)
        self._core.reset_state(edge_batch_size, node_batch_size)
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1, use_global=False)
            latent = self._core(core_input, is_training=is_training)
            #decoded_op = self._decoder(latent, is_training=is_training)
            output_ops.append(self._lookup(latent, is_training=is_training))
        return output_ops



if __name__ == "__main__":
    tf.set_random_seed(SEED)
    tf.reset_default_graph()

    random_state = np.random.RandomState(seed=SEED)

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 20
    num_processing_steps_ge = 20

    # Data / training parameters.
    #num_training_iterations = 50000

    batch_size_tr = 128
    batch_size_ge = 128

    # Number of nodes per graph sampled uniformly from this range.
    min_num_nodes_tr = 8
    max_num_nodes_tr = 20
    min_num_nodes_ge = 25
    max_num_nodes_ge = 35
    num_nodes_min_max_tr = (min_num_nodes_tr, max_num_nodes_tr)
    num_nodes_min_max_ge = (min_num_nodes_ge, max_num_nodes_ge)

    batch_generator_tr = pytop.batch_brite_generator(
        batch_size_tr, num_nodes_min_max_tr, random_state=random_state)
    batch_generator_ge = pytop.batch_brite_generator(
        batch_size_ge, num_nodes_min_max_ge, random_state=random_state)

    # Data.
    # Input and target placeholders.
    input_ph, target_ph, weight_ph, is_training_ph = create_placeholders(batch_generator_tr)

    # Connect the data to the model.
    # Instantiate the model.
    model = EncodeProcessDecode(edge_output_size=2)
    # A list of outputs, one per processing step.
    output_ops_tr = model(input_ph, num_processing_steps_tr, is_training_ph)
    output_ops_ge = model(input_ph, num_processing_steps_ge, is_training_ph)

    # Training loss.
    loss_ops_tr = create_loss_ops(target_ph, output_ops_tr, weight_ph)
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
    # Test/generalization loss.
    loss_ops_ge = create_loss_ops(target_ph, output_ops_ge, weight_ph)
    loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

    # Optimizer.
    ## Fixed
    #learning_rate = 1e-3
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    #step_op = optimizer.minimize(loss_op_tr)
    ## Dynamically TF Way
    starter_learning_rate = 5e-3
    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.cosine_decay_restarts(starter_learning_rate, global_step, first_decay_steps=1000, m_mul=0.99, alpha=5e-6)
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, decay_steps=100000, end_learning_rate=1e-5, power=3)
    optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, momentum=0.6)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, momentum=0.6)
    step_op = optimizer.minimize(loss_op_tr, global_step=global_step)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

    saver = tf.train.Saver()

    try:
        sess.close()
    except NameError:
        pass
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    logged_iterations = []
    losses_tr = []
    corrects_tr = []
    solveds_tr = []
    losses_ge = []
    corrects_ge = []
    solveds_ge = []

    last_iteration = 0
    last_epoch = 0
    best_acc = 0

    restore_path = "weights/last/"
    # if os.path.isdir(restore_path):
    #     saver.restore(sess, os.path.join(restore_path, "dm.ckpt"))
        # sess.run(global_step.assign(10000))

    log_every_seconds = 30

    f = open("training.out", "w")
    header = "Iteration,Elapsed Time (s),Loss Tr,Accuracy Tr,Solved Tr,Accuracy Ge,Solved Ge,\
True Accuracy Tr,True Solved Tr,True Accuracy Ge,True Solved Ge,False Accuracy Tr,False Solved Tr,\
False Accuracy Ge,False Solved Ge,Avg Batch Max Dist Diff Tr,Avg Batch Min Dist Diff Tr,\
Avg Batch Avg Dist Diff Tr,Avg Batch Max Hops Diff Tr,Avg Batch Min Hops Diff Tr,Avg Batch Avg Hops Diff Tr,\
Max Batch unreachable Tr,Min Batch unreachable Tr,Avg Batch unreachable Tr,Avg Batch Max Dist Diff Ge,\
Avg Batch Min Dist Diff Ge,Avg Batch Avg Dist Diff Ge,Avg Batch Max Hops Diff Ge,Avg Batch Min Hops Diff Ge,\
Avg Batch Avg Hops Diff Ge,Max Batch unreachable Ge,Min Batch unreachable Ge,Avg Batch unreachable Ge\n"

    f.write(header)

    start_time = time.time()
    last_log_time = start_time

    n_epochs = 3
    batch_gen_val = next(pytop.batch_files_generator("../../valid_generalization", -1))
    batch_non_gen_val = next(pytop.batch_files_generator("../../valid_non_generalization", -1))
    for epoch in range(last_epoch, n_epochs):
        iteration = 0
        batch_generator_tr = pytop.batch_files_generator("../../train", batch_size_tr)
        for batch_tuple in batch_generator_tr:
            if iteration >= last_iteration:
                feed_dict, _ = create_feed_dict(batch_tuple, True, [.4, 1], input_ph, target_ph, is_training_ph, weight_ph)
                train_values = sess.run({
                    "step": step_op,
                    "input": input_ph,
                    "target": target_ph,
                    "loss": loss_op_tr,
                    "outputs": output_ops_tr
                },
                    feed_dict=feed_dict)

                the_time = time.time()
                elapsed_since_last_log = the_time - last_log_time
                if (elapsed_since_last_log > log_every_seconds) or ((iteration + 1) % 1000 == 0):
                    last_log_time = the_time

                    feed_dict, _ = create_feed_dict(batch_non_gen_val, False, [.4, 1], input_ph, target_ph, is_training_ph, weight_ph)
                    non_gen_values = sess.run({
                        "input": input_ph,
                        "target": target_ph,
                        "loss": loss_op_ge,
                        "outputs": output_ops_ge
                    },
                        feed_dict=feed_dict)
                    # feed_dict, _ = create_feed_dict(batch_gen_val, False, [.4, 1], input_ph, target_ph, is_training_ph, weight_ph)
                    # gen_values = sess.run({
                    #     "input": input_ph,
                    #     "target": target_ph,
                    #     "loss": loss_op_ge,
                    #     "outputs": output_ops_ge
                    # },
                    #     feed_dict=feed_dict)

                    tr_path_metrics = aggregator_path_metrics(non_gen_values["input"], non_gen_values["target"], non_gen_values["outputs"][-1])
                    # ge_path_metrics = aggregator_path_metrics(gen_values["input"], gen_values["target"], gen_values["outputs"][-1])

                    correct_tr, solved_tr, true_correct_tr, true_solved_tr, false_correct_tr, false_solved_tr = compute_accuracy(
                        non_gen_values["target"], non_gen_values["outputs"][-1])
                    # correct_ge, solved_ge, true_correct_ge, true_solved_ge, false_correct_ge, false_solved_ge = compute_accuracy(
                    #     gen_values["target"], gen_values["outputs"][-1])

                    elapsed = time.time() - start_time
                    # losses_tr.append(non_gen_values["loss"])
                    corrects_tr.append(correct_tr)
                    solveds_tr.append(solved_tr)
                    # losses_ge.append(gen_values["loss"])
                    # corrects_ge.append(correct_ge)
                    # solveds_ge.append(solved_ge)
                    # logged_iterations.append(iteration)

                    if best_acc < correct_tr:
                        best_acc = correct_tr
                        sess_path = "weights/best_acc/"
                        if not os.path.isdir(sess_path):
                            os.makedirs(sess_path)
                        _ = saver.save(sess, os.path.join(sess_path, "dm.ckpt"))
                        with open(os.path.join(sess_path, "info.dat"), "w") as finf:
                            finf.write("i {}\n".format(iteration + 1))
                            finf.write("e {}\n".format(epoch + 1))
                            finf.write("acc {}".format(correct_tr))
                    if (iteration + 1) % 1000 == 0:
                        sess_path = "weights/last/"
                        if not os.path.isdir(sess_path):
                            os.makedirs(sess_path)
                        _ = saver.save(sess, os.path.join(sess_path, "dm.ckpt"))
                        with open(os.path.join(sess_path, "info.dat"), "w") as finf:
                            finf.write("i {}\n".format(iteration + 1))
                            finf.write("e {}".format(epoch + 1))
                    out_inf = "{:05d}, {:.1f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, \
{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, \
{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(
                              iteration, elapsed, train_values["loss"], correct_tr, solved_tr,
                              correct_tr, solved_tr,
                              true_correct_tr, true_solved_tr,
                              true_correct_tr, true_solved_tr,
                              false_correct_tr, false_solved_tr,
                              false_correct_tr, false_solved_tr,
                              tr_path_metrics["avg_batch_max_dist_diff"], tr_path_metrics["avg_batch_min_dist_diff"], tr_path_metrics["avg_batch_avg_dist_diff"],
                              tr_path_metrics["avg_batch_max_hops_diff"], tr_path_metrics["avg_batch_min_hops_diff"], tr_path_metrics["avg_batch_avg_hops_diff"],
                              tr_path_metrics["max_batch_unreachable_p"], tr_path_metrics["min_batch_unreachable_p"], tr_path_metrics["avg_batch_unreachable_p"],

                              tr_path_metrics["avg_batch_max_dist_diff"], tr_path_metrics["avg_batch_min_dist_diff"], tr_path_metrics["avg_batch_avg_dist_diff"],
                              tr_path_metrics["avg_batch_max_hops_diff"], tr_path_metrics["avg_batch_min_hops_diff"], tr_path_metrics["avg_batch_avg_hops_diff"],
                              tr_path_metrics["max_batch_unreachable_p"], tr_path_metrics["min_batch_unreachable_p"], tr_path_metrics["avg_batch_unreachable_p"])
                    f.write(out_inf)
                    print("Iter: {:05d}, Loss: {:.4f}, Acc: {:.4f}, Solved: {:.4f}, Avg Unreachable: {:.4f}".format(iteration, train_values["loss"], correct_tr, solved_tr, tr_path_metrics["avg_batch_unreachable_p"]))
            else:
                print("Skip i {}".format(iteration))
            iteration += 1
    f.close()
