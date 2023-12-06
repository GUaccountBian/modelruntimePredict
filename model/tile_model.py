import abc

import tensorflow as tf
import tensorflow_gnn as tfgnn

import tpugraphsv1_implicit_py as implicit

class _ConfigFeatureJoiner(abc.ABC):
    """Defines interface for joining config features with op nodes.
    The implementations join features pre- or post-GNN, respectively, named as
    `_EarlyJoin` and `_LateJoin`.
    """

    @abc.abstractmethod
    def get_op_node_features(
        self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        """Should return feature matrix (or tensor) of op-nodes."""
        raise NotImplementedError()

    def get_penultimate_output(
        self, pooled: tf.Tensor, unused_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """Must return tensor with shape `[batch_size, num_configs, hidden_dim]`."""
        return pooled


def _mlp(dims, hidden_activation, l2reg=1e-4, use_bias=True):
    """Helper function for multi-layer perceptron (MLP)."""
    layers = []
    for i, dim in enumerate(dims):
        if i > 0:
            layers.append(tf.keras.layers.Activation(hidden_activation))
        layers.append(tf.keras.layers.Dense(
            dim, kernel_regularizer=tf.keras.regularizers.l2(l2reg),
            use_bias=use_bias))
    return tf.keras.Sequential(layers)


class _OpEmbedding(tf.keras.Model):
    """Embeds GraphTensor.node_sets['op']['op'] nodes into feature 'op_e'."""

    def __init__(self, num_ops: int, embed_d: int, l2reg: float = 1e-4):
        super().__init__()
        self.embedding_layer = tf.keras.layers.Embedding(
            num_ops, embed_d, activity_regularizer=tf.keras.regularizers.l2(l2reg))

    def call(self, graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
        op_features = dict(graph.node_sets['op'].features)
        op_features['op_e'] = self.embedding_layer(
            tf.cast(graph.node_sets['op']['op'], tf.int32))
        return graph.replace_features(node_sets={'op': op_features})

class _ResGCN(tf.keras.Model, _ConfigFeatureJoiner):
    """Implements GCN backbone with residual connections."""

    def __init__(self, num_ops: int, num_gnns: int = 3,
                 mlp_layers: int = 2, hidden_activation: str = 'leaky_relu', 
                 hidden_dim: int = 64, op_embed_dim: int = 32, 
                 directed: bool = False, reduction: str = 'sum'):

        super().__init__()

        # Assign parameters to instance variables
        self._num_ops = num_ops
        self._directed = directed
        self._reduction = reduction
        self._activation_fn = getattr(tf.nn, hidden_activation)

        # Initialize operation embedding
        self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)

        # Initialize GCN layers
        self._gc_layers = []
        for _ in range(num_gnns):
            if directed:
                configs_mlps = (
                    _mlp([hidden_dim] * mlp_layers, self._activation_fn),
                    _mlp([hidden_dim] * mlp_layers, self._activation_fn),
                    _mlp([hidden_dim] * mlp_layers, self._activation_fn)
                )
            else:
                configs_mlps = (_mlp([hidden_dim] * mlp_layers, self._activation_fn),)

            self._gc_layers.append(tuple(configs_mlps))

        # Initialize pre and post networks
        self._prenet = _mlp([hidden_dim, hidden_dim], self._activation_fn)
        self._postnet = _mlp([hidden_dim, 1], self._activation_fn)

    def call(self, graph: tfgnn.GraphTensor):
        """Perform a forward pass."""
        return self.forward(graph)

    def forward(self, graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
        """Define the forward propagation."""

        graph = self._op_embedding(graph)
        x = self.get_op_node_features(graph)

        am = implicit.AdjacencyMultiplier(graph, 'feed') # op -> op
        am = am.add_eye().normalize_right()

        x = self._prenet(x)

        for gc_layer in self._gc_layers:
            y = self._activation_fn(x)
            forward_layer, reverse_layer, self_layer = gc_layer

            if self._directed:
                y = (forward_layer(am @ y) +
                     reverse_layer(am.transpose() @ y) +
                     self_layer(y))
            else:
                # This means the current node feature y is updated based on:
                # The message from its neighbors (am @ y).
                # The message from itself to its neighbors (am.transpose() @ y).
                # Its own previous feature value y.

                # y = forward_layer((am @ y) + (am.transpose() @ y) + y)
                y = forward_layer((am + am.transpose()).add_eye() @ y)

            # Residual connection
            x += y

        x = self._activation_fn(x)

        # https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/tfgnn/pool_nodes_to_context.md
        pooled = tfgnn.pool_nodes_to_context(graph, 'op', self._reduction, feature_value=x)

        pooled = self.get_penultimate_output(pooled, graph)

        return tf.squeeze(self._postnet(pooled), -1)


class _LateJoin(_ConfigFeatureJoiner):
    """Joins module configuration features after applying GNN backbone."""

    def get_op_node_features(self, graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
        """Retrieve operation node features."""
        return tf.concat([graph.node_sets['op']['op_e'], graph.node_sets['op']['feats']], axis=-1)

    def get_penultimate_output(self, pooled: tf.Tensor,
                               graph: tfgnn.GraphTensor) -> tf.Tensor:
        """Work with pooled features."""

        config_feats = graph.node_sets['config']['feats']
        batch_size = graph.node_sets['config'].sizes.shape[0]

        config_feats = tf.reshape(config_feats, [batch_size, -1, config_feats.shape[-1]])

        # Stack the pooled features for each configuration
        pooled = tf.stack([pooled] * config_feats.shape[1], 1)
        pooled = tf.concat([pooled, config_feats], -1)

        return pooled


class LateJoinResGCN(_LateJoin, _ResGCN):

    def __init__(self, num_ops: int, num_gnns: int = 3,
                 mlp_layers: int = 2, hidden_activation: str = 'leaky_relu',
                 hidden_dim: int = 64, op_embed_dim: int = 32,
                 directed: bool = False, reduction: str = 'sum'):
        # Initialize the _ResGCN superclass with the provided parameters
        _ResGCN.__init__(self, num_ops, num_gnns, mlp_layers,
                         hidden_activation, hidden_dim, op_embed_dim, directed, reduction)
