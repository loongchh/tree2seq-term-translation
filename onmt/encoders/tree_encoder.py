"""tree_encoder.py - Tree-LSTM encoder model

Written by Riddhiman Dasgupta (https://github.com/dasguptar/treelstm.pytorch)
Written by OpenNMT (https://github.com/OpenNMT/OpenNMT-py)
Rewritten in 2018 by Long-Huei Chen <longhuei@g.ecc.u-tokyo.ac.jp>

To the extent possible under law, the author(s) have dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along with
this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory
from onmt.modules.tree_lstm import ChildSumTreeLSTM


class TreeEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self,
                 rnn_type,
                 bidirectional,
                 num_layers,
                 hidden_size,
                 dropout=0.0,
                 embeddings=None,
                 use_bridge=False):
        super(TreeEncoder, self).__init__()
        assert embeddings is not None
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        self.hidden_size = hidden_size // num_directions
        self.input_size = embeddings.embedding_size

        self.embeddings = embeddings
        self.rnn, self.no_pack_padded_seq = rnn_factory(
            rnn_type,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)
        self.treelstm = ChildSumTreeLSTM(
            rnn_type, self.input_size, self.hidden_size, bias=True)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type, self.hidden_size, num_layers)

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        # Only support num_feature == 0 for now
        src_data, src_parse = src
        assert src_data.size(2) == 1
        embeds = self.embeddings(src_data)

        # Tree-LSTM encoder
        if src_parse is not None:
            tree_final = self.treelstm(src_parse, embeds)
            tree_final = tuple(
                s.view(1, -1, self.hidden_size) for s in tree_final)

        # Sequential encoder w/ attention
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            embeds = pack_padded_sequence(embeds, lengths.view(-1).tolist())

        memory_bank, encoder_final = self.rnn(embeds)
        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = pad_packed_sequence(memory_bank)[0]
        if src_parse is not None:
            encoder_final = (
                encoder_final,
                tree_final,
            )

        return encoder_final, memory_bank, lengths
