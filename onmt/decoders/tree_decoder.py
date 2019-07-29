"""tree_decoder.py - Sequential or Tree-generator decoder models

Written by OpenNMT (https://github.com/OpenNMT/OpenNMT-py)
Rewritten in 2018 by Long-Huei Chen <longhuei@g.ecc.u-tokyo.ac.jp>

To the extent possible under law, the author(s) have dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along with
this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
"""

from __future__ import division
import torch
import torch.nn as nn

from onmt.decoders.decoder import InputFeedRNNDecoder, RNNDecoderState
from onmt.utils.rnn_factory import rnn_factory
from onmt.utils.misc import aeq
from onmt.modules.tree_lstm import BinaryTreeLSTM


class Tree2SeqDecoder(InputFeedRNNDecoder):
    """
    Standard fully batched RNN decoder without attention.
    See :obj:`RNNDecoderBase` for options.

    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`
    """

    def __init__(self,
                 rnn_type,
                 bidirectional_encoder,
                 num_layers,
                 hidden_size,
                 attn_type="general",
                 attn_func="softmax",
                 coverage_attn=False,
                 context_gate=None,
                 copy_attn=False,
                 dropout=0.0,
                 embeddings=None,
                 reuse_copy_attn=False,
                 tree_combine_hidden=False):
        super(Tree2SeqDecoder, self).__init__(
            rnn_type, bidirectional_encoder, num_layers, hidden_size,
            attn_type, attn_func, coverage_attn, context_gate, copy_attn,
            dropout, embeddings, reuse_copy_attn)
        if tree_combine_hidden:
            self.combine = BinaryTreeLSTM(rnn_type, hidden_size, bias=False)
        else:
            self.linear = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.combine = lambda c, h: (sum(c), torch.tanh(self.linear(torch.cat(h, dim=2))))

    def init_decoder_state(self, src, memory_bank, encoder_final):
        """ Init decoder state with last state of the encoder """
        rnn_final, tree_final = encoder_final
        child_c = (rnn_final[0], tree_final[0])
        child_h = (rnn_final[1], tree_final[1])
        encoder_final = self.combine(child_c, child_h)

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat(
                    [hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]],
                    2)
            return hidden

        return RNNDecoderState(
            self.hidden_size,
            tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final]))
