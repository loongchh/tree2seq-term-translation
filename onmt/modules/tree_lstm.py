"""treelstm.py - TreeLSTM RNN models

Written by Riddhiman Dasgupta (https://github.com/dasguptar/treelstm.pytorch)
Rewritten in 2018 by Long-Huei Chen <longhuei@g.ecc.u-tokyo.ac.jp>

To the extent possible under law, the author(s) have dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along with
this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
"""

import torch
import torch.nn as nn


class TreeLSTMBase(nn.Module):
    @staticmethod
    def extract_tree(parse):
        """
        Args:
            line: A list of tokens, where each token consists of a word,
                optionally followed by u"￨"-delimited features.
        Returns:
            A sequence of words, a sequence of features, and num of features.
        """
        if parse is None:
            return [], [], -1

        parents = parse.cpu().numpy()
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent

        return root


class ChildSumTreeLSTM(TreeLSTMBase):
    def __init__(self, rnn_type, input_size, hidden_size, bias=True):
        super(ChildSumTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ioux = nn.Linear(input_size, 3 * self.hidden_size, bias=bias)
        self.iouh = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=bias)
        self.fx = nn.Linear(input_size, self.hidden_size, bias=bias)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

    def forward(self, parses, embeds):
        states_c, states_h = zip(*[
            self.tree_forward(
                ChildSumTreeLSTM.extract_tree(parses[:, j]), embeds[:, j, :])
            for j in range(parses.size(1))
        ])
        states_c = torch.cat(states_c, dim=1)
        states_h = torch.cat(states_h, dim=1)
        return (states_c, states_h)

    def tree_forward(self, tree, embed):
        for idx in range(tree.num_children):
            self.tree_forward(tree.children[idx], embed)

        if tree.num_children > 0:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c = torch.cat(child_c, dim=0)
            child_h = torch.cat(child_h, dim=0)
        else:  # leaf nodes
            child_c = embed[0].detach().new_zeros(
                1, self.hidden_size).requires_grad_()
            child_h = embed[0].detach().new_zeros(
                1, self.hidden_size).requires_grad_()

        tree.state = self.node_forward(embed[tree.idx], child_c, child_h)
        return tree.state

    def node_forward(self, embeds, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        iou = self.ioux(embeds) + self.iouh(child_h_sum)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = self.fh(child_h) + self.fx(embeds).repeat(len(child_h), 1)
        fc = torch.mul(torch.sigmoid(f), child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h


class BinaryTreeLSTM(TreeLSTMBase):
    def __init__(self, rnn_type, hidden_size, bias=False):
        super(BinaryTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.iou0 = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=bias)
        self.iou1 = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=bias)
        self.f0 = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.f1 = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

    def forward(self, child_c, child_h):
        iou = self.iou0(child_h[0]) + self.iou1(child_h[1])
        i, o, u = torch.chunk(iou, 3, dim=2)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.cat((self.f0(child_h[0]), self.f1(child_h[1])), dim=0)
        fc = torch.mul(torch.sigmoid(f), torch.cat(child_c, dim=0)).sum(
            dim=0, keepdim=True)
        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))
        return c, h

class Tree():
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.state = None
        self.idx = None

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def __len__(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth
