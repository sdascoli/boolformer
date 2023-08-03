# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from ..utils import to_cuda
import torch.nn.functional as F

MultiDimensionalFloat = List[float]
XYPair = Tuple[MultiDimensionalFloat, MultiDimensionalFloat]
Sequence = List[XYPair]

    
class Embedder(ABC, nn.Module):
    """
    Base class for embedders, transforms a sequence of pairs into a sequence of embeddings.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_length_after_batching(self, sequences: List[Sequence]) -> List[int]:
        pass

class LinearPointEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.input_dim = params.emb_emb_dim
        self.output_dim = params.enc_emb_dim
        self.embeddings = Embedding(
            len(self.env.input_id2word),
            self.input_dim,
            padding_idx=self.env.input_word2id["PAD"],
        )
        self.total_dimension = 1 + self.params.max_vars

        self.activation_fn = getattr(F, params.activation)
        size = self.total_dimension*self.input_dim
        hidden_size = size * self.params.emb_expansion_factor
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(size, hidden_size))
        for i in range(self.params.n_emb_layers-1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, self.output_dim)
        self.max_seq_len = self.params.max_points

    def batch(self, inputs, outputs):
        sequences, sequences_len = self.encode(inputs, outputs)
        sequences, sequences_len = to_cuda(sequences, sequences_len, use_cpu=self.params.cpu)
        return sequences, sequences_len

    def forward(self, sequences) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences_embeddings = self.embed(sequences)
        sequences_embeddings = self.compress(sequences_embeddings)
        return sequences_embeddings

    def compress(
        self, sequences_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(float_descriptor_length, B, d) tensors
        Returns: (N_max, B, d)
        """
        max_len, bs, float_descriptor_length, dim = sequences_embeddings.size()
        sequences_embeddings = sequences_embeddings.view(max_len, bs, -1)
        for layer in self.hidden_layers: sequences_embeddings = self.activation_fn(layer(sequences_embeddings))
        sequences_embeddings = self.fc(sequences_embeddings)
        return sequences_embeddings
    
    def encode(self, inputs, outputs):
        res = []
        for (x_arr, y_arr) in zip(inputs, outputs):
            seq_toks = []
            for (x,y) in zip(x_arr, y_arr):
                x_toks = self.env.input_encoder.encode(x)
                y_toks = self.env.input_encoder.encode(y)
                x_toks = [*x_toks, *["PAD" for _ in range(self.params.max_vars - len(x_toks))]]
                toks = [*x_toks, *y_toks]
                seq_toks.append([self.env.input_word2id[tok] for tok in toks])
            res.append(torch.LongTensor(seq_toks))
        data = res
        data, data_len = self.batch_input_sequences(data)
        return data, data_len

    def batch_input_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n, dim) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) for s in sequences])
        bs, slen = len(lengths), max(lengths)
        dim = sequences[0].shape[-1]

        sent = torch.LongTensor(slen, bs, dim).fill_(self.env.pad_index)
        for i, seq in enumerate(sequences):
            sent[0 : len(seq), i, :] = seq

        return sent, torch.LongTensor(lengths)

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths
    
class TwoHotEmbedder(nn.EmbeddingBag):
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            mode="sum"
        )
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # inp has shape = (seq_len, batch size)
        shape_output = list(idx.shape)
        shape_output.append(self.embedding_dim)
        # for the two-hot representation we need to specify:
        # - the neighboring bins that support the distribution (=support_idcs)
        # - the probability mass that is assigned to each of the bins (=support_weights)
        support_idcs = torch.stack((idx.reshape(-1), idx.reshape(-1))).T # shape=(seq_len * batch size, 2)
        # the right bins contains the decimal value, the left bin contains 1-decimal value
        support_weights = support_idcs % 1
        support_weights[:, 0] = 1 - support_weights[:, 1]
        support_weights = support_weights.to(torch.float32) # TODO: which dtype to use? needs to be consistent with self.weight
        support_idcs[:, 0] = torch.floor(support_idcs[:, 0])
        support_idcs[:, 1] = torch.ceil(support_idcs[:, 1])
        support_idcs= support_idcs.to(torch.int64)
        return super().forward(input=support_idcs, per_sample_weights=support_weights).reshape(shape_output)
