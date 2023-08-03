# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn as nn


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class ModelWrapper(nn.Module):
    """"""

    def __init__(
        self,
        env=None,
        embedder=None,
        encoder=None,
        decoder=None,
        beam_type="search",
        beam_length_penalty=1,
        beam_size=1,
        beam_early_stopping=True,
        max_generated_output_len=200,
        beam_temperature=1.0,
        average_trajectories=False,
    ):
        super().__init__()

        self.env = env
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.beam_type = beam_type
        self.beam_early_stopping = beam_early_stopping
        self.max_generated_output_len = max_generated_output_len
        self.beam_size = beam_size
        self.beam_length_penalty = beam_length_penalty
        self.beam_temperature = beam_temperature
        self.device = next(self.embedder.parameters()).device
        self.average_trajectories = average_trajectories

    @torch.no_grad()
    def forward(self, input):
        """
        x: bags of sequences (B, T)
        """

        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        B, T = len(input), max([len(xi) for xi in input])
        outputs = []

        for chunk in chunks(np.arange(B), int(20000/T)):
            x, x_len = embedder([input[idx] for idx in chunk])
            encoded = encoder("fwd", x=x, lengths=x_len, causal=False).transpose(0, 1)
            bs = encoded.shape[0]

            ### Greedy solution.
            generations, _, two_hot_constant_masks = decoder.generate(
                encoded,
                x_len,
                sample_temperature=None,
                max_len=self.max_generated_output_len,
                env=self.env
            )
            
            generations = generations.unsqueeze(-1).view(generations.shape[0], bs, 1)
            generations = generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
            two_hot_constant_masks = two_hot_constant_masks.unsqueeze(-1).view(two_hot_constant_masks.shape[0], bs, 1)
            two_hot_constant_masks = two_hot_constant_masks.transpose(0, 1).transpose(1, 2).cpu().tolist()

            generations = [
                list(
                    filter(
                        lambda x: x is not None,
                        [
                            env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False, is_two_hot=mask[1:])
                            for hyp, mask in zip(generations[i], two_hot_constant_masks[i])
                        ],
                    )
                )
                for i in range(bs)
            ]

            if self.beam_type == "search":
                _, _, search_generations = decoder.generate_beam(
                    encoded,
                    x_len,
                    beam_size=self.beam_size,
                    length_penalty=self.beam_length_penalty,
                    max_len=self.max_generated_output_len,
                    early_stopping=self.beam_early_stopping,
                    average_across_batch=self.average_trajectories,
                    env=self.env
                )
                search_generations = [
                    sorted(
                        [hyp for hyp in search_generations[i].hyp],
                        key=lambda s: s[0],
                        reverse=True,
                    )
                    for i in range(bs)
                ]
                search_generations = [
                    list(
                        filter(
                            lambda x: x is not None,
                            [
                                env.idx_to_infix(
                                    hyp.cpu().tolist()[1:],
                                    is_float=False,
                                    str_array=False,
                                )
                                for (_, hyp) in search_generations[i]
                            ],
                        )
                    )
                    for i in range(bs)
                ]
                for i in range(bs):
                    #generations[i].extend(search_generations[i])
                    generations[i] = search_generations[i]

            elif self.beam_type == "sampling":
                num_samples = self.beam_size
                # if self.average_trajectories:
                #     encoded = encoded.mean(dim=0, keepdims=True)
                #     x_len = x_len[0].unsqueeze(0)
                #     bs = 1
                encoded = (
                    encoded.unsqueeze(1)
                    .expand((bs, num_samples) + encoded.shape[1:])
                    .contiguous()
                    .view((bs * num_samples,) + encoded.shape[1:])
                )
                x_len = x_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)
                sampling_generations, _, two_hot_constant_masks = decoder.generate(
                    encoded,
                    x_len,
                    sample_temperature=self.beam_temperature,
                    max_len=self.max_generated_output_len,
                    average_across_batch=self.average_trajectories,
                    env=self.env
                )
                sampling_generations = sampling_generations.unsqueeze(-1).view(
                    sampling_generations.shape[0], bs, num_samples
                )
                sampling_generations = (
                    sampling_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
                )
                two_hot_constant_masks = two_hot_constant_masks.unsqueeze(-1).view(
                    two_hot_constant_masks.shape[0], bs, num_samples
                )
                two_hot_constant_masks = (
                    two_hot_constant_masks.transpose(0, 1).transpose(1, 2).cpu().tolist()
                )
                
                sampling_generations = [
                    list(
                        filter(
                            lambda x: x is not None,
                            [
                                env.idx_to_infix(
                                    hyp[1:-1], is_float=False, str_array=False, is_two_hot=mask[1:]
                                )
                                for hyp, mask in zip(sampling_generations[i], two_hot_constant_masks[i])
                            ],
                        )
                    )
                    for i in range(bs)
                ]
                
                for i in range(bs):
                    #generations[i].extend(sampling_generations[i])
                    generations[i] = sampling_generations[i]
            else:
                raise NotImplementedError
            outputs.extend(generations)
        return outputs
