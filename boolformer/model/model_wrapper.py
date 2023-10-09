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


class Boolformer(nn.Module):
    """"""

    def __init__(
        self,
        env=None,
        embedder=None,
        encoder=None,
        decoder=None,
    ):
        super().__init__()

        self.env = env
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.device = next(self.embedder.parameters()).device

    @torch.no_grad()
    def fit(self,
                inputs, outputs,
                verbose=False, 
                store_attention=False, 
                sort_by='error', 
                complexity_metric='n_ops_arbitrary_fan_in',
                beam_size=10,
                beam_type='search',
                beam_temperature=.1):
        
        self.estimators = []
        self.errors = []
                
        encoder, decoder, embedder = self.encoder, self.decoder, self.embedder
        env = self.env
        
        if hasattr(outputs, 'shape'):
            if len(outputs.shape)==1:
                inputs, outputs = [inputs], [outputs]
        elif not isinstance(inputs, list) or not isinstance(outputs, list):
            inputs, outputs = [inputs], [outputs]

        for input in inputs:
            if input.shape[1] > env.params.max_active_vars + env.params.max_inactive_vars:
                raise ValueError('This model can take at most {} input variables'.format(env.params.max_active_vars + env.params.max_inactive_vars))
            
        x, x_len = embedder.batch(inputs, outputs)
        x = embedder(x)

        if not env.params.cpu:
            x, x_len = x.cuda(), x_len.cuda()
        encoded = encoder("fwd", x=x, lengths=x_len, causal=False, store_outputs=store_attention).transpose(0,1)
        bs = encoded.shape[0]

        if beam_type == 'search':
            _, _, gens = decoder.generate_beam(
                        encoded,
                        x_len,
                        beam_size=beam_size,
                        length_penalty=env.params.beam_length_penalty,
                        early_stopping=env.params.beam_early_stopping,
                        max_len = env.params.max_output_len)
            gens = [sorted([hyp for (_, hyp) in gens[i].hyp], key=lambda s: s[0], reverse=True) for i in range(bs)]
        elif beam_type == 'sampling':
            num_samples = beam_size
            encoded = encoded.unsqueeze(1).expand((bs, num_samples) + encoded.shape[1:]).contiguous().view((bs * num_samples,) + encoded.shape[1:])
            x_len = x_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)
            gens, _, _ = decoder.generate(
                        encoded,
                        x_len,
                        sample_temperature=beam_temperature)
            gens = gens.unsqueeze(-1).view(gens.shape[0], bs, num_samples).transpose(0, 1).transpose(1, 2).cpu()#.tolist()   
        else:
            raise
        
        best_trees, best_errors, best_complexities = [], [], []
        
        for problem_idx, gen in enumerate(gens):

            ratio = sum(outputs[problem_idx])/len(outputs[problem_idx])
            if ratio in [0,1]:
                if ratio == 1:
                    best_trees.append(env.output_encoder.decode(['True']))
                else:
                    best_trees.append(env.output_encoder.decode(['False']))
                best_errors.append(0)
                best_complexities.append(0)
                continue

            pred_trees, errors, complexities = [], [], []
            for hyp in gen:
                tokens = [env.output_id2word[wid] for wid in hyp.tolist()[1:]]
                pred_tree = env.output_encoder.decode(tokens)
                if pred_tree is None: continue
                #pred_tree = env.simplifier.simplify_tree(pred_tree, env.params.simplify_form)
                #pred_tree.to_arbitrary_fan_in()
                pred_tree.simplify()
                pred_trees.append(pred_tree)
                pred = pred_tree(inputs[0]).flatten()
                true = outputs[problem_idx]
                error = 1.-sum(pred==true)/len(pred)
                errors.append(error)
                if complexity_metric == 'n_ops_arbitrary_fan_in':
                    complexity = pred_tree.get_n_ops_arbitrary_fan_in()
                elif complexity_metric == 'n_ops':
                    complexity = pred_tree.get_n_ops()
                else:
                    complexity = len(pred_tree.get_variables())
                complexities.append(complexity)

            # sort by error and complexity
            if sort_by=='error':
                idx = sorted(range(len(errors)), key=lambda k: (errors[k], complexities[k]))
            else:
                idx = sorted(range(len(errors)), key=lambda k: (complexities[k], errors[k]))

            pred_trees = [pred_trees[i] for i in idx]
            errors = [errors[i] for i in idx]
            complexities = [complexities[i] for i in idx]
            if verbose:
                print('Errors      ', '   '.join(['{:.3f}'.format(error) for error in errors]))
                print('Complexities', '   '.join(['{:>5d}'.format(complexity) for complexity in complexities]))
            if pred_trees:
                best_trees.append(pred_trees[0])
                self.estimators.append(pred_trees)
                self.errors.append(errors)
                best_errors.append(errors[0])
                best_complexities.append(complexities[0])
            else:
                self.estimators.append(None)
                best_trees.append(None)
                best_errors.append(None)
                best_complexities.append(None)

        #self.prune_estimators()
        
        return best_trees, best_errors, best_complexities
    
    def prune_estimators(self):
        # remove duplicates (exactly) and bad estimators
        for i in range(len(self.estimators)):
            estimators, errors = self.estimators[i], self.errors[i]
            if estimators is None: continue
            new_estimators, new_errors = [], []
            for j in range(len(estimators)):
                if errors[j] > 2*errors[0]: break
                #if j==0 or errors[j]!=errors[j-1]:
                new_estimators.append(estimators[j])
                new_errors.append(errors[j])
            self.estimators[i], self.errors[i] = new_estimators, new_errors
    
    def predict(self, inputs, n_estimators=1):

        assert hasattr(self, 'estimators'), 'No estimator found. Please run fit() first'

        preds = []
        for estimators, errors in zip(self.estimators, self.errors):
            pred = []
            for estimator in estimators[:n_estimators]:
                pred.append(estimator.val(inputs))
            # get majority vote
            pred = np.array(pred)
            pred = np.mean(pred, axis=0)
            pred = np.round(pred).astype(int)
            preds.append(pred)

        preds = np.array(preds)
        return preds
