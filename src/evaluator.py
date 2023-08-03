# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
from numpy.core.fromnumeric import argsort
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
import pandas as pd

from .utils import to_cuda

logger = getLogger()


def idx_to_infix(env, idx, input=True, str_array=True):
    """
    Convert an indexed prefix expression to SymPy.
    """
    if input:
        prefix = [env.input_id2word[wid] for wid in idx]
        infix = env.input_to_infix(prefix,str_array)
    else:
        prefix = [env.output_id2word[wid] for wid in idx]
        infix = env.output_to_infix(prefix,str_array)
    return infix

def idx_to_tree(env, idx):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.output_id2word[wid] for wid in idx]
    tree = env.output_encoder.decode(prefix)
    return tree

def calculate_error(src, hyp_tree, tgt_tree):
    if hyp_tree is None:
        return .5
    pred = hyp_tree.val(src).flatten()
    true = tgt_tree.val(src).flatten()
    error = 1.-sum(pred==true)/len(pred)
    return error       

class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env



    def run_all_evals(self, data_types=['valid1'], baselines=[]):
        """
        Run all evaluations.

        """
        params = self.params
        scores = OrderedDict({"epoch": self.trainer.epoch})

        # save statistics about generated data
        if params.export_data:
            scores["total"] = self.trainer.total_samples
            return scores

        with torch.no_grad():
            for data_type in data_types:
                for task in params.tasks:
                    if params.beam_eval:
                        self.enc_dec_step_beam(data_type, task, scores)
                    else:
                        raise NotImplementedError
                    for baseline in baselines:
                        self.classic_method_evaluate(baseline, data_type, task, scores)
        return scores

    def set_env_copies(self, data_types):
        for data_type in data_types:
            setattr(self, "{}_env".format(data_type), deepcopy(self.env))

    def enc_dec_step_beam(self, data_type, task, scores):
        """
        Encoding / decoding step with beam generation 
        """

        n_infos_prior = 200
        params = self.params

        max_beam_length = self.params.max_output_len
        embedder = (
            self.modules["embedder"].module
            if params.multi_gpu
            else self.modules["embedder"]
        )
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        embedder.eval()
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ["recurrence"]

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(
                params.dump_path, f"eval.beam.{data_type}.{task}.{scores['epoch']}"
            )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res["hyps"]])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\n"
                s += f"src={res['src']}\ntgt={res['tgt_tree']}\n"
                for hyp_tree, output, valid in res["hyps"]:
                    validity = 'Valid' if valid else 'Invalid'
                    s += f"{validity} {output} {hyp_tree}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # iterator

        iterator = self.env.create_test_iterator(
                data_type,
                task,
                data_path=self.trainer.data_path,
                batch_size=params.batch_size_eval,
                params=params,
                size=params.eval_size,
                input_length_modulo=params.eval_input_length_modulo,
                test_env_seed=self.params.test_env_seed
            )
            
        eval_size = len(iterator.dataset)

        # stats
        xe_loss = 0
        n_perfect_match = 0
        n_total=0
        batch_results = defaultdict(list)

        for samples in iterator:

            x1, len1, x2, len2, infos = samples['data'], samples['data_len'], samples['trees'], samples['trees_len'], samples['infos']
            val_inputs, val_outputs = samples['val_inputs'], samples['val_outputs']
            for k,v in infos.items():
                batch_results["info_"+k].extend(v.tolist())

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y, use_cpu=params.cpu)
            #x1, len1 = embedder.batch(inputs, outputs)
            x1 = embedder(x1)
            bs=len1.shape[0]
            n_total+=bs

            # forward
            encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1,
            )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y
            )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            perfect_match = (t.sum(0) == len2 - 1).cpu().long() ##TODO: check here
            valid = perfect_match.clone()
            n_perfect_match += perfect_match.sum().item()

            batch_results['perfect_match'].extend(perfect_match.tolist())

            # stats
            xe_loss += loss.item() * len(y)

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({valid.sum().item()}/{eval_size}) Found "
                f"{bs - len(invalid_idx)}/{bs} valid top-1 predictions. "
                f"Generating solutions ..."
            )

            # generate
            _, _, generations = decoder.generate_beam(
                encoded.transpose(0, 1),
                len1,
                beam_size=params.beam_size,
                length_penalty=params.beam_length_penalty,
                early_stopping=params.beam_early_stopping,
                max_len=max_beam_length,
            )

            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            for i in range(len(generations)):
                lowest_error, best_hyp = .5, None
                tgt = x2[1 : len2[i] - 1, i].tolist()
                tgt = idx_to_tree(self.env, tgt)
                for j, (score, hyp) in enumerate(sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)):

                    hyp = hyp[1:].tolist()
                    src = samples['inputs'][i]

                    hyp = idx_to_tree(self.env, hyp)
                    error = calculate_error(src, hyp, tgt)
                    #test_points = np.random.choice([True,False], size=(10000, src.shape[-1]))
                    test_error = calculate_error(val_inputs[i], hyp, tgt)
                    if error < lowest_error:
                        best_hyp, lowest_error, test_error = hyp, error, test_error
                if best_hyp:
                    batch_results['predicted_tree'].append(best_hyp.infix())
                    batch_results['complexity'].append(len(best_hyp.prefix().split(',')))
                else:
                    batch_results['predicted_tree'].append(None)
                    batch_results['complexity'].append(np.nan)
                batch_results['tree'].append(tgt.infix())
                batch_results['error'].append(lowest_error)
                batch_results['test_error'].append(test_error)
                batch_results['acc'].append(test_error==0 and error==0)

        df = pd.DataFrame.from_dict(batch_results)
        save_file = os.path.join(params.dump_path, f"evals.csv")
        df.to_csv(save_file, index=False)
        logger.info("Saved {} equations to {}".format(len(df), save_file))

        info_columns = [x for x in list(df.columns) if x.startswith("info_")]
        df = df.drop(columns=["predicted_tree"])
        if "tree" in df: df = df.drop(columns=["tree"])
        for column in list(df.columns):
            if column not in info_columns + ['epoch']:
                scores[column] = df[column].mean()
                        
        for ablation in info_columns:
            for val, df_ablation in df.groupby(ablation):
                avg_scores_ablation = df_ablation.mean()
                for k, v in avg_scores_ablation.items():
                    if k not in info_columns:
                        scores[k + "_{}_{}".format(ablation, val)] = v