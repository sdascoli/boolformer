# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
#from .generators import operators_conv, Node
from .recurrence import RecurrenceEnvironment

logger = getLogger()


ENVS = {
    'recurrence': RecurrenceEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)

    # tasks
    if isinstance(params.tasks,str):
        tasks = [x for x in params.tasks.split(',') if len(x) > 0]
    else:
        tasks = params.tasks
    assert len(tasks) == len(set(tasks)) > 0
    assert all(task in env.TRAINING_TASKS for task in tasks)
    params.tasks = tasks
    logger.info(f'Training tasks: {", ".join(tasks)}')

    return env
