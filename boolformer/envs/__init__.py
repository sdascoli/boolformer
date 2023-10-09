from logging import getLogger
#from .generators import operators_conv, Node
from .environment import BooleanEnvironment

logger = getLogger()


def build_env(params):
    """
    Build environment.
    """
    env = BooleanEnvironment(params)

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
