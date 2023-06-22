import asyncio
import datetime
import itertools
import logging
import typing
from functools import partial
from typing import TypeVar, Any, List

import numpy as np

import tot.models

global model_call
global model_call_async

log = logging.getLogger(__name__)


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = model_call(value_prompt, n=n_evaluate_sample, stop=None)
    value: float = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=True) -> List[float]:
    values: List[float] = []
    local_value_cache = {}
    n = len(ys)
    for i, y in enumerate(ys):  # each partial output
        log.debug(f"get_value {i} / {n} -- {datetime.datetime.now()}")
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0.0
            log.debug("--> duplicate candidate, value set to 0")
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


async def aget_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = await model_call_async(value_prompt, n=n_evaluate_sample, stop=None)
    value: float = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


T = TypeVar("T")


async def aconst(x: T) -> T:
    return x


async def aget_values(task, x, ys, n_evaluate_sample, cache_value=True) -> List[float]:
    a_values: List[typing.Coroutine[Any, Any, float]] = []
    local_value_cache = {}
    n = len(ys)
    for i, y in enumerate(ys):  # each partial output
        log.debug(f"get_value {i} / {n} -- {datetime.datetime.now()}")
        if y in local_value_cache:  # avoid duplicate candidates
            a_value = aconst(0.0)
            log.debug("--> duplicate candidate, value set to 0")
        else:
            a_value = aget_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = a_value
        a_values.append(a_value)
    results: List[float | Exception] = await asyncio.gather(*a_values, return_exceptions=True)
    return [result if not isinstance(result, Exception) else 0.0 for result in results]


def get_values_async(task, x, ys, n_evaluate_sample, cache_value=True):
    return asyncio.run(aget_values(task, x, ys, n_evaluate_sample, cache_value=cache_value))


def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = model_call(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = model_call(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = model_call(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


def solve(args, task, idx):
    init_globals(args)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        log.debug(f"Step {step} generation started -- {datetime.datetime.now()}")
        if args.method_generate == 'sample':
            new_ys = [
                get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step])
                for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        else:
            raise ValueError(f"Unknown generation method {args.method_generate}")
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        log.debug(f"Step {step} evaluation started -- {datetime.datetime.now()}")
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value_async':
            values = get_values_async(task, x, new_ys, args.n_evaluate_sample)
        else:
            raise ValueError(f"Unknown evaluation method {args.method_evaluate}")

        # selection
        log.debug(f"Step {step} selection started -- {datetime.datetime.now()}")
        if args.method_select == 'sample':
            # add small value to avoid NaN if all values are 0, or is it better to fail in that case?
            v = np.array(values)  # + 0.001
            ps = v / v.sum()
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        else:
            raise ValueError(f"Unknown selection method {args.method_select}")
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        log.debug(f"Step {step} done -- {datetime.datetime.now()}")

        # log
        if log.isEnabledFor(logging.DEBUG):
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            log.debug(
                f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        infos.append(
            {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys

    log.debug(ys)
    return ys, {'steps': infos}


def naive_solve(args, task, idx):
    init_globals(args)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}


def init_globals(args):
    global model_call, model_call_async
    model_call = partial(tot.models.gpt, model=args.backend, temperature=args.temperature)
    model_call_async = partial(tot.models.agpt, model=args.backend, temperature=args.temperature)
    log.debug(model_call)
