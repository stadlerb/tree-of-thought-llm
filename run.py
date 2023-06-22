import argparse
import asyncio
import datetime
import itertools
import json
import logging
import os
import statistics
import typing
from functools import partial
from typing import TypeVar, Any, List

import numpy as np

import models
from models import gpt_usage
from tasks import get_task

global gpt
global agpt

log = logging.getLogger("tree_of_thought_run")


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
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
    value_outputs = await agpt(value_prompt, n=n_evaluate_sample, stop=None)
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
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


def solve(args, task, idx):
    log.debug(gpt)
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
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}


def run(args):
    task = get_task(args.task, args.task_file_path)
    logs, cnt_avg, cnt_any = [], 0, 0
    global gpt, agpt
    gpt = partial(models.gpt, model=args.backend, temperature=args.temperature)
    agpt = partial(models.agpt, model=args.backend, temperature=args.temperature)
    if args.naive_run:
        file = f'logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    run_start_time = datetime.datetime.now()
    log.info(f"run start time: {run_start_time}")
    task_times = []
    for i in range(args.task_start_index, args.task_end_index):
        task_start_time = datetime.datetime.now()
        log.info(f"task {i} -- start time: {task_start_time}")
        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i)
        else:
            ys, info = solve(args, task, i)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)

        # log main metric
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)

        task_end_time = datetime.datetime.now()
        task_duration = task_end_time - task_start_time
        log.info(
            f"task {i} --"
            f" sum(accs): {sum(accs)},"
            f" cnt_avg: {cnt_avg},"
            f" cnt_any: {cnt_any},"
            f" end time {task_end_time},"
            f" duration: {task_duration}"
        )
        task_times.append(task_duration)

    run_end_time = datetime.datetime.now()
    run_duration = run_end_time - run_start_time
    log.info(f"run end time: {run_end_time}, run duration: {run_duration}")

    task_times_sec = [t.total_seconds() for t in task_times]

    log.info(f"task times: {task_times_sec}")
    log.info(
        f"task times"
        f" mean: {statistics.mean(task_times_sec)},"
        f" stdev: {statistics.stdev(task_times_sec)},"
        f" min: {min(task_times_sec)},"
        f" max: {max(task_times_sec)}"
    )

    n = args.task_end_index - args.task_start_index
    log.info(f'{cnt_avg / n} {cnt_any / n}')
    log.info(f'usage_so_far {gpt_usage(args.backend)}')


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_file_path', type=str, required=True)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str,
                      choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'value_async', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'])
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args.add_argument('--log_level', type=str, choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                      default='INFO')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    log.info(args)
    models.init_api()
    run(args)
