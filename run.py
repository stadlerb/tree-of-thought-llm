import argparse
import datetime
import json
import logging
import os
import statistics

from tot.methods.bfs import solve, naive_solve
from tot.models import gpt_usage, init_api
from tot.tasks import get_task

log = logging.getLogger("tree_of_thought_run")


def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
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
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str,
                      choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'value_async', 'vote'])

    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args.add_argument('--max_open_requests', type=int, default=20)

    args.add_argument('--log_level', type=str, choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                      default='INFO')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    log.info(args)
    init_api()
    run(args)
