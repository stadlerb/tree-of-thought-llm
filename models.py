import logging
import os
import random

import backoff
import openai

log = logging.getLogger(__name__)
completion_tokens = prompt_tokens = 0


def init_api():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key != "":
        openai.api_key = api_key
    else:
        log.warning("OPENAI_API_KEY is not set")

    api_base = os.getenv("OPENAI_API_BASE", "")
    if api_base != "":
        log.info(f"OPENAI_API_BASE is set to {api_base}")
        openai.api_base = api_base


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)


def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)  # What's this magic 20?
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                                       n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs


async def agpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return await achatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)


async def achatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)  # What's this magic 20?
        n -= cnt
        res = await acompletions_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop
        )
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs


def normal_backoff(mu=10.0, sigma=1.0, min_val=0.0, max_val=float('inf')):
    def backoff_generator():
        while True:
            yield max(min_val, min(max_val, random.normalvariate(mu, sigma)))

    return backoff_generator


@backoff.on_exception(normal_backoff(mu=15.0, sigma=2.5, min_val=0.0, max_val=60.0), openai.error.OpenAIError)
async def acompletions_with_backoff(**kwargs):
    return await openai.ChatCompletion.acreate(**kwargs)


def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = (completion_tokens + prompt_tokens) / 1000 * 0.0002
    else:
        cost = "unknown"
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
