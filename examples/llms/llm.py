#!/usr/bin/env python3

from collections import namedtuple
import functools
import logging
import timeit

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import prix
import prix.communicator as comm
from prix.config import cfg

Runtime = namedtuple("Runtime", "mid q1 q3")


def time_me(func=None, n_loops=1):
    """Decorator returning average runtime in seconds over n_loops."""
    if func is None:
        return functools.partial(time_me, n_loops=n_loops)

    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        times = []
        for _ in range(n_loops):
            start = timeit.default_timer()
            return_val = func(*args, **kwargs)
            times.append(timeit.default_timer() - start)
        mid_runtime = np.quantile(times, 0.5)
        q1_runtime = np.quantile(times, 0.25)
        q3_runtime = np.quantile(times, 0.75)
        runtime = Runtime(mid_runtime, q1_runtime, q3_runtime)
        return runtime, return_val

    return timing_wrapper


class LLMs:
    """LLM benchmarks runtime of PriX models."""

    def __init__(self, model, tensor_size, device="cpu", full=True, kv_cache=0):
        from examples.llms.models.bert import BertBase, BertLarge, BertTiny
        from examples.llms.models.gpt2 import GPT2
        from examples.llms.models.gpt_neo import GPTNeo
        from examples.llms.models.llama import Llama1B, Llama8B
        from examples.llms.models.modern_bert import ModernBert, ModernBertLarge

        all_models = {
            "gpt2": GPT2,
            "gptneo": GPTNeo,
            "berttiny": BertTiny,
            "bertbase": BertBase,
            "bertlarge": BertLarge,
            "modernbert": ModernBert,
            "modernbertlarge": ModernBertLarge,
            "llama": Llama1B,
            "llama8b": Llama8B,
        }

        self.device = device
        self.tensor_size = tensor_size
        self.df = None
        self.full = full
        self.kv_cache = kv_cache
        model = model.lower()
        if model in all_models:
            if "llama" in model:
                m_clear = all_models[model](
                    seq_len=tensor_size[1], full=full, cache=(kv_cache != 0)
                )
            else:
                m_clear = all_models[model](seq_len=tensor_size[1], full=full)
            if hasattr(m_clear, "to") and self.device is not None:
                m_clear = m_clear.to(self.device)
            self.models = [m_clear.encrypt(src=0)]
        else:
            raise ValueError(
                f"Invalid model name: {model}. Choose from: {', '.join(all_models.keys())}"
            )

    def __repr__(self):
        if self.df is not None:
            return " ".join(self.df.astype(str).values.flatten())
        return "No Function Benchmarks"

    @staticmethod
    @time_me
    def time_llm(x, model, kv_cache=0):
        with prix.no_grad():
            output = model(x)
        for _ in range(kv_cache):
            output = model(x[:, :1, :])
        return output

    def get_runtimes(self):
        """Returns PriX encrypted runtimes."""
        runtimes_enc = []
        key = jax.random.PRNGKey(42)
        for llm in self.models:
            if self.full:
                x = jax.random.uniform(key, shape=self.tensor_size, dtype=jnp.float32)
            else:
                shape = (self.tensor_size[0], self.tensor_size[1], llm.embed_dim)
                x = jax.random.uniform(key, shape=shape, dtype=jnp.float32)
            x_enc = prix.cryptensor(x, device=self.device)

            llm.eval()

            runtime_enc, _ = LLMs.time_llm(x_enc, llm, kv_cache=self.kv_cache)
            runtimes_enc.append(runtime_enc)

        return runtimes_enc

    def run(self):
        """Runs and stores benchmarks in self.df."""
        runtimes_enc = self.get_runtimes()

        self.df = pd.DataFrame.from_dict(
            {
                "function": self.models,
                "runtime": [r.mid for r in runtimes_enc],
            }
        )


def run_llm(
    tensor_size,
    model,
    fill_cache=False,
    communication=False,
    full=True,
    device=None,
    kv_cache=0,
):
    rank = comm.get().get_rank()
    logging.info(f"[Party {rank}][Device] running in {device}")
    logging.info(f"[Party {rank}] Tensor size {tensor_size}")

    if communication:
        comm.get().set_verbosity(True)

    functions_data = cfg.config.get("functions", {})
    filtered_data = {
        key: dict(value)["method"]
        for key, value in functions_data.items()
        if "method" in dict(value)
    }
    logging.info(f"[Party {rank}] Config: {filtered_data}")

    provider = prix.mpc.get_default_provider()
    if fill_cache:
        logging.info(f"[Party {rank}] Tracing requests for the cache " + "=" * 20)
        provider.trace_once()
    else:
        provider.load_cache()

    benches = LLMs(model, tensor_size, device=device, full=full, kv_cache=kv_cache)
    benches.run()

    logging.info(f"[Party {rank}] {benches}")

    if fill_cache:
        logging.info(f"[Party {rank}] Filling the cache " + "=" * 20)
        provider.fill_cache()

    if communication:
        comm.get().print_communication_stats()
        return

    logging.info(f"[Party {rank}] Done")
