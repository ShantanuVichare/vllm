
from typing import List, Optional, Tuple
import json
import random
import time

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

import torch
torch.cuda.empty_cache()
import gc
gc.collect()

from vllm import LLM, SamplingParams
from vllm.spec_decode.util import nvtx_range

def log_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

output_len = 512
sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=output_len)

# prompts = [
#     "When she opened the door, she immediately noticed the",
#     "He never expected that the book would contain a",
#     "On their way to the park, they stumbled upon a",
#     "She glanced at the mysterious message left on her",
#     "During the darkest night, the stars revealed the path",
#     "The ancient map led them not to treasure, but to",
#     "In the quiet library, an old diary fell from a",
#     "The whispered secrets of the forest became louder as",
#     "He was shocked to find an old letter inside the",
#     "They never spoke again after the incident near the",
#     "The clock struck midnight as the glass slipper was",
#     "Under the old oak tree, they found a buried",
#     "The last thing she expected to see was a",
#     "As the fog cleared, the abandoned ship appeared more",
#     "Nobody knew why the painting was hidden away for",
#     "The message in the bottle traveled across the ocean",
#     "The legend said only at sunset could you see",
#     "The camera captured something unexpected in the background during",
#     "She heard the music play even though the room",
#     "They decided to explore the cave despite the warnings",
#     "The recipe was supposed to be for a simple",
#     "When they looked up, the sky was filled with",
#     "The experiment was nearly complete when suddenly the power",
#     "He reached into his pocket and realized his key",
#     "She found the old photograph tucked in the pages",
#     "The instructions on the old typewriter were cryptic but",
#     "As the train pulled away, she saw a figure",
#     "The bookshop closed each night, but at midnight the",
#     "The villagers spoke of a castle that vanished at",
#     "Every time he heard the melody, an old memory",
#     "The mirror was supposed to be an ordinary household",
#     "They followed the trail, which ended abruptly at a",
# ]


# With ngram spec
# k = 5
# llm = LLM(
#     model="facebook/opt-6.7b",
#     tensor_parallel_size=1,
#     speculative_model="[ngram]",
#     num_speculative_tokens=k,
#     use_v2_block_manager=True,
#     disable_log_stats=False,
#     enable_prefix_caching=True,
#     ngram_prompt_lookup_min=2,
#     ngram_prompt_lookup_max=k,
# )

# With spec decode
# k = 5
# llm = LLM(
#     model="facebook/opt-6.7b",
#     tensor_parallel_size=1,
#     speculative_model="facebook/opt-125m",
#     num_speculative_tokens=k,
#     use_v2_block_manager=True,
#     disable_log_stats=False,
#     disable_logprobs_during_spec_decoding=False, # default is True
#     device="cuda",
# )

# Without spec decode
# llm = LLM(
#     model="facebook/opt-6.7b",
#     tensor_parallel_size=1,
#     use_v2_block_manager=True,
#     disable_log_stats=False,
# )

# with nvtx_range("warmup"):
#     outputs = llm.generate(prompts[:], sampling_params)
    

# with nvtx_range("llm_generate"):
#     outputs = llm.generate(prompts[:], sampling_params)
    
def mean(lst):
    return sum(lst) / len(lst)

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=None)
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset

def warmup(llm):
    print('warmup started')
    import numpy as np
    from vllm.inputs import PromptInputs
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(16,
                                                     128))
    dummy_inputs: List[PromptInputs] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]
    for _ in range(10):
        llm.generate(
            dummy_inputs,
            sampling_params=\
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=16,
                ),
            use_tqdm=False)
    print('warmup ended')

model = "facebook/opt-6.7b"
speculative_model = "facebook/opt-125m"
dataset_path = "benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json"

import os
k = int(os.getenv('K_LEN'))
if k == 0: 
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        # speculative_model=speculative_model,
        # num_speculative_tokens=k,
        use_v2_block_manager=True,
        disable_log_stats=False,
        device="cuda",
    )
else:
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        speculative_model=speculative_model,
        num_speculative_tokens=k,
        use_v2_block_manager=True,
        disable_log_stats=False,
        device="cuda",
    )
avg_runs = 5

warmup(llm)

print('k =', k)
for bs in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]:

    requests = sample_requests(dataset_path, bs, model, 100)

    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                # n=n,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
            ))
    
    elapsed_times = []
    acceptance_rates = []
    efficiences = []
    msts = []
    
    for _ in range(avg_runs):
        start = time.perf_counter()
        results = llm.generate(prompts, sampling_params, use_tqdm=True)
        end = time.perf_counter()
        elapsed_time = end - start
        spec_metrics = results[-1]

        total_num_tokens = sum(prompt_len + output_len for _, prompt_len, output_len in requests)
        mst = elapsed_time/total_num_tokens*1000

        elapsed_times.append(elapsed_time)
        acceptance_rates.append(spec_metrics.draft_acceptance_rate)
        efficiences.append(spec_metrics.system_efficiency)
        msts.append(mst)
    print('log:', bs, 'mst:', mean(msts))
    # print('log:', bs, 'elapsed_time:', mean(elapsed_times), 'acceptance_rate:', mean(acceptance_rates), 'efficiency:', mean(efficiences), 'mst:', mean(msts))

