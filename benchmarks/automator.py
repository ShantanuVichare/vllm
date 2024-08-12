import subprocess

def execute_commands(commands, filenames):
    if len(commands) != len(filenames):
        raise ValueError("The number of commands must match the number of filenames.")

    for command, filename in zip(commands, filenames):
        try:
            # Execute the command and capture both stdout and stderr
            result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
            output_to_write = result.stdout

        except subprocess.CalledProcessError as e:
            # Log both stdout and stderr from the failed command
            output_to_write = f"Command failed with exit code {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}"

        finally:
            # Write output to the corresponding file
            with open(filename, 'w') as file:
                file.write(output_to_write)

# Example usage:
commands = [
    # 'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-125m --num-speculative-tokens 3 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    # 'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-350m --num-speculative-tokens 3 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    # 'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-1.3b --num-speculative-tokens 3 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    # 'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-2.7b --num-speculative-tokens 3 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    
    # 'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-125m --num-speculative-tokens 5 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    # 'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-350m --num-speculative-tokens 5 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    # 'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-1.3b --num-speculative-tokens 5 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    # 'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-2.7b --num-speculative-tokens 5 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    
    'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-2.7b --num-speculative-tokens 2 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-2.7b --num-speculative-tokens 3 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-2.7b --num-speculative-tokens 4 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
    'python benchmarks/benchmark_throughput.py --model facebook/opt-6.7b --speculative-model facebook/opt-2.7b --num-speculative-tokens 5 --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --output-len 512 --num-prompts 100',
]
filenames = [
    # 'benchmark_throughput_6.7b_125m_3.log',
    # 'benchmark_throughput_6.7b_350m_3.log',
    # 'benchmark_throughput_6.7b_1.3b_3.log',
    # 'benchmark_throughput_6.7b_2.7b_3.log',
    
    # 'benchmark_throughput_6.7b_125m_5.log',
    # 'benchmark_throughput_6.7b_350m_5.log',
#     'benchmark_throughput_6.7b_1.3b_5.log',
#     'benchmark_throughput_6.7b_2.7b_5.log',

    'benchmark_throughput_6.7b_2.7b_2.log',
    'benchmark_throughput_6.7b_2.7b_3.log',
    'benchmark_throughput_6.7b_2.7b_4.log',
    'benchmark_throughput_6.7b_2.7b_5.log',
]

# commands = [
#     'K_LEN=0 python benchmarks/benchmark_run.py',
#     'K_LEN=1 python benchmarks/benchmark_run.py',
#     'K_LEN=2 python benchmarks/benchmark_run.py',
#     'K_LEN=3 python benchmarks/benchmark_run.py',
#     'K_LEN=4 python benchmarks/benchmark_run.py',
#     'K_LEN=5 python benchmarks/benchmark_run.py',
# ]
# filenames = [
#     'benchmark_throughput_6.7b_125m_2.log',
#     'benchmark_throughput_6.7b_350m_2.log',
#     'benchmark_throughput_6.7b_1.3b_2.log',
#     'benchmark_throughput_6.7b_2.7b_2.log',
    
#     'benchmark_throughput_6.7b_125m_4.log',
#     'benchmark_throughput_6.7b_350m_4.log',
#     'benchmark_throughput_6.7b_1.3b_4.log',
#     'benchmark_throughput_6.7b_2.7b_4.log',
# ]

execute_commands(commands, filenames)
