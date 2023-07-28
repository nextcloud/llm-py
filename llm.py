from langchain.llms import GPT4All
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

threads = os.environ.get('LLM_THREADS', 4)

llm = GPT4All(model=dir_path + "/models/ggml-model-gpt4all-falcon-q4_0.bin", n_threads=int(threads))