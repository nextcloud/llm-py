from langchain.llms import GPT4All
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

llm = GPT4All(model=dir_path + "/models/ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=512, n_threads=6)