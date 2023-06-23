from langchain.llms import GPT4All

llm = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=512, n_threads=6)