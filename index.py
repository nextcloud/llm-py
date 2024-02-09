import argparse
import time
from chains.formalize import FormalizeChain
from chains.simplify import SimplifyChain
from chains.summarize import SummarizeChain
from chains.headline import HeadlineChain
from chains.topics import TopicsChain
from chains.free_prompt import FreePromptChain
from models.model_config import ModelConfig
from langchain.llms import GPT4All
from langchain.chains import LLMChain
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", choices=['summarize', 'simplify', 'formalize', 'headline', 'topics', 'free_prompt'], default='simplify')
parser.add_argument("-f", "--file", default='./test/fixtures/article.txt')
parser.add_argument("--text")
parser.add_argument("--model", default='llama-2-7b-chat.Q4_K_M.gguf')

args = parser.parse_args()

model_config = ModelConfig(args.model)

ctx_length = model_config.get_context_length()

dir_path = os.path.dirname(os.path.realpath(__file__))
threads = os.environ.get('LLM_THREADS', 4)
llm = GPT4All(model=os.path.join(dir_path, "models", args.model), n_threads=int(threads), max_tokens=ctx_length, n_predict=ctx_length/2)

prompt = model_config.get_prompt_template()

llm_chain = LLMChain(llm=llm, prompt=prompt)

chains = {}
chains['summarize'] = SummarizeChain(llm_chain=llm_chain)
chains['simplify'] = SimplifyChain(llm_chain=llm_chain)
chains['formalize'] = FormalizeChain(llm_chain=llm_chain)
chains['headline'] = HeadlineChain(llm_chain=llm_chain)
chains['topics'] = TopicsChain(llm_chain=llm_chain)
chains['free_prompt'] = FreePromptChain(llm_chain=llm_chain)

if args.text:
    if args.text != '-':
        text = args.text
    else:
        text = '\n'.join(sys.stdin.readlines())
elif args.file:
    with open(args.file) as f:
        text = '\n'.join(f.readlines())

out = chains[args.task].run(text)

# Remove eos tokens and any subsequent text:
eos_tokens = model_config.get_eos_tokens()
if eos_tokens is not None:
    for eos_token in eos_tokens:
        out = out.split(eos_token)[0]

print('###')
print(out)
