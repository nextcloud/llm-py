import argparse
import time
from chains.formalize import FormalizeChain
from chains.simplify import SimplifyChain
from chains.summarize import SummarizeChain
from chains.headline import HeadlineChain
from chains.topics import TopicsChain
from chains.free_prompt import FreePromptChain
import sys
from langchain.llms import GPT4All
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", choices=['summarize', 'simplify', 'formalize', 'headline', 'topics', 'free_prompt'], default='simplify')
parser.add_argument("-f", "--file", default='./test/fixtures/article.txt')
parser.add_argument("--text")
parser.add_argument("--model", default='llama-2-7b-chat.ggmlv3.q4_0.bin')

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
threads = os.environ.get('LLM_THREADS', 4)
llm = GPT4All(model=dir_path + "/models/"+args.model, n_threads=int(threads))

chains = {}
chains['summarize'] = SummarizeChain(llm=llm)
chains['simplify'] = SimplifyChain(llm=llm)
chains['formalize'] = FormalizeChain(llm=llm)
chains['headline'] = HeadlineChain(llm=llm)
chains['topics'] = TopicsChain(llm=llm)
chains['free_prompt'] = FreePromptChain(llm=llm)

if args.text:
    if args.text != '-':
        text = args.text
    else:
        text = '\n'.join(sys.stdin.readlines())
elif args.file:
    with open(args.file) as f:
        text = '\n'.join(f.readlines())

out = chains[args.task].run(text)
print('###')
print(out)
