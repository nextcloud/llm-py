import argparse

from chains.formalize import FormalizeChain
from chains.simplify import SimplifyChain
from chains.summarize import SummarizeChain
from chains.headline import HeadlineChain
from chains.topics import TopicsChain
from chains.free_prompt import FreePromptChain
from llm import llm
import sys

chains = {}
chains['summarize'] = SummarizeChain(llm=llm)
chains['simplify'] = SimplifyChain(llm=llm)
chains['formalize'] = FormalizeChain(llm=llm)
chains['headline'] = HeadlineChain(llm=llm)
chains['topics'] = TopicsChain(llm=llm)
chains['free_prompt'] = FreePromptChain(llm=llm)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", choices=chains.keys(), default='summarize')
parser.add_argument("-f", "--file", default='./test/fixtures/meeting2.txt')
parser.add_argument("--text")

args = parser.parse_args()

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
