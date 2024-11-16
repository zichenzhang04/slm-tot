import os
import json
import argparse
import csv
from dotenv import load_dotenv
from openai import OpenAI
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_solutions(index):
    args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='game24',naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
    task = Game24Task()
    ys, infos = solve(args, task, 900)
    return ys

if __name__ == '__main__':
    print(os.getenv('OPENAI_API_KEY'))