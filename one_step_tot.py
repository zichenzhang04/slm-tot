import os
import json
import argparse
import csv
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def read_csv(file_path):
    puzzles = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            puzzles.append(row)
    return puzzles


# TODO: work on prompting, make it better
def generate_prompt(puzzle):
    """One Step Tree-of-Thoughts prompting."""
    in_context_demo = (
        "Example:\n"
        "Puzzle: 5 10 4 6\n"
        "Step 1: Start by considering possible operations for each pair of numbers.\n"
        "Step 2: Try a path (a pair of two numbers) like 5 + 10 = 15, see if the remaining numbers 15 4 6 can possibly reach the goal 24. If not, backtrack and attempt another.\n"
        "Step 3: Branch out to try different orders of operations and combinations, evaluating each outcome.\n"
        "Step 4: If one path doesn't lead to a solution, backtrack and try alternative operations.\n"
        "After reaching the result 24, only output the following exactly in the same format:\n"
        "10 - 4 = 6 (left: 5 6 6)\n"
        "5 * 6 = 30 (left: 6 30)\n"
        "30 - 6 = 24 (left: 24)\n"
        "Answer: (5 * (10 - 4)) - 6 = 24\n\n"
    )

    prompt = (
        f"{in_context_demo}"
        f"Now, solve the following puzzle:\n{puzzle}\n"
    )
    return prompt


def prompt_gpt(puzzle, backend, temperature):
    prompt = generate_prompt(puzzle)
    response = client.chat.completions.create(
        model=backend,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


def generate_log_filename(args):
    filename = f"./logs/{args.task}_{args.backend}_{args.temperature}_{args.test_mode}.json"
    return filename


def run(args):
    # Choose dataset path based on test_mode
    dataset_path = './datasets/24_test.csv' if args.test_mode else './datasets/24.csv'
    puzzles = read_csv(dataset_path)
    log = []

    # Solve every puzzle in the dataset
    for puzzle in puzzles:
        puzzle_text = puzzle['Puzzles']
        response = prompt_gpt(puzzle_text, args.backend, args.temperature)
        # Log the model's response
        log_entry = {
            "original_puzzle": puzzle,  # Record the question
            "response": response
        }
        log.append(log_entry)

    os.makedirs('./logs', exist_ok=True)
    # Generate and save the log file
    log_filename = generate_log_filename(args)
    with open(log_filename, 'w') as f:
        json.dump(log, f, indent=4)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str,
                      choices=['gpt-4o', 'gpt-4o-mini'], default='gpt-4o')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--task', type=str,
                      choices=['game24', 'text', 'crosswords'], default="game24")
    args.add_argument('--test_mode', action='store_true',
                      help="Use test dataset if set", default=False)
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
