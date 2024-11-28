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


def generate_prompt(puzzle):
    """One Step Tree-of-Thoughts prompting."""
    # 5 examples
    in_context_demo = '''
        Input: 4 4 6 8
        Steps:
        4 + 8 = 12 (left: 4 6 12)
        6 - 4 = 2 (left: 2 12)
        2 * 12 = 24 (left: 24)
        Answer: (6 - 4) * (4 + 8) = 24\n
        Input: 2 9 10 12
        Steps:
        12 * 2 = 24 (left: 9 10 24)
        10 - 9 = 1 (left: 1 24)
        24 * 1 = 24 (left: 24)
        Answer: (12 * 2) * (10 - 9) = 24\n
        Input: 4 9 10 13
        Steps:
        13 - 10 = 3 (left: 3 4 9)
        9 - 3 = 6 (left: 4 6)
        4 * 6 = 24 (left: 24)
        Answer: 4 * (9 - (13 - 10)) = 24\n
        Input: 1 4 8 8
        Steps:
        8 / 4 = 2 (left: 1 2 8)
        1 + 2 = 3 (left: 3 8)
        3 * 8 = 24 (left: 24)
        Answer: (1 + 8 / 4) * 8 = 24\n
        Input: 5 5 5 9
        Steps:
        5 + 5 = 10 (left: 5 9 10)
        10 + 5 = 15 (left: 9 15)
        15 + 9 = 24 (left: 24)
        Answer: ((5 + 5) + 5) + 9 = 24\n
    '''

    system_prompt = (
        "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.\n"
        "Step 1: Start by considering possible operations for each pair of numbers.\n"
        "Step 2: Try a path (a pair of two numbers), see if the remaining numbers can possibly reach the goal 24. If not, backtrack and attempt another.\n"
        "Step 3: Branch out to try different orders of operations and combinations, evaluating each outcome.\n"
        "Step 4: If one path doesn't lead to a solution, backtrack and try alternative operations.\n"
    )

    prompt = (
        f"{system_prompt}"
        f"{in_context_demo}"
        f"Now, solve the following puzzle:\n{puzzle}\n"
        "Output in the same format as this example including three steps and final answer:\n"
        "Steps:\n"
        "5 + 5 = 10 (left: 5 9 10)\n"
        "10 + 5 = 15 (left: 9 15)\n"
        "15 + 9 = 24 (left: 24)\n"
        "Answer: ((5 + 5) + 5) + 9 = 24"
    )
    return prompt


def prompt_gpt(puzzle, backend, temperature):
    prompt = generate_prompt(puzzle)
    # TODO: consider adding response_format parameter to format the JSON output
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
    #count = 0
    # Solve every puzzle in the dataset
    for puzzle in puzzles:
        if 'count >= 901 and count <= 1000':
            puzzle_text = puzzle['Puzzles']
            response = prompt_gpt(puzzle_text, args.backend, args.temperature)
            # Log the model's response
            log_entry = {
                "original_puzzle": puzzle,  # Record the question
                "response": response
            }
            log.append(log_entry)
        #count += 1
        #print(count)

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
