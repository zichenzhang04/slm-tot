import json
import os
import re
import string
from test import Gameof24OutputTester
import csv

# function to extract all parsed steps from a response's steps.
def extract_steps(steps: str):
    reg_line_pattern = r"\d+\s(-|\*|\+|\/)\s\d+\s=\s\d+\s+\(left:\s((\d+\s*)+|(\d+,\s*)+)\d+\)\n*"
    steps_pieces = steps.split("\n")
    steps_pieces = [pc.strip() for pc in steps_pieces]
    # find and extract the useful information from each line according to the regular expression
    def match_value(expr):
        out = re.search(pattern=reg_line_pattern, string=expr)
        return None if out == None else out.group()
    matches = list(map(lambda rp: match_value(rp), steps_pieces))
    important_steps_pieces = list(map(lambda rp: matches[steps_pieces.index(rp)], steps_pieces))
    while True:
        try:
            important_steps_pieces.remove(None)
        except:
            break
    return "\n".join(important_steps_pieces)

# function to extract the answer from a response's answer and verify its validity
def extract_answer(ans: str):
    proc_ans = ans.replace("Answer:", "")
    unique_chars = set(proc_ans)
    # all characters in this string must be either digits, arithmetic operators (including =), parentheses(), or brackets[].
    is_valid_equation = unique_chars.issubset(set(string.digits+"*/+-()[]= "))
    return (is_valid_equation, "Answer: " + proc_ans if is_valid_equation else "")

if __name__ == "__main__":
    json_files = os.listdir('logs/finetune')
    json_files = sorted(json_files, key=lambda fn: int(fn[fn.find("False_")+len("False_"):fn.find(".json")].replace("-200","")))
    json_files.pop(3)   # we don't need to look at the 151-200 file since it's redundant
    schema = ["Rank","Puzzle","Response"]
    correct, manual = [], []
    correctFile = "datasets/finetune.csv"
    manualFile = correctFile.replace(".csv", "_bad_format.csv")
    # read each file and get the steps and answer, and evaluate for correctness.
    for jf in json_files:
        file = open(f"logs/finetune/{jf}",'r')
        cases = json.load(file)
        file.close()
        for case in cases:
            rank = case["original_puzzle"]['Rank']
            numbers = case["original_puzzle"]['Puzzles']
            steps = extract_steps(case['response']['Steps'])
            good_ans, answer_details = extract_answer(case['response']['Answer'])
            row = dict(zip(schema,[rank,numbers,f"{steps}\n{answer_details}".replace("\n","\\n")]))
            if good_ans:
                tester = Gameof24OutputTester(puzzle=numbers, response=f"{steps}\n{answer_details}")
                _, status, _ = tester.eval_response()
                if not status:
                    correct.append(row)
            else:
                manual.append(row)
    # write csv files
    with open(correctFile, "w", newline='') as cf:
        writer1 = csv.DictWriter(cf, schema)    
        writer1.writeheader()
        writer1.writerows(correct)        
    with open(manualFile, "w", newline='') as mf:
        writer2 = csv.DictWriter(mf, schema)    
        writer2.writeheader()
        writer2.writerows(manual)        
    # print accuracy
    print(len(correct)/1362)

'''p = "4 + 4 = 8 (left: 5 8 8)\n8 + 8 = 16 (left: 5 16)\n16 - 5 = 11 (left: 11)\n\nBacktrack:\n\n4 * 5 = 20 (left: 4 8 20)\n20 - 8 = 12 (left: 4 12)\n4 + 12 = 16 (left: 16)\n\nBacktrack:\n\n5 - 4 = 1 (left: 4 8 1)\n1 + 4 = 5 (left: 8 5)\n8 * 5 = 40 (left: 40)\n\nBacktrack:\n\nTry a different set of operations:\n\n4 * 5 = 20 (left: 4 8 20)\n20 - 4 = 16 (left: 8 16)\n\nBacktrack:\n\n5 + 8 = 13 (left: 4 4 13)\n13 + 4 = 17 (left: 4 17)\n17 - 4 = 13 (left: 13)\n\nBacktrack:\n\nTry another path:\n\n5 + 4 = 9 (left: 4 8 9)\n9 - 4 = 5 (left: 8 5)\n8 * 5 = 40 (left: 40)\n\nBacktrack:\n\n4 * 8 = 32 (left: 4 5 32)\n32 - 5 = 27 (left: 4 27)\n27 - 4 = 23 (left: 23)\n\nFinally, try:\n\n5 - 4 = 1 (left: 4 8 1)\n1 + 8 = 9 (left: 4 9)\n4 * 9 = 36 (left: 36)\n\nBacktrack:\n\nTry a new path:\n\n8 - 4 = 4 (left: 4 5 4)\n4 + 5 = 9 (left: 4 9)\n9 * 4 = 36 (left: 36)\n\nBacktrack:\n\n4 + 8 = 12 (left: 4 5 12)\n12 - 4 = 8 (left: 5 8)\n5 * 8 = 40 (left: 40)\n\nFinally, solve successfully:\n\n4 + 4 = 8 (left: 5 8 8)\n8 + 5 = 13 (left: 8 13)\n13 + 8 = 21 (left: 21)\n\nBacktrack:\n\n5 + 4 = 9 (left: 4 8 9)\n9 + 4 = 13 (left: 8 13)\n13 + 8 = 21 (left: 21)\n\nTry:\n\n4 + 4 = 8 (left: 5 8 8)\n8 - 5 = 3 (left: 8 3)\n3 * 8 = 24 (left: 24)\n\nSolution Found!"
l = re.split('\n+',p)
matches = map(lambda s: match_value(s))


def match_value(expr):
    out = re.search(pattern=reg_line_pattern, astr=expr)
    return out.group() if type(out) != None else None'''
    