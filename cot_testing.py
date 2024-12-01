import json
import csv
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from collections import Counter
from sys import exit
from test import Gameof24OutputTester

# function to produce output file for 'answers_game24_gpt-4_0.7_False.json'
def produce_output_gpt_4():
    infile = "logs/COT/answers_COT_game24_gpt-4o_0.7_False.json"
    with open(infile, "r") as resfile:
        other_examples = json.load(resfile)
    # output file
    outfile = infile.replace(".json", "_test_results.csv")
    results = []
    for oe in other_examples:
        try:
            numbers = oe['original_puzzle']['Puzzles']
            rank = oe['original_puzzle']['Rank']
            resp = oe['response']
            tester = Gameof24OutputTester(puzzle=numbers, response=resp)
            res, status, statement = tester.eval_response()
            puzzResult = {"Rank": rank,
                        "Puzzle": numbers,
                        "Info": res,
                        "Line": res['Failure Step Number'],
                        "Status": status,
                        "Statement": statement
                        }
            results.append(puzzResult)
            # write to csv
            with open(outfile, 'w', newline="") as csvfile:
                fields = list(results[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                writer.writerows(results)
        except:
            continue
    return 

# function to get rid of crap from the 'game24_gpt-4o_0.7_False.json'. Still need to do a minor double check.
def remove_crap_from_gpt4o_output():
    infile2 = "logs/COT/COT_game24_gpt-4o_0.7_False.json"
    keywords = ["try", "another", "backtrack", "different", "incorrect", "attempt", "mistake", 
                "approach", "path", "correct", "steps", "solution", "add", "divide", "multiply", "subtract", 
                "finally", "conclusion", "sequence", "explanation", "doesn't", "does not", "operations", "still", 
                "re-evaluate", "combin", "(not", "already", "trivial", "rethink", "error", "adjusting", "let's finalize", "none"]
    outfile = open(infile2.replace(".json", ".txt").replace("_False", "_False_intermediate"), mode='w')
    phrase_replace = {"(": ['*('],
                    ")": [')*'],
                    "*": ["times"],
                    "left": ["Now the numbers are"],
                    "(left:": ["(:"],
                    "": ["( ", " )", "quad ", "text{", "}"]
                }
    outfile.writelines("False"+'\n')
    with open(infile2) as df:
        jj = json.load(df)
        for dc in jj:
            puzz = dc['original_puzzle']['Puzzles']
            rank = dc['original_puzzle']['Rank']
            split_response = dc['response'].split('\n')
            split_response = [y.strip().replace("**","") for y in split_response]
            for i in range(len(split_response)):
                if len(split_response[i]) == 0:
                    split_response[i] = None
            while True:
                try:
                    split_response.remove(None)
                except:
                    break
            split_response.pop(0)
            while True:
                try:
                    split_response.remove("Steps:")
                except:
                    break
            while True:
                try:
                    split_response.remove("### Steps:")
                except:
                    break
            if "Input" in split_response[0]:
                split_response.pop(0)
            for j in range(len(split_response)):
                for n in range(10):
                    split_response[j] = split_response[j].replace(f"{n}.", "").strip()
                split_response[j] = split_response[j].replace("Remaining numbers", "left").replace("remaining numbers", "left")\
                                    .replace("Remaining number","").replace("remaining number","").replace("\\","")
                if split_response[j].startswith("-"):
                    split_response[j] = split_response[j][1:]
                # replace some common phrases
                for replacement, incorrects in phrase_replace.items():
                    for ip in incorrects:
                        split_response[j] = split_response[j].replace(ip, replacement)
                for on in range(10):
                    split_response[j] = split_response[j].replace(f"(using the hypothetical operation {on} = {on})", "").strip()
                    split_response[j] = split_response[j].replace(f"Step {on}:", "").strip()
                if split_response[j].startswith(":"):
                    split_response[j] = "None"
                # remove lines with common keywords
                contains = list(map(lambda kw: split_response[j].lower().find(kw) != -1, keywords))
                if any(contains):
                    split_response[j] = None
            while True:
                try:
                    split_response.remove(None)
                except:
                    break
            while True:
                try:
                    split_response.remove('[')
                except:
                    break
            while True:
                try:
                    split_response.remove(']')
                except:
                    break
            outfile.writelines(rank+'\n')
            outfile.writelines(puzz+'\n')
            for sr in split_response:
                outfile.writelines(sr+'\n')
            outfile.writelines('\n')
    outfile.close()
    return
            
# function to produce output file for 'game24_gpt-4o_0.7_False.json', using the "intermediate"
def produce_output_gpt_4o():
    proceed = False
    infile = 'logs/COT/COT_game24_gpt-4o_0.7_False_intermediate.txt'
    outfile = infile.replace("_intermediate.txt", "_test_results.csv")
    responses = []
    results = []
    numbers_next = False
    try:
        intdfile = open(infile, 'r')
    except:
        return False
    for line in intdfile:
        newline = line.replace("\n","")
        if newline == "False" or newline == "True":
            proceed = eval(newline)
            if not proceed:
                return True
            else:
                continue
        else:
            # blank line
            if len(newline) == 0:
                continue
            # for each number, we have a puzzle and response. Puzzle comes after Rank
            elif newline.isnumeric() or newline == "1000" or newline == "1001":
                responses.append({"rank": newline, "puzzle": "", "response": ""})
                numbers_next = True
            # numbers line
            elif numbers_next:
                responses[-1]['puzzle'] = newline
                numbers_next = False
            else:
                responses[-1]['response'] += (newline + "\n")
    for puzz in responses:
        rank = puzz['rank']
        numbers = puzz['puzzle']
        resp = puzz['response']
        tester = Gameof24OutputTester(puzzle=numbers, response=resp)
        info, status, statement = tester.eval_response()
        puzzResult = {"Rank": rank,
                        "Puzzle": numbers,
                        "Info": info,
                        "Line": info['Failure Step Number'],
                        "Status": status,
                        "Statement": statement
                        }
        results.append(puzzResult)
        intdfile.close()
    # write to csv
    with open(outfile, 'w', newline="") as csvfile:
        fields = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    return True


def generate_plots_one_step_tot():
    matplotlib.use('Agg')
    #gpt4csv = 'logs/COT/answers_game24_gpt-4_0.7_False_test_results.csv'
    gpt4ocsv = 'logs/COT/COT_game24_gpt-4o_0.7_False_test_results.csv'
    '''gpt4_df = pd.read_csv(gpt4csv)
    gpt4o_df = pd.read_csv(gpt4ocsv)'''
    # data for each model
    for data in [gpt4ocsv]:
        df = pd.read_csv(data)
        counts = Counter(df['Line'].astype(str).tolist())
        missing_counts = {"Correct", "1","2","3","4"} - set(counts.keys())
        for mc in missing_counts:
            counts[mc] = 0
        categories = sorted(list(counts.keys()))
        values = []
        for sc in categories:
            values.append(counts[sc])
        total_of_values = sum(values)
        for v in range(len(values)):
            values[v] /= total_of_values
        print(counts['Correct']/total_of_values)
        fig,ax = plt.subplots()
        ax.bar(categories, values)
        ax.set_ylabel("Fraction")
        extension = "(CoT)" if data==gpt4ocsv else ""
        ax.set_title("Fraction of samples failed at each step " + extension)
        plt.savefig("plots/cot_game24.png")
    return
        

## quick testing
if __name__ == "__main__":
    
    # ex1 = '''10 - 4 = 6 (left: 5 6 6)
    #         5 * 6 = 30 (left: 6 30)
    #         30 - 6 = 24 (left: 24)
    #         Answer: (5 * (10 - 4)) - 6 = 24'''
    # ex2 = "Steps:\n        12 / 4 = 3 (left: 3 5 10)\n        5 * 3 = 15 (left: 10 15)\n        15 + 10 = 25 (left: 25)\n        Answer: (12 / 4 * 5) + 10 = 25\n\nNote: There seems to be a mistake in the given numbers. It's not possible to get 24 with the given numbers (4,5,10,12) using basic arithmetic operations. It can only yield 25 as shown above."
    # tester = Gameof24OutputTester(puzzle='4 5 10 12', response=ex2)
    # res, status, statement = tester.eval_response()
    # for k,v in res.items():
    #     print(f"{k}:",v)
    # print("\n"+statement)

    # testing on the file "answers_game24_gpt-4_0.7_False.json"
    #produce_output_gpt_4()
    
    # testing on the file 'logs/game24_gpt-4o_0.7_False.json'
    ok = produce_output_gpt_4o()
    if not ok:
        remove_crap_from_gpt4o_output()
        exit(1)
        
    
    # generate the plots as in Figure 3(a)
    generate_plots_one_step_tot()