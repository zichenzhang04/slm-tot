### testing framework for outputs
### sample output:
'''10 - 4 = 6 (left: 5 6 6)
5 * 6 = 30 (left: 6 30)
30 - 6 = 24 (left: 24)
Answer: (5 * (10 - 4)) - 6 = 24
'''
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sys import exit
'''
We should test the accuracy based on two metrics: 
    (1) 
        [1.1] check if the numbers used are in the left: 
        of the previous step
        [1.2] check if the left of the current step contains 
        the numbers not used 
        [1.3] the output of the expression is correct.
    (2) check if last step outputs 24
'''
################## BEGIN CLASS DEFINITION ##################
## class that is an evaluator
class Gameof24OutputTester():
    # constructor
    def __init__(self, puzzle: str, response: str, solution: list = []):
        # original puzzle
        self.puzzle = puzzle.split()
        # lists to store unused and used numbers
        self.unused_nums = self.puzzle.copy()
        self.used_nums = []
        # do we need the solution?
        self.solution = solution.copy()
        # preprocess response string
        self.response = response.strip().split("\n")
        for j in range(len(self.response)):
            self.response[j] = self.response[j].strip()
        if self.response[0] == "Steps:":
            self.response.pop(0)
        self.response[-1] = self.response[-1].replace('###', '').strip()
        self.response[-1] = self.response[-1].replace("Final", "").strip()
        while len(self.response) > 0 and not self.response[-1].startswith("Answer:"):
            self.response.pop()

    # get the left values of a regular line
    def get_left(self, rline: str):
        leftPos = rline.find("(left:") + len("(left:")
        closeParPos = rline.find(")")
        leftNums = rline[leftPos:closeParPos]
        leftNums = leftNums.replace(",","").strip()
        return leftNums.split()

    # get the expression and numbers of a regular line
    def get_nums_and_expr(self, rline: str):
        leftPos = rline.find("(left:")
        expr = rline[:leftPos].strip()
        # subfunction to find the oprator within a line
        def find_operator(exp: str):
            for op in "+-*/":
                if op in exp:
                    return op
        opr = find_operator(expr)
        num1, num2 = expr.split("=")[0].strip().split(opr)
        num1 = num1.strip()
        num2 = num2.strip()
        return (num1, num2, expr.replace("=","=="))
    
    # check that the used in expression are in the left of the previous step (1.1)
    def check_used_in_left_of_previous(self, thisStep: str, prevStep: str):
        this1, this2, _ = self.get_nums_and_expr(rline=thisStep)
        last_lefts = self.get_left(rline=prevStep)
        return (this1 in last_lefts) and (this2 in last_lefts)

    # check if the unused numbers are in the left of the current step (1.2)
    def check_unused_in_left_of_current(self, step: str):
        lefts = self.get_left(rline=step)
        for un in self.unused_nums:
            if un not in lefts:
                return False
        return True

    # evaluate a regular line (not the answer) according to the guidelines in the string above
    # Ex: 5 * 6 = 30 (left: 6 30)
    def eval_regular_line(self, rline: str, first: bool, prevLine: str = ""):
        try:
            # list that keeps track of which conditions we met, in the order as described above
            conditions = [first, False, False]
            # get the expression and numbers of this line
            n1, n2, expr = self.get_nums_and_expr(rline=rline)
            # remove the used numbers from unused, and put into used
            if len(self.unused_nums) != 0 and n1 in self.puzzle and n1 in self.unused_nums:
                self.unused_nums.remove(n1)
                self.used_nums.append(n1)
            if len(self.unused_nums) != 0 and n2 in self.puzzle and n2 in self.unused_nums:
                self.unused_nums.remove(n2)
                self.used_nums.append(n2)
            # check conditions [1.2] and [1.3]
            cond12 = self.check_unused_in_left_of_current(step=rline)
            cond13 = eval(expr)
            conditions[1] = cond12
            conditions[2] = cond13
            # if this is not the first line, we also need to check condition [1.1]
            if not first:
                cond11 = self.check_used_in_left_of_previous(thisStep=rline, prevStep=prevLine)
                conditions[0] = cond11
        except:
            conditions = [False]*3
        return conditions
    
    # evaluate the answer line
    # Ex: Answer: (5 * (10 - 4)) - 6 = 24
    def eval_answer(self, ans: str):
        this_ans = ans.replace("Answer:", "").replace("=","==").strip() + " == 24"
        this_ans = this_ans.replace("###", "")
        this_ans = this_ans.replace("Final", "")
        try:
            expr_result = eval(this_ans)
            if hasattr(expr_result, "__iter__") and len(expr_result)==1:
                expr_result = expr_result.pop()
        except:
            expr_result = False
        return expr_result
    
    # main procedure - iterate through all lines of response
    def eval_response(self):
        out = {"Numbers used are in the left of the previous step": None,
               "The left of the current step contains the numbers not used": None,
               "The output of the expression is correct": None,
               "Last step outputs 24": None,
               "Failure Step Number": "Correct",
               "Failed Step": ""
               }
        outFlag = False
        
        # if the length of the list, excluding the answer, is not at least 4 or not a multiple of 3, we know it did something wrong.
        #   output a message of "fundamental error" so we can distingusih this from the other kind of failure.
        #   This is counted as failing on the first step.
        # if it is a multiple of 3, look at the last 4 lines ONLY.
        if len(self.response) < 4 or (len(self.response)-1)%3 != 0:
            count = 0
            for stt in out.keys():
                if count < 4:
                    out[stt] = False
                    count += 1  
            out["Failure Step Number"] = "1"
            return out, True, "fundamental error"
        else:
            self.response = self.response[-4:]
        
        # normal process
        for i in range(len(self.response)):
            # if first line, remember - we don't need to check 1.1
            # if last line, it is the answer
            if i != len(self.response)-1:
                c11, c12, c13 = self.eval_regular_line(rline=self.response[i],
                                                    first= i==0,
                                                    prevLine= self.response[i-1] if i != 0 else ""
                                                    )
                out["Numbers used are in the left of the previous step"] = c11
                out["The left of the current step contains the numbers not used"] = c12
                out["The output of the expression is correct"] = c13
                # if any conditions fail, note the line number and step.
                if not c11 or not c12 or not c13:
                    out["Failure Step Number"] = str(i+1)
                    out["Failed Step"] = self.response[i]
                    outFlag = True
                    break
            else:
                final_correct = self.eval_answer(ans=self.response[-1])
                out["Last step outputs 24"] = final_correct
                # if final answer is incorrect, note this.
                if not final_correct:
                    out["Failure Step Number"] = str(len(self.response))
                    out["Failed Step"] = self.response[-1]
                    outFlag = True
        return out, outFlag, ("Note: Conditions with a check result of 'None' were not evaluated because of an intermediate failure.\n"
                              "Alternatively, the final answer may not have been possible to calculate according to the model.") \
                    if outFlag else "OK"
################## END CLASS DEFINITION ##################

# function to produce output file for 'answers_game24_gpt-4_0.7_False.json'
def produce_output_gpt_4():
    infile = "logs/answers_game24_gpt-4_0.7_False.json"
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
    infile2 = "logs/game24_gpt-4o_0.7_False.json"
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
    infile = 'logs/game24_gpt-4o_0.7_False_intermediate.txt'
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
            elif len(newline) == 3 or newline == "1000" or newline == "1001":
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
    gpt4csv = 'logs/answers_game24_gpt-4_0.7_False_test_results.csv'
    gpt4ocsv = 'logs/game24_gpt-4o_0.7_False_test_results.csv'
    '''gpt4_df = pd.read_csv(gpt4csv)
    gpt4o_df = pd.read_csv(gpt4ocsv)'''
    # data for each model
    for data in [gpt4csv, gpt4ocsv]:
        df = pd.read_csv(data)
        counts = Counter(df['Line'].astype(str).tolist())
        missing_counts = {"Correct", "1","2","3","4"} - set(counts.keys())
        for mc in missing_counts:
            counts[mc] = 0
        categories = sorted(list(counts.keys()))
        values = []
        for sc in categories:
            values.append(counts[sc])
        values = pd.Series(counts.values())
        values = (values/sum(values)).tolist()
        fig,ax = plt.subplots()
        ax.bar(categories, values)
        ax.set_ylabel("Fraction")
        extension = "(One-Step ToT)" if data==gpt4ocsv else ""
        ax.set_title("Fraction of samples failed at each step " + extension)
        plt.savefig(data.replace("logs/", "plots/").replace(".csv", ".png"))
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
    produce_output_gpt_4()
    
    # testing on the file 'logs/game24_gpt-4o_0.7_False.json'
    ok = produce_output_gpt_4o()
    if not ok:
        remove_crap_from_gpt4o_output()
        exit(1)
        
    
    # generate the plots as in Figure 3(a)
    generate_plots_one_step_tot()
