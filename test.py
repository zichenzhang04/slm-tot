### testing framework for outputs
### sample output:
'''10 - 4 = 6 (left: 5 6 6)
5 * 6 = 30 (left: 6 30)
30 - 6 = 24 (left: 24)
Answer: (5 * (10 - 4)) - 6 = 24
'''
import json
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
        while not self.response[-1].startswith("Answer:"):
            self.response.pop()

    # get the left values of a regular line
    def get_left(self, rline: str):
        leftPos = rline.find("(left:") + len("(left:")
        closeParPos = rline.find(")")
        leftNums = rline[leftPos:closeParPos]
        leftNums = leftNums.strip()
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
        return conditions
    
    # evaluate the answer line
    # Ex: Answer: (5 * (10 - 4)) - 6 = 24
    def eval_answer(self, ans: str):
        this_ans = ans.replace("Answer:", "").replace("=","==").strip() + " == 24"
        try:
            expr_result = eval(this_ans)
        except:
            expr_result = False
        return expr_result
    
    # main procedure - iterate through all lines of response
    def eval_response(self):
        out = {"Numbers used are in the left of the previous step": None,
               "The left of the current step contains the numbers not used": None,
               "The output of the expression is correct": None,
               "Last step outputs 24": None,
               }
        outFlag = False
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
                    out["Failure Step Number"] = i+1
                    out["Failed Step"] = self.response[i]
                    outFlag = True
                    break
            else:
                final_correct = self.eval_answer(ans=self.response[-1])
                out["Last step outputs 24"] = final_correct
                # if final answer is incorrect, note this.
                if not final_correct:
                    out["Failure Step Number"] = len(self.response)
                    out["Failed Step"] = self.response[-1]
                    outFlag = True
        return out, outFlag, ("Note: Conditions with a check result of 'None' were not evaluated because of an intermediate failure.\n"
                              "Alternatively, the final answer may not have been possible to calculate according to the model.") \
                    if outFlag else "OK"
################## END CLASS DEFINITION ##################

# function to produce output file for multiple examples
def produce_output(infile: str):
    with open(infile, "r") as resfile:
        other_examples = json.load(resfile)
    # output file
    outfile = infile.replace(".json", "_test_results.txt")
    finalfile = open(outfile, "a")
    for oe in other_examples:
        try:
            numbers = oe['original_puzzle']['Puzzles']
            resp = oe['response']
            tester = Gameof24OutputTester(puzzle=numbers, response=resp)
            res, status, statement = tester.eval_response()
            #finalfile.writelines(f"{numbers} | {resp} | {res} | {status} | {statement}\n")
            finalfile.writelines(numbers+'\n')
            finalfile.writelines(resp+'\n')
            for k, v in res.items():
                finalfile.writelines(f"{k}: {v}\n")
            finalfile.writelines(statement+'\n\n')
        except:
            continue

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
    infile = "answers_game24_gpt-4_0.7_False.json"
    produce_output(infile=infile)
    
