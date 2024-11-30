import json
import copy

class Solution:
    def judgePoint24(self, cards):
        sols = []
        def compute(a,b,op):
            if op==0:
                return a+b
            if op==1:
                return a-b
            if op==2:
                return a*b
            return a/b

        def solve(nums):
            nonlocal solCount
            n = len(nums)
            if n==1:
                if goal-e < nums[0] < goal+e:
                    sols.append(copy.deepcopy(sol))
                    solCount += 1
                return
            # stop at one solution
            if solCount>0:
                return
            for i in range(n):
                for j in range(n):
                    if i==j: continue
                    for op in range(3 + (nums[j]!=0)):
                        if i>j and op%2==0: continue
                        x = compute(nums[i], nums[j], op)
                        nums2 = [x]
                        for k in range(n):
                            if k!=i and k!=j:
                                nums2.append(nums[k])
                        sol.append("%s = %s%s%s"%(x,nums[i],operator[op],nums[j]))
                        solve(nums2)
                        sol.pop()
        e = 10**-5
        goal = 24
        operator = "+-*/"
        sol = []
        solCount = 0
        solve(cards)
        print(sols)
        return sols



def find_str_solution(puzzle_str):
    puzzle = puzzle_str.split(" ")
    puzzle = [int(str_num) for str_num in puzzle]
    solution = Solution()
    calc = solution.judgePoint24(puzzle)
    if calc:
        return calc[0]
    else:
        return "No solution"


def add_to_json(filename):
    filepath = "logs/" + filename
    data = {}
    with open(filepath, 'r') as f:
        data = json.load(f)
        for puzzle in data:
            puzzle["original_puzzle"]["solution"] = find_str_solution(puzzle["original_puzzle"]["Puzzles"])
    
    output_filepath = "logs/COT/answers_COT_game24_gpt-4o_0.7_False.json"
    with open(output_filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    add_to_json("COT/COT_game24_gpt-4o_0.7_False.json")  # Change filename here.
        
    

