from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_combined_plot():
    files = {"One-Step-ToT": "logs/one_step_ToT/game24_gpt-4o_0.7_False_test_results.csv",
            "CoT": "logs/COT/COT_game24_gpt-4o_0.7_False_test_results.csv"}
    # data for each model
    categories = ["1","2","3","4", "Correct"]
    step_counts = {"CoT": [0]*5, "One-Step-ToT": [0]*5}
    for title, data in files.items():
        df = pd.read_csv(data)
        counts = Counter(df['Line'].astype(str).tolist())
        missing_counts = {"Correct", "1","2","3","4"} - set(counts.keys())
        for mc in missing_counts:
            counts[mc] = 0
        values = []
        for sc in categories:
            values.append(counts[sc])
        total_of_values = sum(values)
        for v in range(len(values)):
            values[v] /= total_of_values
        step_counts[title] = values.copy()
    # generate bar plot
    x = np.arange(len(categories))  # the label locations
    width = 0.3  # the width of the bars
    mult = 0
    fig,ax = plt.subplots(layout='constrained')
    for model, counts in step_counts.items():
        offset = width*mult
        bars = ax.bar(x+offset, counts, width, label=model)
        ax.bar_label(bars, padding=2)
        mult += 1
    ax.set_ylabel("Fraction")
    ax.set_title("Fraction of samples failed at each step for CoT and One-Step ToT")
    ax.set_xticks(x+width, categories)
    ax.legend(loc="upper right", ncols=2)
    plt.savefig("plots/failed_sampled_cot_and_one_step_tot")
    return

if __name__ == "__main__":
    generate_combined_plot()