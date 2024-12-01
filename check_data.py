import pandas as pd


df = pd.read_csv('datasets/finetune.csv')
print(df['Response'][0])