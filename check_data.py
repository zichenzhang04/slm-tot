import pandas as pd


df = pd.read_csv('datasets/finetune.csv')
df = df.drop(columns=['Rank'])
df.to_csv('datasets/finetune.csv', index=False)