import pandas as pd

df = pd.read_csv(r'E:\Natural Language Processing Projects\Fake news Classifier\train.csv')
df = df.dropna()
X = df.drop('label',axis=1)
Y = df['label']



