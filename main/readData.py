import sys
sys.path.append("../")
import pandas as pd


df=pd.read_csv("../data/kor-eng/conversations.csv")
print(df.shape)
print(df.head())

print(df.loc[:,["kor_sent","eng_sent"]])
selected_rows=df.loc[:,["kor_sent","eng_sent"]]
selected_rows.to_csv("../data/kor-eng/selected_conversations.txt",sep="\t" , index=False)