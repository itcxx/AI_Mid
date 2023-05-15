import sys
sys.path.append("../")
import pandas as pd
import numpy as np

# add conversation dataset to selected_conversation.txt
df=pd.read_csv("../data/kor-eng/conversations.csv",error_bad_lines=False)
sf=pd.read_csv("../data/kor-eng/1000sents.csv",error_bad_lines=False)
# print(df.head())

# print(df.loc[:,["kor_sent","eng_sent"]])
selected_rows=df.loc[:,["kor_sent","eng_sent"]]
selected_rows2=sf.loc[:,["HEADWORD","ENGLISH"]]
selected_rows3=sf.loc[:,["EXAMPLE (KO)","EXAMPLE (EN)"]]


selected=np.concatenate([selected_rows.values,selected_rows2.values,selected_rows3.values])
selected=pd.DataFrame(selected)


# selected_rows=df.loc[:,["kor_sent","eng_sent"]]
selected.to_csv("../data/kor-eng/selected_conversations1.txt",sep="\t" , index=False)