import pandas as pd
import numpy as np
import sys
import pickle
import yaml

from sklearn.linear_model import LogisticRegression

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)

input = sys.argv[1]
output = sys.argv[2]

params = yaml.safe_load(open("params.yaml"))["train"]

train_df = pd.read_csv(input)

y = train_df['IS_WIN_NUM']
y = np.array(y).reshape(-1, 1)
X = train_df.drop('IS_WIN_NUM', axis=1)

model = LogisticRegression().fit(X, y)

with open(output, "wb") as fd:
    pickle.dump(model, fd)
