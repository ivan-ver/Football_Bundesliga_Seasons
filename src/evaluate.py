import pandas as pd
import sys
import pickle
from dvclive import Live
import os
import json
import math
from matplotlib import pyplot as plt

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


EVAL_PATH = "eval"

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

input_path = sys.argv[1]
model_path = sys.argv[2]


with open(model_path, "rb") as fd:
    model = pickle.load(fd)

live = Live(os.path.join(EVAL_PATH, "live"), dvcyaml=False)


def evaluate(path, model, live, df_type_name):
    df = pd.read_csv(path)
    y = np.array(df['IS_WIN_NUM'])
    X = df.drop('IS_WIN_NUM', axis=1)

    predictions_prod = model.predict_proba(X)
    predictions = np.array(predictions_prod[:, 1])
    
    
    avg_prec = average_precision_score(y, predictions)
    roc_auc_prec = roc_auc_score(y, predictions)


    precision, recall, prc_thresholds = precision_recall_curve(y, predictions)

    if not live.summary:
        live.summary = {"avg_prec": {}, "roc_auc": {}}
    live.summary["avg_prec"][df_type_name] = avg_prec
    live.summary["roc_auc"][df_type_name] = roc_auc_prec

    live.log_sklearn_plot("roc", y, predictions, name=f"roc/{df_type_name}")

    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
    prc_dir = os.path.join(EVAL_PATH, "prc")
    
    os.makedirs(prc_dir, exist_ok=True)
    prc_file = os.path.join(prc_dir, f"{df_type_name}.json")
    
    with open(prc_file, "w") as fd:
        json.dump({"prc": [{"precision": p, "recall": r, "threshold": t} for p, r, t in prc_points]}, fd, indent=4,)


    live.log_sklearn_plot("confusion_matrix",
                          y.squeeze(),
                          predictions_prod.argmax(-1),
                          name=f"cm/{df_type_name}"
                         )
    
 


evaluate(os.path.join(input_path, "train.csv"), model, live, "train")
evaluate(os.path.join(input_path, "test.csv"), model, live, "test")

live.make_summary()
