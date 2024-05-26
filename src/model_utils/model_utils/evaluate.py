import evaluate
import numpy as np
import os
import pandas as pd
import pickle as pkl

PWD = os.environ["WORKSPACE_PATH"]

PREC = evaluate.load("precision")
ACC = evaluate.load("recall")
F1= evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)

    precision = PREC.compute(predictions=predictions, references=labels, average='macro')["precision"]
    recall = ACC.compute(predictions=predictions, references=labels, average='macro')["recall"]
    f1 = F1.compute(predictions=predictions, references=labels, average='macro')["f1"]

    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1": f1}

def get_overall_metrics(predictions, references):
    args = {"predictions": predictions,  "references": references, "average": "macro"}

    scores = {}
    scores.update(F1.compute(**args))
    scores.update(ACC.compute(**args))
    scores.update(PREC.compute(**args))

    return scores

def get_class_metrics(predictions, references, data_type):
    args = {"predictions": predictions,  "references": references, "average": None}
    scores = {}
    scores.update(F1.compute(**args))
    scores.update(ACC.compute(**args))
    scores.update(PREC.compute(**args))


    class_scores = np.concatenate([val.reshape(-1, 1) for val in scores.values()], axis=1)

    with open(f"{PWD}/data/label_encoders/{data_type}.pkl", "rb+") as fp:
        label_enc = pkl.load(fp)

    class_names = np.array(label_enc.inverse_transform(range(2))).reshape(-1, 1)
    return pd.DataFrame(np.concatenate([class_names, class_scores], axis=1), columns=["label", *scores.keys()])
