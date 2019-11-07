import warnings

import numpy as np
import pandas as pd

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


def make_clf_datasets(adatas, y="kg_structure_org_score"):

    channels = adatas["train"].var["channel"]
    feats = {
        "morphological": channels.isin(["segmentation", "bright field", "DNA"]),
        "probe": channels.isin(["FISH"]),
        "structure": channels.isin(["structure"]),
        "all_nonstruct": channels.isin(["segmentation", "bright field", "DNA", "FISH"]),
        "all": channels.isin(
            ["segmentation", "FISH", "bright field", "DNA", "structure"]
        ),
    }

    probes = adatas["train"].obs["FISH_probe"].unique()
    dsets = {}
    for split, ad in adatas.items():
        dsets[split] = {}
        for probe in probes:
            dsets[split][probe] = {}
            for feat, cols in feats.items():
                dsets[split][probe][feat] = {}
                D = ad[ad.obs["FISH_probe"] == probe, :][:, cols].copy()
                dsets[split][probe][feat]["X"] = D.X
                dsets[split][probe][feat]["y"] = D.obs[y].values
                dsets[split][probe][feat]["adata"] = D
    return dsets


def get_classifiers(test=False):

    if not test:
        iter_dummies = ("stratified", "most_frequent", "prior", "uniform")
        iter_logistics = np.geomspace(1e-05, 1e2, num=8)
        iter_knns = [1, 2, 3, 5, 9]
        iter_randforests = np.linspace(3, 10, num=8)
        n_trees = 128
    else:
        iter_dummies = ["most_frequent"]
        iter_logistics = [1e-03]
        iter_knns = [3]
        iter_randforests = [3]
        n_trees = 32

    dummies = {
        f"Dummy {s}": DummyClassifier(strategy=s, random_state=0) for s in iter_dummies
    }

    logistics = {
        f"Logistic Regression C={c}": LogisticRegression(
            multi_class="multinomial",
            penalty="l2",
            C=c,  # 1/lambda
            l1_ratio=None,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=32,
            random_state=0,
        )
        for c in iter_logistics
    }

    knns = {f"KNN k={k}": KNeighborsClassifier(n_neighbors=k) for k in iter_knns}

    randforests = {
        f"Random Forest depth = {d}": RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=d,
            class_weight="balanced",
            n_jobs=32,
            random_state=0,
        )
        for d in iter_randforests
    }

    return {**dummies, **logistics, **knns, **randforests}


def train_classifiers(dsets, classifiers, verbose=False):
    out = {}
    for probe, feat_dset in dsets["train"].items():
        out[probe] = {}
        for feats, dset in feat_dset.items():
            out[probe][feats] = {}
            for clf_name, Clf in classifiers.items():
                if verbose:
                    print(probe, feats, clf_name, dset["X"].shape)
                    clf = sklearn.clone(Clf)
                out[probe][feats][clf_name] = clf.fit(dset["X"], dset["y"])
    return out


def predict_on_all_datasets(trained_classifiers, dsets, verbose=False):
    out = {}
    for split, split_dset in dsets.items():
        out[split] = {}
        for probe, feat_dset in split_dset.items():
            out[split][probe] = {}
            for feats, dset in feat_dset.items():
                out[split][probe][feats] = {}
                for clf_name, trained_clf in trained_classifiers[probe][feats].items():
                    if verbose:
                        print(split, probe, feats, clf_name)
                    out[split][probe][feats][clf_name] = {}
                    out[split][probe][feats][clf_name]["true"] = dset["y"]
                    out[split][probe][feats][clf_name]["pred"] = trained_clf.predict(
                        dset["X"]
                    )
    return out


def add_human_classifier(
    predictions, dsets, hc_col="mh_structure_org_score", hc_name="Melissa"
):
    for split, pred_splits in predictions.items():
        for probe, pred_probe in pred_splits.items():
            for feats, pred_feats in pred_probe.items():
                predictions[split][probe][feats][hc_name] = {
                    "pred": dsets[split][probe][feats]["adata"].obs[hc_col].values,
                    "true": dsets[split][probe][feats]["y"],
                }
    return predictions


def evaluate_predictions(predictions):
    df_clf_report_ave = pd.DataFrame()
    for split, pred_splits in predictions.items():
        for probe, pred_probe in pred_splits.items():
            for feats, pred_feats in pred_probe.items():
                for clf_name, pred_clf in pred_feats.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UndefinedMetricWarning)
                        cr_dict = classification_report(
                            pred_clf["true"], pred_clf["pred"], output_dict=True
                        )
                        cr_dict_ave = {
                            k: v
                            for k, v in cr_dict.items()
                            if k in ["macro avg", "weighted avg"]
                        }
                        df_report_ave = (
                            pd.DataFrame(cr_dict_ave)
                            .transpose()
                            .reset_index()
                            .rename({"index": "Reduction"}, axis="columns")
                        )
                        df_report_ave["Split"] = split
                        df_report_ave["Features"] = feats
                        df_report_ave["Probe"] = probe
                        df_report_ave["Classifier"] = clf_name
                        df_clf_report_ave = df_clf_report_ave.append(
                            df_report_ave, sort=True
                        ).reset_index(drop=True)
                        df_clf_report_ave["Classifier type"] = (
                            df_clf_report_ave["Classifier"]
                            .apply(lambda x: x.split()[0])
                            .replace(
                                {
                                    "Logistic": "Logistic regression",
                                    "Random": "Random Forest",
                                }
                            )
                        )
    return df_clf_report_ave
