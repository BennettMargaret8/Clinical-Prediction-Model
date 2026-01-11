import lightgbm as lgb
from catboost import CatBoostClassifier
import pandas as pd
import os

A_dir = os.getcwd() + os.sep + 'model' + os.sep + "A_model.txt"
B_dir = os.getcwd() + os.sep + 'model' + os.sep + "B_model.json"
subA1_dir = os.getcwd() + os.sep + 'model' + os.sep + "A1_model.json"
subA2_dir = os.getcwd() + os.sep + 'model' + os.sep + "subA2_model.txt"

A_clf = lgb.Booster(model_file=A_dir)

B_clf = CatBoostClassifier()
B_clf.load_model(B_dir, 'json')

subA1_clf = CatBoostClassifier()
subA1_clf.load_model(subA1_dir, 'json')

subA2_clf = lgb.Booster(model_file=subA2_dir)


def Agroup_predict(val):
    cols = ['modality',
            'therapy',
            'Gender',
            'Comorbidity',
            'Etiology',
            'diameter',
            'number',
            'invasion',
            'Metastasis',
            'BCLC',
            'ALB',
            'ALT',
            'AST',
            'TBIL',
            'ALBI',
            'INR',
            'PLT',
            'CRP',
            'Cre',
            'Neu',
            'Ly',
            'NLR',
            'AFP',
            'PIVKA']
    X_test = []
    for col in cols:
        X_test.append(int(val[col]))
    # X_test = np.array(X_test).reshape((-1, 1))
    # df = pd.DataFrame(data=X_test, columns=cols)
    df = pd.DataFrame([X_test])
    res = A_clf.predict(df)[0]
    return int(round(res, 4) * 10000)


def Bgroup_predict(val):
    cols = ['modality',
            'therapy',
            'Age',
            'Gender',
            'ECOG',
            'Comorbidity',
            'Cirrhosis',
            'Ascites',
            'diameter',
            'number',
            'invasion',
            'Metastasis',
            'BCLC',
            'ALB',
            'ALT',
            'AST',
            'TBIL',
            'ALBI',
            'PT',
            'INR',
            'PLT',
            'CRP',
            'Cre',
            'Neu',
            'Ly',
            'NLR',
            'PLR',
            'SII',
            'AFP',
            'PIVKA']
    X_test = []
    for col in cols:
        X_test.append(int(val[col]))
    res = B_clf.predict_proba(X_test)[1]
    return int(round(res, 4) * 10000)


def subA1group_predict(val):
    cols = ['therapy',
            'Gender',
            'diameter',
            'number',
            'invasion',
            'Metastasis',
            'BCLC',
            'ALB',
            'ALT',
            'AST',
            'TBIL',
            'Cre',
            'Neu',
            'Ly',
            'AFP']
    X_test = []
    for col in cols:
        X_test.append(int(val[col]))
    res = subA1_clf.predict_proba(X_test)[1]
    return int(round(res, 4) * 10000)

def subA2group_predict(val):
    cols = ['therapy',
            'Age',
            'Gender',
            'Comorbidity',
            'Etiology',
            'Ascites',
            'diameter',
            'number',
            'invasion',
            'Metastasis',
            'BCLC',
            'ALB',
            'ALT',
            'AST',
            'TBIL',
            'ALBI',
            'INR',
            'PLT',
            'CRP',
            'Neu',
            'Ly',
            'NLR',
            'AFP',
            'PIVKA']
    X_test = []
    for col in cols:
        X_test.append(int(val[col]))
    df = pd.DataFrame([X_test])
    res = subA2_clf.predict(df)[0]
    return int(round(res, 4) * 10000)


def hnzl_predict(model, val):
    if model == "a":
        return Agroup_predict(val)
    elif model == "b":
        return Bgroup_predict(val)
    elif model == "suba1":
        return subA1group_predict(val)
    elif model == "suba2":
        return subA2group_predict(val)
    



