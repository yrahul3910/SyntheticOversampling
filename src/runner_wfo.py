import pandas as pd
import csv
import os
import sys
import time
import numpy as np
import random
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import evaluate_result, read_data, create_models
from smote_oversampling import RandomOversampling
from smote_oversampling import ADASYNOversampling
from smote_oversampling import BorderlineSMOTEOversampling
from smote_oversampling import SMOTEOversampling
from smote_oversampling import SVMSMOTEOversampling
from smote_oversampling import SMOTUNEDOversampling
from smote_oversampling import WFOOversampling
from dazzle import DAZZLEOversampling
from WGAN import WGANOversampling
from random_projection import RandomProjectionOversampling
from howso_engine import howsoOversampling
from ds_engine import DSOversampling
from sdv_engine import SDVOversampling

def main(project, repeats=10, rp_threshold=12):
    rs_list = random.sample(range(50, 500), repeats)

    for repeat in range(repeats):
        print(f"----- in repeat {repeat+1} -----")
        rs = rs_list[repeat]


        write_path = f"{project}_res_r{repeat+1}_rn{rs}.csv"
        write_path = f"{os.getcwd()}/result/{project}/{write_path}"
        with open(write_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["oversampling_scheme", "runtime", "learner", "acc", "prec", "recall", "fpr", "f1", "auc", "g_score", "d2h"])

        if project != "Ambari_Vuln":
            df = read_data(project)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            print("y value counts: \n", str(y.value_counts()))
            print("y class ratio: 1:", str(round(y.value_counts()[0]/y.value_counts()[1])))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)
            print("--- y train classes count: \n" + str(y_train.value_counts()))
            print("--- y train ratio: 1:" + str(round(y_train.value_counts()[0] / y_train.value_counts()[1])))
            print(" ")
            print("--- y test classes count: \n" + str(y_test.value_counts()))
            print("--- y test ratio: 1:" + str(round(y_test.value_counts()[0] / y_test.value_counts()[1])))
        else:
            train_df, test_df = read_data(project)
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]
            print("--- y train classes count: \n" + str(y_train.value_counts()))
            print("--- y train ratio: 1:" + str(round(y_train.value_counts()[0] / y_train.value_counts()[1])))
            print(" ")
            print("--- y test classes count: \n" + str(y_test.value_counts()))
            print("--- y test ratio: 1:" + str(round(y_test.value_counts()[0] / y_test.value_counts()[1])))

        ### wfo run ###
        print("----- wfo -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = WFOOversampling(X_train=X_train_copy,
                                                          y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train_new)
        X_test_scale = scaler.transform(X_test)

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["WFO", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["WFO", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["WFO", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["WFO", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["WFO", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["WFO", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["WFO", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["WFO", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        print("----- wfov2 -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = WFOOversampling(X_train=X_train_copy,
                                                          y_train=y_train_copy, ultrasample=True)
        
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train_new)
        X_test_scale = scaler.transform(X_test)

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["WFO2", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["WFO2", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["WFO2", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["WFO2", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["WFO2", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["WFO2", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["WFO2", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["WFO2", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        print("----- end of experiment ------")


if __name__ == "__main__":
    case_to_run = sys.argv[1]
    repeats = int(sys.argv[2])
    rp_threshold = int(sys.argv[3])

    main(case_to_run, repeats=repeats ,rp_threshold=rp_threshold)
