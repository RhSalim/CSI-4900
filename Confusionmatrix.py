import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_clean_data(temp_file_path, coord_file_path):

    temp_data = pd.read_csv(temp_file_path, sep='\t', header=None)
    coord_data = pd.read_csv(coord_file_path, sep='\t', header=None)

    temp_data_cleaned = temp_data.drop(index=0).rename(columns={0: 'Node', 1: 'Temperature'})
    coord_data_cleaned = coord_data.drop(index=0).rename(columns={0: 'Node', 1: 'Unknown', 2: 'X', 3: 'Y', 4: 'Z'})


    temp_data_cleaned['Temperature'] = temp_data_cleaned['Temperature'].str.replace(',', '.').astype(float)
    coord_data_cleaned['X'] = coord_data_cleaned['X'].astype(float)
    coord_data_cleaned['Y'] = coord_data_cleaned['Y'].astype(float)
    coord_data_cleaned['Z'] = coord_data_cleaned['Z'].astype(float)

    return temp_data_cleaned, coord_data_cleaned

def create_classification_target(temp_data_cleaned, threshold=30):
    temp_data_cleaned['Temp_Category'] = (temp_data_cleaned['Temperature'] > threshold).astype(int) 
    return temp_data_cleaned

def train_random_forest_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def main(temp_file_path, coord_file_path):
    temp_data_cleaned, coord_data_cleaned = load_and_clean_data(temp_file_path, coord_file_path)
    temp_data_with_target = create_classification_target(temp_data_cleaned)
    combined_data = pd.merge(temp_data_with_target, coord_data_cleaned, on='Node')

    X = combined_data[['X', 'Y', 'Z']] 
    y = combined_data['Temp_Category'] 


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    clf = train_random_forest_classifier(X_train, y_train)

    evaluate_classifier(clf, X_test, y_test)

temp_file_path = "C:/Users/Rhola/Desktop/CSI 4900/Temp_noeuds1.txt"
coord_file_path = "C:/Users/Rhola/Desktop/CSI 4900/Coord_noeuds1.txt"

main(temp_file_path, coord_file_path)
