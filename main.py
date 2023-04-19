import json
import pandas as pd
import numpy as np
import seaborn as sns
import argparse

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, hinge_loss, precision_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from ANN import ANN

import torch
import torch.nn as nn

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="How do you want to analyze the Iris dataset?")
    
    parser.add_argument('-g', '--graph', action='store_true', help='specify if you want to graph the dataset', required=False)
    parser.add_argument('-m', '--model', type=str, help='specify model to implement')
    args = parser.parse_args()
    plot_vis = args.graph
    model_type = args.model
    
    # convert JSON to CSV for easier preprocessing
    with open('iris.json', encoding='utf-8') as file:
        data = json.loads(file.read())
        
    df = pd.json_normalize(data)
    
    df.to_csv('iris.csv', index=False, encoding='utf-8')
    
    data = pd.read_csv('iris.csv')
    
    # map flower types to number classifiers
    classes = {
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2
    }
    
    data['species'] = data['species'].apply(lambda x: classes[x])
    
    # plotting
    if (plot_vis):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13,9))
        fig.tight_layout()

        plots = [(0,1),(2,3),(0,2),(1,3)]
        colors = ["r", "g", "b"]
        labels = ["setosa","virginica","versicolor"]

        for i, ax in enumerate(axes.flat):
            for j in range(3):
                x = data.columns[plots[i][0]]
                y = data.columns[plots[i][1]]
                ax.scatter(data[data["species"]==j][x], data[data["species"]==j][y], color=colors[j])
                ax.set(xlabel=x, ylabel=y)

        fig.legend(labels=labels, loc=4, bbox_to_anchor=(1.0,0.85))
        plt.show()
    
    # splitting the dataset
    X = data.drop("species",axis=1).values
    y = data["species"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

    # Method 1: Artificial neural network
    if model_type == "ANN":
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        model = ANN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        epochs = 100
        losses = []
        
        for i in range(epochs):
            y_pred = model.forward(X_train)
            loss = criterion(y_pred, y_train)
            losses.append(loss.item())
            print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # plotting the losses
        plt.plot(range(epochs), losses)
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.show()
    
        preds = []
        with torch.no_grad():
            for val in X_test:
                y_hat = model.forward(val)
                preds.append(y_hat.argmax().item())
                
        df = pd.DataFrame({'Y': y_test, 'YHat': preds})
        df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
        accuracy = df['Correct'].sum() / len(df)
        
        print(accuracy)
    
    # Method 2: Logistic Regression
    elif model_type == "logreg":
        # Define logistic regression model
        model = LogisticRegression(max_iter=1000)

        # Train the model
        num_epochs = 100
        for epoch in range(num_epochs):
            model.fit(X_train, y_train)
            loss = log_loss(y_train, model.predict_proba(X_train))
            accuracy = accuracy_score(y_test, model.predict(X_test))
            print("Epoch:", epoch+1, "Loss:", loss, "Accuracy:", accuracy)
    
    # Method 3: Support Vector Machine (SVM)
    elif model_type == "SVM":
        # Define SVM model
        model = SVC(kernel='linear')

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate hinge loss
        loss = hinge_loss(y_test, model.decision_function(X_test))

        print("Accuracy:", accuracy)
        print("Hinge Loss:", loss)
    
    # Method 4: Decision Tree
    elif model_type == "dt":
        # Define decision tree model
        model = DecisionTreeClassifier()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)
    
    # Method 5: Random Forest
    elif model_type == "rf":
        # Define random forest model
        model = RandomForestClassifier(n_estimators=100)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)
    
    # Method 6: K-nearest neighbors (KNN)
    elif model_type == "knn":
        # Define the KNN classifier and fit it to the training data
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train, y_train)

        # Evaluate the accuracy of the classifier on the testing data
        accuracy = knn_classifier.score(X_test, y_test)
        print("Accuracy:", accuracy)
    
    # Method 7: XGBoost Classifier
    elif model_type == "xgboost":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        #paramaters 
        param = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': 3}  # the number of classes that exist in this datset
        num_round = 5  # the number of training iterations
        
        # model builing using training data
        bst = xgb.train(param, dtrain, num_round)
        
        preds = bst.predict(dtest)
        best_preds = np.asarray([np.argmax(line) for line in preds])
        print(precision_score(y_test, best_preds, average='macro'))
        
        # tree
        xgb.plot_tree(bst, num_trees=1)
        fig = plt.gcf()
        fig.set_size_inches(150, 100)
        fig.savefig('treeIris.png') 
        
        # feature importance
        plot_importance(bst)
        pyplot.show()
            
    else:
        raise Exception("Model not defined")
    
if __name__ == '__main__':
    main()