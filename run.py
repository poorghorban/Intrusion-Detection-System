import numpy as np 
import matplotlib.pyplot as plt
import warnings

import config

from datasets import load_NSL_KDD

from preprocessing import one_hot_encoding , label_encoder , min_max_scaler
from util import resample_data
from feature_extraction import principal_component_analysis

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score , classification_report,confusion_matrix,ConfusionMatrixDisplay


warnings.simplefilter("ignore")


def get_features():
    ## Load dataset 
    print("Reading data...")
    X_train , y_train,_ = load_NSL_KDD(config.path_Train_NSL_KDD)
    X_test , y_test,_ = load_NSL_KDD(config.path_Test_NSL_KDD)
    
    ## Balancing train data 
    if config.balance_data:
        print("Balancing train dataset...")
        X_train , y_train = resample_data(X_train , y_train)

    ## Preprocessing train and test dataset 
    print("Preprocessing data...")
    # OneHotEncodeing for categorical variable 
    X_train , X_test = one_hot_encoding(X_train , X_test , [1,2,3])
    # LabelEncodeing for labels
    y_train , y_test, label_names = label_encoder(y_train , y_test)
    # Min Max Scaler - range [0,1]
    if config.normalize:
        X_train = min_max_scaler(X_train)
        X_test = min_max_scaler(X_test)
    ## Extract feature 
    print("Extract features by PCA method...")
    X_train , X_test = principal_component_analysis(X_train , X_test , config.max_feature)

    ## return Train and Test sets 
    return X_train , X_test , y_train , y_test , label_names

def best_params_gridsearchcv(X,y,model , parameters):
    gsc = GridSearchCV(model, parameters)
    gsc.fit(X,y)
    print("estimate best parameters[{} - {:.2f}]".format(gsc.best_params_,gsc.best_score_))
    return gsc.best_params_

def train_model(X , y):
    if config.model_SVM:
        print("Training SVM...")
        if config.gridsearch:
            parameters = {'C':[1,5,10] , 'kernel':[config.kernel] , 'gamma':[config.gamma]}
            best_param = best_params_gridsearchcv(X , y , SVC() , parameters)
            model = SVC(kernel=best_param['kernel'], C=best_param['C'] , gamma=best_param['gamma'])
        else:
            model = SVC(kernel=config.kernel, C=config.C , gamma=config.gamma)
    elif config.model_NaiveBayes:
        print("Training Naive Bayes...")
        model = GaussianNB()
    elif config.model_J48:
        print("Training J48 (Decision Tree C4.5)...")
        if config.gridsearch:
            parameters = {'max_depth':[5,10]}
            best_param = best_params_gridsearchcv(X , y , DecisionTreeClassifier() , parameters)
            model = DecisionTreeClassifier(random_state=0 , max_depth=best_param['max_depth'])
        else:
            model = DecisionTreeClassifier(random_state=0 , max_depth=config.max_depth)
    else:
        print("Did not select the training model!!!")
        raise FileNotFoundError('Did not select the training model - select model in config.py file')

    model.fit(X_train , y_train)

    ## return model 
    return model 

def evaluate(X , y , model , title , label_names , path):
    print("Evaluating the results...")
    ## Predict label 
    y_pred = model.predict(X)

    ## Calculate accuracy score 
    acc = accuracy_score(y , y_pred)
    print('accuracy:{:.2f}'.format(acc))

    ## Calculate classification report 
    classifiction_rep = classification_report(y , y_pred , target_names=label_names)
    print(classifiction_rep)

    ## Confusion matrix
    cm = confusion_matrix(y, y_pred, normalize='true')
    cmd =ConfusionMatrixDisplay(cm, display_labels=label_names)
    cmd = cmd.plot(cmap=plt.get_cmap('Blues'))
    cmd.ax_.set_title(title)

    ## Save plot 
    plt.savefig(path)

    ## Show plot 
    plt.show()

if __name__ == "__main__":
    ## get features 
    X_train , X_test , y_train , y_test , label_names = get_features()
    ## train model 
    model = train_model(X_train , y_train)
    ## evaluate model 
    evaluate(X_test , y_test , model ,
            'Intrusion Detection System (SVM)',
            label_names ,
            'result\confusion_matrix_SVM_balanced.png')