# path Train NSL-KDD dataset
path_Train_NSL_KDD = "data\KDDTrain+.txt"

# path Test NSL-KDD dataset
path_Test_NSL_KDD = "data\KDDTest+.txt"

# balancing train data 
balance_data = False

# normlized train and test (Min Max Scaler [0,1])
normalize = True

# number of  features extract with pca method
max_feature = 6

# type model for training 
model_J48 = True  # Decision Tree C4.5
model_SVM = False
model_NaiveBayes = False

# set parameters model automatic
gridsearch = False

# params J48
max_depth = 10

# params SVM 
kernel = 'rbf'
C = 1
gamma = 'auto'

