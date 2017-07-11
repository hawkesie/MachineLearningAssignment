import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, naive_bayes, svm, tree
from sklearn.cross_validation import cross_val_score

'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls different functions to perform the required tasks.

'''

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (7521922, 'Jordan', 'Hawkes'), (7561555, 'Stewart', 'Whitehead')]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def prepare_dataset():
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of c
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    ##         "INSERT YOUR CODE HERE"    
    #Retrieve the data from the data file
    records=pd.read_csv("medical_records.data", header=None)
    #Remove the first column from the datafile which contains the id's of the rows
    records.drop([0],1,inplace=True)
    
    #Place the second coumn of the data into a numpy array
    z= np.array(records[1])
    
    #Declare a numpy array
    y=np.array([])
    
    #Go through the second column in the dataset and convert the M's to 1's and B's to )'s and append 
    #to the y numpy array
    for item in z:
        if item == 'M':
            y= np.append(y,1)
        if item == 'B':
            y= np.append(y,0)

    #Place the remainder of the dataset in the X numpy array without the first and second columns
    x=np.array(records.drop([1],1))
   
    #return the x and y arrays 
    return (x,y)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


#Naive bayes classifier
def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"   
    #declare clf as a NB classifier
    clf=naive_bayes.GaussianNB()
    
    #train the naive bayes classifier
    clf.fit(X_training, y_training)
    
    #return the trained naive bayes classifier
    return clf
    
#Return cross validation score for Decision Tree 
def validateDT():
    dt = tree.DecisionTreeClassifier()
    valScore = cross_val_score(dt, x,y, cv=10, scoring='accuracy')
    return valScore.mean()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Decision Tree classifier
def build_DT_classifier(X_testing, y_testing):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE" 
    
    #best_depth=0
    #best_score=0
    #for each in range(1,1000):
        
    #clf=tree.DecisionTreeClassifier()#(max_depth) removed from brackets so algorithm runs normally
    
    #train the DT classifier
      #  clf.fit(X_training, y_training)
        #if(clf.score(X_test,y_test)>best_score):
        #        best_depth=each
        
    #declare clf as a DT classifier
    clf=tree.DecisionTreeClassifier()
    clf.fit(X_training, y_training)
    #print (best_depth)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 #function to validate nearest neighbors classifier
def validateNN():
    k_scores=0
    for k in range(1,10):
        knn = neighbors.KNeighborsClassifier(k)
        scores = cross_val_score(knn, x,y, cv=10, scoring='accuracy')
        if scores.mean()>k_scores:
            k_scores=scores.mean()
    return (k_scores)

#nearest Neighbor classifier
def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE" 
    accuracy = 0
    #loop through values of k
    for x in range(0, 10):
        #declare the classsifier
        cl=neighbors.KNeighborsClassifier((x*2+1))
        
        #train the classifier
        cl.fit(X_training, y_training)
        
        #set clf as the most successful classifier
        if (cl.score(X_test, y_test) >= accuracy):
            accuracy = cl.score(X_test, y_test)
            
            clf=neighbors.KNeighborsClassifier((x*2+1))
            #train the classifier
            clf.fit(X_training, y_training)
    #return the classifier
    return clf

    


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Return cross validation score for SVM
def validateSVM():
    sv = svm.SVC(kernel='linear')
    valScore = cross_val_score(sv, x,y, cv=10, scoring='accuracy')
    return valScore.mean()

#Support Vector Machine Classifier
#Support Vector Machine Classifier
def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"   
    #declare clf as a SVM classifier
    svmKernels=['linear']# , 'rbf', 'sigmoid'] #remove comment to test using all kernel types
    score = 0
    kernelType = ''
    for kernels in svmKernels:
        
        clf=svm.SVC(kernel=kernels)
    
        #train the SVM classifier
        clf.fit(X_training, y_training)
        accuracy = clf.score(X_test, y_test)
        if (accuracy > score):
            score = accuracy
            kernelType = kernels
    
    #return the classifierS
    clf=svm.SVC(kernel=kernelType)
    clf.fit(X_training, y_training)
    print ("SVM kernel type = "+kernelType)
    return clf



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # call your functions here
#Call the prepare dataset function and store the arrays it returns as X and Y
(x,y)=prepare_dataset()


X_training, X_test, y_training, y_test = cross_validation.train_test_split(x,y,test_size=0.20)    

    

#calling the Naive Bayes classifer on the test data
NB = build_NB_classifier(X_test, y_test)
#Retrieving the accuracy score of the Naive Bayes classifier
accuracy = NB.score(X_test, y_test)
print('NB',accuracy)

#calling the Decision Tree classifer on the test data
dtVal= validateDT()
print('DT Validator Score', dtVal)
DT = build_DT_classifier(X_test, y_test)
#Retrieving the accuracy score of the Decision Tree classifier
accuracy = DT.score(X_test, y_test)
print('DT Test Score',accuracy)


svVal=validateSVM()
print ('SVM Validator Score',svVal)
#calling the SUpport Vector Machine classifer on the test data
SVM = build_SVM_classifier(X_training, y_training)
#Retrieving the accuracy score of the Support Vector Machine classifier
accuracy = SVM.score(X_test, y_test)
print('SVM Test Score',accuracy)

#calling the Nearest Neighbour classifer on the test data
nnVal= validateNN()
print ('NN Validator Score ',nnVal)
NN = build_NN_classifier(X_training, y_training)

#Retrieving the accuracy score of the Nearest Neighbour classifier
accuracy = NN.score(X_test, y_test)
print('NN Test Score',accuracy)

