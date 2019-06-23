import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# a function that uses one hot encoder to encode categorical columns
def categoryToBinary(dataFrame , columnName):
    enc = OneHotEncoder(categories='auto')
    categories = dataFrame[columnName].unique()
    categories.sort()
    categories = [[i] for i in categories ]
    enc.fit(categories)
    values =dataFrame[columnName].values.tolist()
    values= [[i] for i in values]
    Encoded= pd.DataFrame( enc.transform(values).toarray()
                          ,columns=[str(columnName)+"_"+str(cat[0]) for cat in categories]
                           )
    return Encoded

# loading data from CSV file 
trainDf = pd.read_csv("train.csv")
testDf = pd.read_csv("test.csv")

# Merginf the two daasaet to do some data cleaning
testDf["Survived"]=None
trainDf["isTrainingSet"]=True
testDf["isTrainingSet"]=False
fullDf=pd.concat([trainDf,testDf] , ignore_index=False , axis=0 , sort=False)

# Filling missing values in the data set
fullDf["Embarked"]= fullDf["Embarked"].fillna('S')
fullDf.Age = fullDf.Age.fillna(fullDf["Age"].median()) 
fullDf.Fare = fullDf.Fare.fillna(fullDf["Fare"].median()) 

# spliting the dataset again to start model fitting process
trainDf =fullDf[fullDf.isTrainingSet==True] 
testDf =fullDf[fullDf.isTrainingSet==False] 
# Setting model data sets
target= trainDf["Survived"]
features= trainDf[["Pclass","Sex","Age","Fare","Embarked"]]
Test = testDf[["Pclass","Sex","Age","Fare","Embarked"]]

## encoding sex in Training dataset to numerical vals so scikit can use it
sexToBinary = categoryToBinary(features, "Sex" )
features = pd.concat([features,sexToBinary] ,axis= 1)
features = features.drop("Sex",axis = 1)

## encoding Embarked in Training dataset to numerical vals so scikit can use it
embarkedToBinary = categoryToBinary(features, "Embarked" )
features = pd.concat([features,embarkedToBinary] ,axis= 1)
features = features.drop("Embarked",axis = 1)

## encoding Pclass in Training dataset to numerical vals so scikit can use it
pclassToBinary = categoryToBinary(features, "Pclass" )
features = pd.concat([features,pclassToBinary] ,axis= 1)
features = features.drop("Pclass",axis = 1)

## encoding sex in Test dataset to numerical vals so scikit can use it
sexToBinary = categoryToBinary(Test, "Sex" )
Test = pd.concat([Test,sexToBinary] ,axis= 1)
Test = Test.drop("Sex",axis = 1)

## encoding Embarked in Test dataset to numerical vals so scikit can use it
embarkedToBinary = categoryToBinary(Test, "Embarked" )
Test = pd.concat([Test,embarkedToBinary] ,axis= 1)
Test = Test.drop("Embarked",axis = 1)

## encoding Pclass in Test dataset to numerical vals so scikit can use it
pclassToBinary = categoryToBinary(Test, "Pclass" )
Test = pd.concat([Test,pclassToBinary] ,axis= 1)
Test = Test.drop("Pclass",axis = 1)

# Fitting random forest model to the data
titanicModel = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=1)
titanicModel.fit(features,target.astype('int'))

# Fitting Decission tree model to the data
titanicModel2 =DecisionTreeRegressor(random_state=1)
titanicModel2.fit(features,target.astype('int'))

#predict using (RandomForest) and saving results 
result= testDf[["PassengerId","Survived"]]
p=pd.DataFrame(titanicModel.predict(Test),columns=["Survived"])
result["Survived"] = p["Survived"]
#Writing results into new CSV file
result.to_csv(r'./RandomForest1.csv',index = None, header=True)

#predict using and (Decision tree) saving results                                      
result= testDf[["PassengerId","Survived"]]
p=pd.DataFrame(titanicModel2.predict(Test),columns=["Survived"])
result["Survived"] = p["Survived"].apply( lambda x: round(x))

#Writing results into new CSV file
result.to_csv(r'./Decisiontree1.csv',index = None, header=True)

print(result)
