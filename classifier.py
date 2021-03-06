import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from time import sleep


#FUNCOES AUX
def transform_results(dataFrame):
    resultsHome = []
    resultsTie = []
    resultsAway = []
    for e in dataFrame:
        if e == 'H':
            resultsHome += [1]
            resultsTie += [0]
            resultsAway += [0]
        elif e == 'A':
            resultsHome += [0]
            resultsTie += [0]
            resultsAway += [1]
        else:
            resultsHome += [0]
            resultsTie += [1]
            resultsAway += [0]
    df = pd.DataFrame(list(zip(resultsHome,resultsTie, resultsAway)))
    return df
#FUNCOES AUX








#importar os dataframes 
# X_train = pd.read_csv("./breast_cancer_examples/xtrain.csv", header=None)  #X=inputs
# Y_train = pd.read_csv("./breast_cancer_examples/ytrain.csv", header=None)  #Y=results
# X_test = pd.read_csv("./breast_cancer_examples/xtest.csv", header=None)
# Y_test = pd.read_csv("./breast_cancer_examples/ytest.csv", header=None)

ligaNOS2019 = pd.read_csv("./ligaNOS/portugal-liga-i-2018-to-2019.csv", low_memory=False)
ligaNOS2020 = pd.read_csv("./ligaNOS/portugal-liga-i-2019-to-2020.csv", low_memory=False)

ligaNOS2019Results = ligaNOS2019["FTR"]  #full time Result   H-home D-draw A-away
ligaNOS2020Results = ligaNOS2020["FTR"]  


#transforming H,D,H to 1 0 0(Home)  0 1 0(Draw)  0 0 1(Away)
ligaNOS2019Results = transform_results(ligaNOS2019Results) 
ligaNOS2020Results = transform_results(ligaNOS2020Results)  



ligaNOS2019.drop("FTR", axis=1, inplace=True)
ligaNOS2019.drop("HTR", axis=1, inplace=True)
ligaNOS2019.drop("Div", axis=1, inplace=True)
ligaNOS2019.drop("Date", axis=1, inplace=True)
ligaNOS2019.drop("HomeTeam", axis=1, inplace=True)
ligaNOS2019.drop("AwayTeam", axis=1, inplace=True)

ligaNOS2020.drop("FTR", axis=1, inplace=True)
ligaNOS2020.drop("HTR", axis=1, inplace=True)
ligaNOS2020.drop("Div", axis=1, inplace=True)
ligaNOS2020.drop("Date", axis=1, inplace=True)
ligaNOS2020.drop("HomeTeam", axis=1, inplace=True)
ligaNOS2020.drop("AwayTeam", axis=1, inplace=True)

#initialize the ANN
classifier = Sequential()

classifier.add(Dense(units = 16, activation="relu", input_dim=30)) #supondo que ha 30 features, ir reduzindo ate ter 3 outputs se o Ytrain tiver tres colunas
classifier.add(Dense(units = 8, activation="relu"))
classifier.add(Dense(units = 6, activation="relu"))
classifier.add(Dense(units = 3, activation="sigmoid"))  #sigmoid vai obrigar a devolver valores entre 0 e 1

classifier.compile(optimizer = "rmsprop", loss = "binary_crossentropy")

#training
classifier.fit(ligaNOS2019, ligaNOS2019Results, batch_size = 1, epochs = 250)
# classifier.fit(X_train, Y_train, batch_size = 1, epochs = 100)

sleep(1000)

#predicting
Y_pred = classifier.predict(X_test)
Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred ]  #ESTAO A TRANSFORMAR TODOS OS RESULTADOS EM 0 OU 1 PQ O Y_TEST DELES SO TEM ESSES RESULTADOS

#ESTATISTICAS DOS RESULTADOS (mais uma vez depende se o Y (output) tem apenas uma coluna ou 3 )
total = 0
correct = 0
wrong = 0
for i in range(len(Y_pred)):
    total += 1
    if(Y_test.at[i,0] == Y_pred[i]):
        correct += 1
    else:
        wrong += 1

print("Total: " + str(total))
print("Correct: " + str(correct))
print("Wrong: " + str(wrong))
print("Accuracy: " + str(correct/total)) 




