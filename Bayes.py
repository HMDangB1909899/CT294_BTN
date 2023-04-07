import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv',delimiter=',')

X = df.drop('class', axis=1)
X=(X-X.min())/(X.max()-X.min()) #Chuẩn hoá dữ liệu
y = df['class']
X=X.values
y=y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

#Huấn luyện mô hình
nb = GaussianNB()
nb.fit(X_train, y_train)

#Ma trận nhầm lẫn
thucte= y_test
dubao= nb.predict(X_test)
mtx=confusion_matrix(thucte,dubao)
print(mtx)

#Độ chính xác mô hình
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Độ chính xác thuật toán Bayes:", accuracy_nb)

# t=nb.predict([[0.10,0.34,0.22,0.12,0.13,0.50,0.71,0.55,0.04,0.44]])
# print(t)