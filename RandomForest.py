import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Đọc dữ liệu từ file csv vào dataframe
df = pd.read_csv('data.csv',delimiter=',')

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X = df.drop('class', axis=1)
X=(X-X.min())/(X.max()-X.min()) #Chuẩn hoá dữ liệu
y = df['class']
X=X.values
y=y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Huấn luyện mô hình rừng ngẫu nhiên
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
rf.fit(X_train, y_train)

thucte= y_test
dubao= rf.predict(X_test)
mtx=confusion_matrix(thucte,dubao)
print(mtx)

y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Độ chính xác thuật toán rừng ngẫu nhiên:", accuracy_rf)
# t=rf.predict([[0.10,0.34,0.22,0.12,0.13,0.50,0.71,0.55,0.04,0.44]])
# print(t)

