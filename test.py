import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ file csv vào dataframe
df = pd.read_csv('data.csv',delimiter=',')

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X = df.drop('class', axis=1)
X=(X-X.min())/(X.max()-X.min()) #Chuẩn hoá dữ liệu
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Huấn luyện mô hình rừng ngẫu nhiên
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
rf.fit(X_train, y_train)

# Huấn luyện mô hình Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Độ chính xác thuật toán rừng ngẫu nhiên:", accuracy_rf)

y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Độ chính xác thuật toán Bayes:", accuracy_nb)
