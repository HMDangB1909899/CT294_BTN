import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ file csv vào dataframe
df = pd.read_csv('data.csv',delimiter=',')

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X = df.drop('class', axis=1)
X=(X-X.min())/(X.max()-X.min()) #Chuẩn hoá dữ liệu
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Xây dựng mô hình Decision Tree và huấn luyện trên tập huấn luyện
clf = DecisionTreeClassifier(random_state=5)
clf.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác cây quyết định:", accuracy)
