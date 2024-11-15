#ALGORITHM PLANING
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

import warnings
warnings.filterwarnings("ignore") #kütüphane versiyonları ile ilgili uyarıları kapatmak için kullanılır


#load dataset ve EDA
heart_disease_data_path = ("C:/Users/koralsenturk/AppData/Local/anaconda3/SagliktaYapayZeka/3_kodlar_verisetleri/4_SagliktaMakineOgrenmesiUygulamalari/heart_disease_uci.csv")
df = pd.read_csv(heart_disease_data_path)
df = df.drop(columns=["id"])
# df.info()
describe = df.describe()
# print(describe)
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure()
sns.pairplot(df, vars=numerical_features, hue = "num") #korelasyon analizi yapabilmek adına görselleştirilir
# plt.show()
plt.figure()
sns.countplot(x = "num", data=df) #num kolonu(target değişkeni) bazında diğer kolonların sayısının analizi
# plt.show()


#handling missing value
# print(df.isnull().sum())
df = df.drop(columns="ca") #categorik bir veri ve 611 satır eksik yani gereksiz
df["trestbps"].fillna(df["trestbps"].median(), inplace = True) #sayısal verinin boş satırlarını median değeri ile doldur
df["chol"].fillna(df["chol"].median(), inplace = True)
df["fbs"].fillna(df["fbs"].mode()[0], inplace = True) #true false verisi içeren kolonda en fazla olan veri ne ise onunla doldurulacak
df["restecg"].fillna(df["restecg"].mode()[0], inplace = True)
df["thalch"].fillna(df["thalch"].median(), inplace = True)
df["thalch"].fillna(df["thalch"].median(), inplace = True)
df["exang"].fillna(df["exang"].mode()[0], inplace = True)
df["oldpeak"].fillna(df["oldpeak"].median(), inplace = True)
df["slope"].fillna(df["slope"].mode()[0], inplace = True)
df["thal"].fillna(df["thal"].mode()[0], inplace = True)
print(df.isnull().sum())


#train test split
X = df.drop(["num"], axis = 1)
y = df["num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

categorical_features = ["sex", "dataset", "cp","restecg", "exang", "slope", "thal"]
numerical_features = ["age", "trestbps", "chol","fbs", "thalch", "oldpeak"]

X_train_num = X_train[numerical_features]
X_test_num = X_test[numerical_features]


#standardizasyon
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)


#kategorik kodlama (label encoding)
encoder = OneHotEncoder(sparse=False, drop="first") #categorik sütunu sayısal verilere çevirir
X_train_cat = X_train[categorical_features]
X_test_cat = X_test[categorical_features]
X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

X_train_transformed = np.hstack((X_train_cat_encoded, X_train_num_scaled))
X_test_transformed = np.hstack((X_test_cat_encoded, X_test_num_scaled))


#modeling: RF, KNN, Voting Classifier train ve test
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier()

voting_clf = VotingClassifier(estimators=[("rf", rf), ("knn", knn)], voting="soft")

voting_clf.fit(X_train_transformed, y_train)
y_pred = voting_clf.predict(X_test_transformed)



#CM (Confsing Matrix)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("confusion Matrix: ")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))