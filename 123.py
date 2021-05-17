import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid

df = pd.read_csv('BRCA_pam50.tsv', sep='\t', index_col=0)
X = df.iloc[:, :-1].to_numpy()
y = df['Subtype'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=17)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=1, weights="distance", p=2))
    ])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))
cl = np.unique(y)
ac = [np.mean([model.predict(X_test)[y_test == i] == i]) for i in cl]
df = pd.DataFrame({'class': cl, 'accuracy':ac})
df = df.set_index('class')
print(df)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", NearestCentroid())
    ])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))
cl = np.unique(y)
ac = [np.mean([model.predict(X_test)[y_test == i] == i]) for i in cl]
df = pd.DataFrame({'class': cl, 'accuracy':ac})
df = df.set_index('class')
print(df)