import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load & check the data
MNIST_HoYin = fetch_openml('mnist_784', version=1)
print("Keys:", MNIST_HoYin.keys())
copy = MNIST_HoYin

X_HoYin, y_HoYin = MNIST_HoYin["data"], MNIST_HoYin["target"]
print("Type of X:", type(X_HoYin))
print("Type of y:", type(y_HoYin))
print("Shape of X:", X_HoYin.shape)
print("Shape of y:", y_HoYin.shape)
some_digit1 = X_HoYin.loc[7].values
some_digit2 = X_HoYin.loc[5].values
some_digit3 = X_HoYin.loc[0].values
some_digit1_image = some_digit1.reshape(28, 28)
some_digit2_image = some_digit2.reshape(28, 28)
some_digit3_image = some_digit3.reshape(28, 28)
plt.imshow(some_digit1_image, cmap=mpl.cm.binary)
plt.show()
plt.imshow(some_digit2_image, cmap=mpl.cm.binary)
plt.show()
plt.imshow(some_digit3_image, cmap=mpl.cm.binary)
plt.show()

# Pre-process the data 
y_HoYin = y_HoYin.astype(np.uint8)
classes = np.where(y_HoYin < 4, 0, (np.where(y_HoYin > 6, 9, 1)))
freq = {}
for i in range(len(classes)):
  if classes[i] in freq.keys():
    freq[classes[i]] += 1
  else:
    freq[classes[i]] = 1

print(freq)
keys = list(freq.keys())
ls = []
for i in keys:
  ls.append(str(i))
plt.bar(ls, freq.values())
plt.show()
X_HoYin_train, X_HoYin_test, y_HoYin_train, y_HoYin_test = X_HoYin[:50000], X_HoYin[50000:], y_HoYin[:50000], y_HoYin[50000:]

# Bulid Claasification Models
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

NB_clf_HoYin = MultinomialNB()
NB_clf_HoYin.fit(X_HoYin_train, y_HoYin_train)
NB_clf_HoYin_score = cross_val_score(NB_clf_HoYin, X_HoYin_train, y_HoYin_train, cv=3, scoring="accuracy")
print(NB_clf_HoYin_score)

NB_clf_HoYin_test_score = NB_clf_HoYin.score(X_HoYin_test,y_HoYin_test)
print(NB_clf_HoYin_test_score)

y_HoYin_train_pred = cross_val_predict(NB_clf_HoYin, X_HoYin_train, y_HoYin_train, cv=3)
NB_confusion_matrix = confusion_matrix(y_HoYin_train, y_HoYin_train_pred)
print(NB_confusion_matrix)
some_digit1_pred = NB_clf_HoYin.predict([some_digit1])
print(some_digit1_pred)
some_digit2_pred = NB_clf_HoYin.predict([some_digit2])
print(some_digit2_pred)
some_digit3_pred = NB_clf_HoYin.predict([some_digit3])
print(some_digit3_pred)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

LR_clf_HoYin_lbfgs = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1200, tol=0.1)
LR_clf_HoYin_lbfgs.fit(X_HoYin, y_HoYin)

LR_clf_HoYin_saga = LogisticRegression(solver="saga", multi_class="multinomial", max_iter=1200, tol=0.1)
LR_clf_HoYin_saga.fit(X_HoYin, y_HoYin)

lbfgs_score = cross_val_score(LR_clf_HoYin_lbfgs, X_HoYin_train, y_HoYin_train, cv=3, scoring="accuracy")
print(lbfgs_score)
saga_score = cross_val_score(LR_clf_HoYin_saga, X_HoYin_train, y_HoYin_train, cv=3, scoring="accuracy")
print(saga_score)

lbfgs_test_score = LR_clf_HoYin_lbfgs.score(X_HoYin_test,y_HoYin_test)
print(lbfgs_test_score)
saga_test_score = LR_clf_HoYin_saga.score(X_HoYin_test,y_HoYin_test)
print(saga_test_score)

y_HoYin_train_lbfgs_pred = cross_val_predict(LR_clf_HoYin_lbfgs, X_HoYin_train, y_HoYin_train, cv=3)
y_HoYin_train_saga_pred = cross_val_predict(LR_clf_HoYin_saga, X_HoYin_train, y_HoYin_train, cv=3)

lbfgs_confusion_matrix = confusion_matrix(y_HoYin_train, y_HoYin_train_lbfgs_pred)
print(lbfgs_confusion_matrix)
saga_confusion_matrix = confusion_matrix(y_HoYin_train, y_HoYin_train_saga_pred)
print(saga_confusion_matrix)

lbfgs_precision_score = precision_score(y_HoYin_train, y_HoYin_train_lbfgs_pred, average='macro')
print(lbfgs_precision_score)
lbfgs_recall_score = recall_score(y_HoYin_train, y_HoYin_train_lbfgs_pred, average='macro')
print(lbfgs_recall_score)
saga_precision_score = precision_score(y_HoYin_train, y_HoYin_train_saga_pred, average='macro')
print(saga_precision_score)
saga_recall_score = recall_score(y_HoYin_train, y_HoYin_train_saga_pred, average='macro')
print(saga_recall_score)

lbfgs_pred1 = LR_clf_HoYin_lbfgs.predict([some_digit1])
print(lbfgs_pred1)
lbfgs_pred2 = LR_clf_HoYin_lbfgs.predict([some_digit2])
print(lbfgs_pred2)
lbfgs_pred3 = LR_clf_HoYin_lbfgs.predict([some_digit3])
print(lbfgs_pred3)

saga_pred1 = LR_clf_HoYin_saga.predict([some_digit1])
print(saga_pred1)
saga_pred2 = LR_clf_HoYin_saga.predict([some_digit2])
print(saga_pred2)
saga_pred3 = LR_clf_HoYin_saga.predict([some_digit3])
print(saga_pred3)