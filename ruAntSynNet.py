import os
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Папка с файлами, содержащими массивы numpy (т.е. вектора простых путей)
numpy_arrays = 'YOUR_PATH'

X = []  # вектора
y = []  # метки классов
print('Загрузка данных...')
for file in tqdm(os.listdir(numpy_arrays)):
    filepath = numpy_arrays + file
    loaded_array = np.load(filepath)
    X.append(loaded_array)
    if 'syn' in file:
        y.append(0)
    else:
        y.append(1)

X = np.array(X)
y = np.array(y)

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, t_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создадим классификатор: LogisticRegression, Perceptron, LinearSVC, DecisionTreeClassifier,
# RandomForestClassifier или GradientBoostingClassifier
cls = LogisticRegression()  # изменить классификатор здесь
# Обучим классификатор
print(f'Обучение классификатора {cls}...')
cls.fit(X_train, t_train)

# Сделаем предсказание на тестовой выборке и посчитаем метрики: accuracy, precision, recall, f1-score, roc_auc_score
print('Классификация примеров из тестовой выборки и оценка качества классификации...')
y_pred = cls.predict(X_test)
print(f'Accuracy: {cls.score(X_test, y_test)}')

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))

print(f'roc_auc_score: {roc_auc_score(y_test, y_pred)}')
