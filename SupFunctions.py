import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay



def barplot_balance(data: pd.DataFrame, feature: str) -> None:
    
    '''Построение баланса классов для признака(атрибута)'''

    x = plt.figure(figsize=(15, 8))
    feature_count = data[feature].value_counts()
    total = data.shape[0]
    feature_values = (feature_count / total) * 100
    ax = sns.barplot(x=feature_values.index, y=feature_values, palette='pastel')
    plt.title('Balance')
    plt.ylabel('Percentage')
    plt.xlabel('LoanApproved')

    for p in ax.patches:
        ax.annotate('{:.3f}%'.format(p.get_height()),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 2),
                    textcoords='offset points')
    plt.show()
    
def boxplot_hue_target(data: pd.DataFrame, x: str, y: str, target: str, violin: bool = False):
    
    '''Построение boxplot или violinplot для двух числовых признаков 
    в разрезе целевой переменной'''
    
    plt.figure(figsize=(15, 5))
    
    if violin:
        sns.violinplot(x=x, y=y, hue=target, data=data, palette='pastel')

    else:    
        sns.boxplot(x=x, y=y, hue=target, data=data, palette='pastel')

    plt.title(f'{x}-{y}-{target}', fontsize=16)
    plt.ylabel(y, fontsize=14)
    plt.xlabel(x, fontsize=14)
    plt.show()

def barplot_group(df_data: pd.DataFrame, col_main: str, col_group: str,
                     title: str, rotate: bool=False, xydist: int=10) -> None:
    """
    Построение barplot с нормированными данными с выводом значений на графике
    """

    plt.figure(figsize=(15, 8))

    data = (df_data.groupby(
        [col_group])[col_main].value_counts(normalize=True).rename(
            'percentage').mul(100).reset_index().sort_values(col_group))

    ax = sns.barplot(x=col_main, y="percentage", hue=col_group, data=data)

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,  # текст
            (p.get_x() + p.get_width() / 2., p.get_height()),  # координата xy
            ha='center',  # центрирование
            va='center',
            rotation=90*rotate,
            xytext=(0, xydist),
            textcoords='offset points',  # точка смещения относительно координаты
            fontsize=12)

    plt.title(title, fontsize=16)
    plt.ylabel('Percentage density', fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.xticks(rotation=30)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()
    
    
def multiclass_distplot_hue(data: pd.DataFrame, target: str, feature: str):
    # Создаем фигуру и оси
    plt.subplots(figsize=(15, 8))
    for i in data[target].value_counts().index.tolist():
        # Рисуем распределения дохода по n группам на осях
        sns.distplot(data[data[target] == i][feature], hist=False, kde=True, label=f'{target} = {i}')
    # Настраиваем заголовок, метки осей и легенду
    plt.title(f'Density distribution of {feature} by {target}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Density')
    plt.legend()
    # Показываем график
    plt.show()
    
def check_overfitting(model, X_train, y_train, X_test, y_test, metric_fun):
    """
    Проверка на overfitting для регрессии
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    value_train = metric_fun(y_train, y_pred_train)
    value_test = metric_fun(y_test, y_pred_test)

    print(f'{metric_fun.__name__} train: %.3f' % value_train)
    print(f'{metric_fun.__name__} test: %.3f' % value_test)
    print(f'delta = {(abs(value_train - value_test)/value_test*100):.1f} %')
    
def scale_pos_weight_calc(y: pd.Series) -> float:
    '''
    Функция для вычисления значения scale_pos_weight
    '''
    num_positive = np.sum(y)
    num_negative = len(y) - num_positive
    
    return num_negative / num_positive   

def scale_pos_weight_calc_multiclass(y: pd.Series) -> list:
    """
    Функция для вычисления весов классов для многоклассовой классификации

    Args:
        y: Серия с целевыми переменными

    Returns:
        Список весов классов, где индекс соответствует метке класса
    """

    class_counts = y.value_counts()
    total_count = len(y)
    class_weights = [total_count / (len(y[y == cls]) * class_counts.size) for cls in class_counts.index]
    return class_weights


from sklearn.calibration import CalibratedClassifierCV

def check_overfitting_multiclass(model, X_train, y_train, X_test, y_test, proba=True):
    """
    Проверка на overfitting для многоклассовой классификации, используя метрику ROC-AUC

    Args:
        model: обученная модель
        X_train: обучающая выборка
        y_train: целевые значения для обучающей выборки
        X_test: тестовая выборка
        y_test: целевые значения для тестовой выборки
        proba: флаг, указывающий, нужно ли использовать predict_proba или decision_function
    """

    if proba:
        y_pred_proba_train = model.predict_proba(X_train)
        y_pred_proba_test = model.predict_proba(X_test)
    else:
        # Калибровка для получения вероятностей
        clf = CalibratedClassifierCV(model, cv=5)
        clf.fit(X_train, y_train)
        y_pred_proba_train = clf.predict_proba(X_train)
        y_pred_proba_test = clf.predict_proba(X_test)

    roc_auc_train = roc_auc_score(y_train, y_pred_proba_train, multi_class='ovr')
    roc_auc_test = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr')

    print(f"ROC-AUC train: {roc_auc_train:.3f}")
    print(f"ROC-AUC test: {roc_auc_test:.3f}")
    print(f"Delta ROC-AUC: {(roc_auc_train - roc_auc_test)*100:.1f}%")


def plot_confusion_matrix(y_true, X, ax, model=None, prediction=None):
    """Визуализация ConfusionMatrix"""
    dict_code = {2: 'Good', 1: 'Standard', 0: 'Poor'}
    if prediction is None:
        prediction = model.predict(X)
        prediction = np.array(list(map(lambda x: dict_code[x],prediction)))
    labels = sorted(set(prediction))
    y_true_encoded = y_true.map(dict_code).array
    cm_ovo = confusion_matrix(y_true_encoded, prediction, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ovo, display_labels=labels)
    
    if ax:
        disp.plot(ax=ax)

def calculate_class_weights(y):
    class_counts = y.value_counts()
    total_count = len(y)
    class_weights = [total_count / count for count in class_counts]
    return class_weights
    
def check_overfitting_classification(model, X_train, y_train, X_test, y_test):
    """
    Проверка на overfitting для классификации с использованием ROC-AUC.
    """
    try:  # Пытаемся получить вероятности классов
        y_pred_train = model.predict_proba(X_train)[:, 1]  # Вероятность класса 1
        y_pred_test = model.predict_proba(X_test)[:, 1]   # Вероятность класса 1
    except AttributeError: # Если model не имеет predict_proba
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)


    value_train = roc_auc_score(y_train, y_pred_train)
    value_test = roc_auc_score(y_test, y_pred_test)

    print(f'ROC-AUC train: %.3f' % value_train)
    print(f'ROC-AUC test: %.3f' % value_test)

    if value_test == 0: # Избегаем деления на 0
        delta = np.inf if value_train > 0 else 0
    else:
        delta = (abs(value_train - value_test) / value_test) * 100

    print(f'delta = {delta:.1f} %')

    return value_train, value_test, delta # Возвращаем значения для дальнейшего анализа
