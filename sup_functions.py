import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def barplot_balance(data: pd.DataFrame, feature: str) -> None:
    
    '''Построение баланса классов для признака(атрибута)'''

    x = plt.figure(figsize=(15, 8))
    feature_count = data[feature].value_counts()
    total = data.shape[0]
    feature_values = (feature_count / total) * 100
    ax = sns.barplot(x=feature_values.index, y=feature_values, palette='pastel')
    plt.title('Balance')
    plt.ylabel('Percentage')
    plt.xlabel(f'{feature} balance')

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