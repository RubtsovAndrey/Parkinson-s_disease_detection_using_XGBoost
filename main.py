import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import time
import cupy as cp
import matplotlib.pyplot as plt
import os

# Создание папки для графиков, если она не существует
# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')


def load_data(file_path):
    # Загрузка данных из файла CSV
    # Load data from CSV file
    return pd.read_csv(file_path)


def preprocess_data(data):
    # Предварительная обработка данных: удаление ненужных столбцов, заполнение пропусков, создание новых признаков
    # Data preprocessing: remove columns, handle missing values, create new features
    x = data.drop(columns=['name', 'status'])
    y = data['status']
    x = x.fillna(x.mean())
    x['mdvp_fo_fhi_interaction'] = x['MDVP:Fo(Hz)'] * x['MDVP:Fhi(Hz)']
    x['mdvp_fo_flo_interaction'] = x['MDVP:Fo(Hz)'] * x['MDVP:Flo(Hz)']
    return x, y


def scale_data(x_train, x_test):
    # Нормализация данных
    # Data normalization
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def prepare_dmatrix(x_train, x_test, y_train, y_test):
    # Создание DMatrix для XGBoost с использованием GPU
    # Prepare DMatrix for XGBoost using GPU
    x_train_gpu = cp.array(x_train)
    x_test_gpu = cp.array(x_test)
    dtrain = xgb.DMatrix(x_train_gpu, label=y_train)
    dtest = xgb.DMatrix(x_test_gpu, label=y_test)
    return dtrain, dtest


def train_xgboost(dtrain, dtest):
    # Обучение модели XGBoost с использованием GPU
    # Train XGBoost model using GPU
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'lambda': 1.0
    }
    cv_results = xgb.cv(params, dtrain, num_boost_round=200, nfold=5, early_stopping_rounds=10, verbose_eval=0)
    best_num_boost_round = cv_results.shape[0]
    model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round, evals=[(dtest, 'eval')],
                      verbose_eval=0, early_stopping_rounds=10)
    return model, cv_results


def evaluate_model(model, dtest, y_test):
    # Оценка модели и точности
    # Model evaluation and accuracy assessment
    y_pred = model.predict(dtest)
    y_pred_binary = np.round(y_pred)
    return accuracy_score(y_test, y_pred_binary) * 100


def plot_training_history(cv_results):
    # Создание и сохранение графика logloss
    # Plot and save logloss graph
    plt.figure(figsize=(10, 6))
    plt.plot(cv_results['train-logloss-mean'], label='Train logloss')
    plt.plot(cv_results['test-logloss-mean'], label='Test logloss')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Logloss')
    plt.title('Training and Validation Logloss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/training_history.png')
    plt.close()


def plot_feature_importance(model):
    # Создание и сохранение графика важности признаков
    # Plot and save feature importance graph
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, importance_type='weight', title='Feature Importance',
                        xlabel='Feature Weight', ylabel='Features')
    plt.savefig('plots/feature_importance.png')
    plt.close()


def plot_accuracy(accuracy):
    # Создание и сохранение графика с итоговой точностью модели
    # Plot and save a graph with final model accuracy
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, f'Accuracy: {accuracy:.2f}%', fontsize=30, ha='center')
    plt.axis('off')
    plt.title('Model Accuracy')
    plt.savefig('plots/model_accuracy.png')
    plt.close()


def main():
    # Основная функция для запуска всех этапов: загрузка данных, обучение модели, оценка, визуализация
    # Main function to run all steps: data loading, model training, evaluation, visualization
    start_time = time.time()

    data_path = 'data/parkinsons.data'
    data = load_data(data_path)
    x, y = preprocess_data(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    dtrain, dtest = prepare_dmatrix(x_train_scaled, x_test_scaled, y_train, y_test)
    model, cv_results = train_xgboost(dtrain, dtest)
    accuracy = evaluate_model(model, dtest, y_test)

    # Сохранение графиков
    # Save plots
    plot_training_history(cv_results)
    plot_feature_importance(model)
    plot_accuracy(accuracy)

    execution_time = time.time() - start_time
    device_used = 'GPU' if cp.get_array_module(cp.array([1])) == cp else 'CPU'

    print(f"Точность модели: {accuracy:.2f}% / Model Accuracy: {accuracy:.2f}%")
    print(f"Время выполнения: {execution_time:.2f} секунд / Execution Time: {execution_time:.2f} seconds")
    print(f"Использовался {device_used} / {device_used} was used")


if __name__ == "__main__":
    main()
