import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm

# --- 1. Загрузка и предобработка данных для получения ссылочного `df` для опций UI ---
# Эта часть имитирует шаги загрузки и предобработки данных из блокнота,
# чтобы гарантировать согласованность элементов UI (мин/макс/уникальные значения) с обучением.

item = 'credit_data'
package = 'modeldata'
df_raw = sm.datasets.get_rdataset(item, package, cache=True).data

# Сохраняем оригинальные категориальные значения до кодирования для отображения в UI
# (Эти списки будут использоваться для получения уникальных значений после заполнения NaN)
# original_home_values = df_raw['Home'].dropna().unique().tolist()
# original_marital_values = df_raw['Marital'].dropna().unique().tolist()
# original_job_values = df_raw['Job'].dropna().unique().tolist()

# Обработка пропущенных значений для числовых столбцов
for col in ['Income', 'Assets', 'Debt']:
    df_raw[col] = df_raw[col].fillna(df_raw[col].median())

# Обработка пропущенных значений для категориальных столбцов
for col in ['Home', 'Marital', 'Job']:
    df_raw[col] = df_raw[col].fillna(df_raw[col].mode()[0])

# Замена нулей на среднее значение ненулевых значений для 'Assets', 'Debt'
for col in ['Assets', 'Debt']:
    median_val = df_raw[df_raw[col] != 0][col].mean()
    df_raw[col] = df_raw[col].replace(0, median_val)

# Удаление столбца 'Seniority'
df_processed_step1 = df_raw.drop('Seniority', axis=1)

# Label-кодирование категориальных столбцов
# Примечание: 'Status' также является категориальным и был закодирован в блокноте.
# 'Records' был закодирован, а затем удален.
le_home = LabelEncoder()
df_processed_step1['Home'] = le_home.fit_transform(df_processed_step1['Home'])
le_marital = LabelEncoder()
df_processed_step1['Marital'] = le_marital.fit_transform(df_processed_step1['Marital'])
le_job = LabelEncoder()
df_processed_step1['Job'] = le_job.fit_transform(df_processed_step1['Job'])
le_records = LabelEncoder()
df_processed_step1['Records'] = le_records.fit_transform(df_processed_step1['Records'])
le_status = LabelEncoder()
df_processed_step1['Status'] = le_status.fit_transform(df_processed_step1['Status'])

# Создаем словари для обратного отображения закодированных значений на оригинальные (английские) названия
home_map_reverse = {idx: label for idx, label in enumerate(le_home.classes_)}
marital_map_reverse = {idx: label for idx, label in enumerate(le_marital.classes_)}
job_map_reverse = {idx: label for idx, label in enumerate(le_job.classes_)}

# Словари для перевода английских категорий на русский язык
home_translations = {
    'rent': 'Аренда',
    'owner': 'Собственник',
    'free': 'Бесплатно',
    'priv': 'Приватизированное',
    'other': 'Другое',
    'rent1': 'Аренда (подтип 1)' # Предполагаем, что rent1 это подтип аренды
}
marital_translations = {
    'married': 'Женат/Замужем',
    'single': 'Холост/Не замужем',
    'widow': 'Вдовец/Вдова',
    'divorced': 'Разведен/Разведена',
    'separated': 'Раздельное проживание'
}
job_translations = {
    'freelance': 'Фрилансер',
    'fixed': 'Постоянная работа',
    'partime': 'Частичная занятость',
    'unempl': 'Безработный'
}


# Удаление выбросов с использованием метода IQR
Q1 = df_processed_step1.quantile(0.25)
Q3 = df_processed_step1.quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df_processed_step1[~((df_processed_step1 < (Q1 - 1.5 * IQR)) | (df_processed_step1 > (Q3 + 1.5 * IQR))).any(axis=1)]

# Окончательное удаление столбцов (Debt и Records были удалены после удаления выбросов)
# df_reference будет DataFrame, который выглядит как X
df_reference = df_no_outliers.drop(['Debt', 'Records', 'Status'], axis=1, errors='ignore')

# --- 2. Загрузка модели и стандартизатора ---
# Указываем путь к файлам модели и стандартизатора (в текущей директории)
model_path = 'final_gradient_boosting_model.pkl'
scaler_path = 'scaler.pkl'

# Загрузка обученной модели
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Загрузка стандартизатора данных
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# --- 3. Пользовательский интерфейс Streamlit ---
st.title('Предсказание кредитного статуса')

st.header('Введите данные заемщика:')

# Получение уникальных значений для категориальных признаков из df_reference
home_options = sorted(df_reference['Home'].unique().tolist())
marital_options = sorted(df_reference['Marital'].unique().tolist())
job_options = sorted(df_reference['Job'].unique().tolist())

# Ввод данных пользователем с русским переводом
home = st.selectbox('Тип жилья (Home)', options=home_options, format_func=lambda x: home_translations.get(home_map_reverse[x], home_map_reverse[x]))
marital = st.selectbox('Семейное положение (Marital)', options=marital_options, format_func=lambda x: marital_translations.get(marital_map_reverse[x], marital_map_reverse[x]))
job = st.selectbox('Тип занятости (Job)', options=job_options, format_func=lambda x: job_translations.get(job_map_reverse[x], job_map_reverse[x]))

time = st.number_input('Срок кредита (Time)', min_value=int(df_reference['Time'].min()), max_value=int(df_reference['Time'].max()), value=int(df_reference['Time'].mean()))
age = st.number_input('Возраст (Age)', min_value=int(df_reference['Age'].min()), max_value=int(df_reference['Age'].max()), value=int(df_reference['Age'].mean()))
expenses = st.number_input('Расходы (Expenses)', min_value=int(df_reference['Expenses'].min()), max_value=int(df_reference['Expenses'].max()), value=int(df_reference['Expenses'].mean()))
income = st.number_input('Доход (Income)', min_value=float(df_reference['Income'].min()), max_value=float(df_reference['Income'].max()), value=float(df_reference['Income'].mean()))
assets = st.number_input('Активы (Assets)', min_value=float(df_reference['Assets'].min()), max_value=float(df_reference['Assets'].max()), value=float(df_reference['Assets'].mean()))
amount = st.number_input('Сумма кредита (Amount)', min_value=int(df_reference['Amount'].min()), max_value=int(df_reference['Amount'].max()), value=int(df_reference['Amount'].mean()))
price = st.number_input('Стоимость покупки (Price)', min_value=int(df_reference['Price'].min()), max_value=int(df_reference['Price'].max()), value=int(df_reference['Price'].mean()))

predict_button = st.button('Предсказать')

# --- 4. Логика предсказания ---
if predict_button:
    input_data = pd.DataFrame([[home, time, age, marital, job, expenses, income, assets, amount, price]],
                              columns=['Home', 'Time', 'Age', 'Marital', 'Job', 'Expenses', 'Income', 'Assets', 'Amount', 'Price'])

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    st.subheader('Результат предсказания:')
    if prediction[0] == 1:
        st.success('Кредитный статус: Хороший (Good)')
    else:
        st.error('Кредитный статус: Плохой (Bad)')
