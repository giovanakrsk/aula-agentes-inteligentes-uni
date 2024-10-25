import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
df = pd.read_csv('dataset.csv')

colunas_independentes_x = ["nAlunos", "TempAmbient", "NcompSala", "CapacSala", "Hr"]
colunas_dependentes_y = ["TempAr"]

dados_x = df[colunas_independentes_x]
dados_y = df[colunas_dependentes_y]

modelo = LinearRegression().fit(dados_x, dados_y)

num_alunos_test = 21
num_temp_ambiente_test = 27
num_comp_ligados_test = 27
num_capacidade_sala_test = 30
num_horario_test = 20

valores_test = np.array([[num_alunos_test, num_temp_ambiente_test, num_comp_ligados_test, num_capacidade_sala_test, num_horario_test]])

predicao = modelo.predict(valores_test)
print(predicao)