import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# carrega corretamente o arquivo que servirá de base para o treinamento dos modelos
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
caminho_treino = os.path.join(diretorio_atual, '02_treino_sinais_vitais_com_label.txt')

# carrega os dados do arquivo de treinamento
# id: identificação da vítima (sequencial)
# s1: pressão sistólica (pSist) [5,22] - usada no cálculo de s3
# s2: pressão diastólica (pDiast) [0,15] - usada no cálculo de s3
# s3: qualidade de pressão (qPA) [-10,10] - onde 0 é a qualidade máxima, -10 a pior qualidade baixa e +10 a pior qualidade alta
# s4: pulso [0,200] - bpM
# s5: respiração [0,22] - FpM
# g: gravidade
# y: classe de gravidade [1-4] - 1 é crítico, 2 é instável, 3 é potencialmente estável e 4 é estável
colunas = ['id', 's1', 's2', 's3', 's4', 's5', 'g', 'y']

# lê o arquivo de treinamento usando pandas e atribui os nomes das colunas (usa pandas pra facilitar a manipulação dos dados)
df = pd.read_csv(caminho_treino, names=colunas)

# guarda em x as variáveis de entrada (que servirão para treinar os modelos)
X = df[['s3', 's4', 's5']]

# guarda a variável de gravidade para a regressão
y_reg = df['g'] 

# guarda a variável das classes para a classificação
y_class = df['y']  # classe 1, 2, 3 ou 4

# divide os dados em conjuntos de treinamento e teste (70% treino, 30% teste)
# 42 é a semente para gerar os mesmos conjuntos de treino e teste em cada execução
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.3, random_state=42)

# escalonamento dos dados, para que todos os valores fiquem na mesma escala
scaler = StandardScaler()

# o escalonamento deve ser feito apenas com os dados de treinamento e depois aplicado aos dados de teste, para evitar vazamento de dados
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ****************************
# RANDOM FOREST
# ****************************

print("#"*15)
print("RANDOM FOREST")
print("#"*15)

print("*"*10)
print("REGRESSÃO")
print("*"*10)

# usa a função do sklearn para criar o modelo de random forest para regressão
# n_estimators é o número de árvores na floresta e random_state é a semente para gerar os mesmos resultados em cada execução
rf_reg = RandomForestRegressor(n_estimators=10, random_state=42)

# treina o modelo de random forest para regressão usando os dados de treinamento 
# depois faz as predições com os dados de teste
rf_reg.fit(X_train, y_reg_train)
rf_pred_reg = rf_reg.predict(X_test)

# calcula o RMSE (Root Mean Squared Error) para avaliar a performance do modelo de regressão
# é a distância geométrica entre o valor real e o valor predito pelo modelo
rmse_rf = root_mean_squared_error(y_reg_test, rf_pred_reg)
print(f"RMSE = {rmse_rf:.4f}")

print("*"*10)
print("CLASSIFICAÇÃO")
print("*"*10)

# de novo usa a função do sklearn para criar o modelo de random forest para classificação, com os mesmos parâmetros de antes
rf_clf = RandomForestClassifier(n_estimators=10, criterion='gini', random_state=42)

# treina o modelo de random forest para classificação usando os dados de treinamento e depois faz as predições com os dados de teste
rf_clf.fit(X_train, y_class_train)
rf_pred_class = rf_clf.predict(X_test)

# aqui ele imprime a matriz de confusão e o relatório de métricas para avaliar a performance do modelo de classificação
print("\nMatriz de Confusão:")
print(confusion_matrix(y_class_test, rf_pred_class))
print("\nRelatório de Métricas:")
print(classification_report(y_class_test, rf_pred_class))


# ****************************
# REDE NEURAL MLP
# ****************************

print("\n" + "#"*15)
print("REDE NEURAL MLP")
print("#"*15)

print("*"*10)
print("REGRESSÃO")
print("*"*10)

# usa a função do sklearn para criar o modelo de rede neural MLP para regressão
# usa 2 camadas ocultas (8 e 4 neurônios), função de ativação ReLU, otimizador Adam e 1500 iterações
mlp_reg = MLPRegressor(hidden_layer_sizes=(4, 2), activation='relu', solver='adam', max_iter=1500, random_state=42)

# cria o escalonador exclusivo para a variável alvo (gravidade)
scaler_g = MinMaxScaler()

# transforma o y_train para a escala 0 a 1
y_reg_train_scaled = scaler_g.fit_transform(y_reg_train.values.reshape(-1, 1)).ravel()

# treina o modelo de rede neural MLP com os dados de entrada E saída escalonados
mlp_reg.fit(X_train_scaled, y_reg_train_scaled)

# faz as predições (o resultado sairá na escala 0 a 1)
mlp_pred_reg_scaled = mlp_reg.predict(X_test_scaled)

# --- NOVA LINHA AQUI: Impede a rede de extrapolar os limites no teste de validação ---
mlp_pred_reg_scaled = np.clip(mlp_pred_reg_scaled, 0, 1)

# reverte as predições para a escala real da gravidade para podermos calcular o RMSE corretamente
mlp_pred_reg = scaler_g.inverse_transform(mlp_pred_reg_scaled.reshape(-1, 1)).ravel()

# calcula o RMSE para avaliar a performance do modelo de regressão da rede neural MLP 
rmse_mlp = root_mean_squared_error(y_reg_test, mlp_pred_reg)
print(f"RMSE = {rmse_mlp:.4f}")

print("*"*10)
print("CLASSIFICAÇÃO")
print("*"*10)

# usa novamente a função do sklearn para criar o modelo de rede neural MLP para classificação, com os mesmos parâmetros de antes
mlp_clf = MLPClassifier(hidden_layer_sizes=(4, 2), activation='relu', solver='adam', max_iter=1500, random_state=42)

# treina o modelo e faz as predições
mlp_clf.fit(X_train_scaled, y_class_train)
mlp_pred_class = mlp_clf.predict(X_test_scaled)

# imprime a matriz de confusão e o relatório de métricas
print("\nMatriz de Confusão:")
print(confusion_matrix(y_class_test, mlp_pred_class))
print("\nRelatório de Métricas:")
print(classification_report(y_class_test, mlp_pred_class))


# ****************************
# TESTE CEGO
# ****************************

print("\n" + "#"*15)
print("TESTE CEGO")
print("#"*15)

# carrega o arquivo de teste cego
caminho_cego = os.path.join(diretorio_atual, '01_treino_sinais_vitais_sem_label.txt')

if os.path.exists(caminho_cego):

    # não possui os parâmetros s1, s2, g e y
    colunas_cego = ['id', 's3', 's4', 's5']

    # lê o arquivo de teste cego usando pandas e atribui os nomes das colunas
    df_cego = pd.read_csv(caminho_cego, names=colunas_cego)
    X_cego = df_cego[['s3', 's4', 's5']]
    
    # executa as predições com o modelo treinado do random forest usando regressão e classificação
    rf_cego_g = rf_reg.predict(X_cego)
    rf_cego_y = rf_clf.predict(X_cego)
    
    # executa as predições com o modelo treinado da rede neural MLP usando regressão e classificação
    X_cego_scaled = scaler.transform(X_cego)

    # faz a predição da gravidade (sairá na escala de 0 a 1)
    mlp_cego_g_scaled = mlp_reg.predict(X_cego_scaled)
    
    # --- NOVA LINHA AQUI: Força os palpites da rede a ficarem estritamente entre 0 e 1 ---
    mlp_cego_g_scaled = np.clip(mlp_cego_g_scaled, 0, 1)
    
    # reverte para a escala original (agora o menor valor possível será o mínimo do seu treino)
    mlp_cego_g = scaler_g.inverse_transform(mlp_cego_g_scaled.reshape(-1, 1)).ravel()

    mlp_cego_y = mlp_clf.predict(X_cego_scaled)
    
    # prepara os dados para salvar em arquivos de texto, arredondando a gravidade para 4 casas decimais
    df_saida_rf = pd.DataFrame({
        'i': df_cego['id'],
        'gravid': np.round(rf_cego_g, 4),
        'classe': rf_cego_y
    })
    
    df_saida_mlp = pd.DataFrame({
        'i': df_cego['id'],
        'gravid': np.round(mlp_cego_g, 4),
        'classe': mlp_cego_y
    })
    
    # salva os resultados em arquivos de texto, um para cada modelo, com as colunas i, gravid e classe
    nome_arquivo_rf = 'teste_random_forest.txt'
    nome_arquivo_mlp = 'teste_mlp.txt'
    df_saida_rf.to_csv(nome_arquivo_rf, index=False, header=True)
    df_saida_mlp.to_csv(nome_arquivo_mlp, index=False, header=True)
    
    # calcula a concordância entre os modelos na classe e a diferença média na gravidade para os novos pacientes
    concordancia_classes = np.sum(rf_cego_y == mlp_cego_y) / len(df_cego) * 100
    dif_media_gravidade = np.mean(np.abs(rf_cego_g - mlp_cego_g))
    
    print(f"Concordância na classe: {concordancia_classes:.2f}%")
    print(f"Diferença média na gravidade: {dif_media_gravidade:.4f} pontos.")

# ==============================================================================
    # MÉTRICAS EXTRAS PARA O APÊNDICE DO TESTE CEGO
    # ==============================================================================
    print("\n" + "="*40)
    print("MÉTRICAS COMPLEMENTARES PARA O APÊNDICE")
    print("="*40)
    
    # 1. Distribuição de Classes (Frequência Absoluta)
    print("\nDistribuicao de Classes - Random Forest:")
    print(df_saida_rf['classe'].value_counts().sort_index())
    
    print("\nDistribuicao de Classes - MLP:")
    print(df_saida_mlp['classe'].value_counts().sort_index())
    
    # 2. Matriz de Cruzamento (RF vs MLP)
    print("\nMatriz de Cruzamento de Decisoes (Linha: RF, Coluna: MLP):")
    matriz_cruzada = pd.crosstab(df_saida_rf['classe'], df_saida_mlp['classe'])
    print(matriz_cruzada)
    
    # 3. Estatísticas da Gravidade Continua
    print("\nEstatísticas da Gravidade - Random Forest:")
    print(df_saida_rf['gravid'].describe()[['mean', 'std', 'min', 'max']])
    
    print("\nEstatísticas da Gravidade - MLP:")
    print(df_saida_mlp['gravid'].describe()[['mean', 'std', 'min', 'max']])