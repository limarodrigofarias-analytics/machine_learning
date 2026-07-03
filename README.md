# Detecção de Spam com Machine Learning — Comparação de Algoritmos Supervisionados

Projeto acadêmico que compara **4 algoritmos clássicos de classificação** (KNN, SVM, Árvore de Decisão e MLP) na tarefa de detectar e-mails de spam, usando um dataset real com 5.172 e-mails.

## Objetivo

Treinar, testar e comparar diferentes algoritmos de aprendizado supervisionado para classificação binária, avaliando qual deles apresenta o melhor equilíbrio entre acurácia, precisão, recall e F1-score na detecção de spam.

## Dataset

- **Fonte:** [Email Spam Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data)
- **5.172 e-mails**, representados como vetores de contagem de palavras (*bag of words*) com as **3.000 palavras mais frequentes** do corpus
- **Rótulo:** `0` = e-mail legítimo · `1` = spam
- Dataset desbalanceado (~29% spam), sem valores ausentes ou duplicados

## Metodologia

1. **Preparação dos dados:** limpeza, análise exploratória (balanceamento de classes, distribuição de palavras, correlação) e divisão *holdout* em treino (60%) / validação (20%) / teste (20%), com estratificação
2. **Pré-processamento:** padronização (`StandardScaler`) para os modelos sensíveis à escala (KNN, SVM, MLP)
3. **Modelos treinados:**
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM, kernel RBF)
   - Árvore de Decisão
   - Multi-Layer Perceptron (MLP)
4. **Avaliação:** acurácia, precisão, recall, F1-score e matrizes de confusão no conjunto de teste
5. **Análise de hiperparâmetros:** efeito do número de vizinhos (K) no KNN e impacto da padronização no SVM

## Resultados

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|---|---|---|---|---|
| KNN | 0.80 | 0.61 | 0.93 | 0.73 |
| SVM | 0.92 | 0.98 | 0.75 | 0.85 |
| Árvore de Decisão | 0.91 | 0.82 | 0.89 | 0.85 |
| **MLP** | **0.97** | 0.91 | **0.99** | **0.95** |

O **MLP (rede neural)** teve o melhor desempenho geral, com destaque para o recall de 99% — ou seja, quase nenhum spam passou despercebido.

## Tecnologias

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## Estrutura

```
├── Trabalho_Pratico_Comparacao_Algoritmos_ML.ipynb   # notebook completo e executado
├── emails.csv                                          # dataset utilizado
└── README.md
```

## Como executar

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Trabalho_Prático_—_Comparação_de_Algoritmos_de_Aprendizado_de_Máquina_Supervisionado01.ipynb
```

O notebook é autocontido: basta ter o `emails.csv` na mesma pasta e rodar todas as células.

## Principais aprendizados

- Alta dimensionalidade (3.000 atributos) exige atenção especial à padronização e ao tempo de treinamento, principalmente no SVM
- Em datasets desbalanceados, acurácia sozinha não conta toda a história — recall e F1-score revelam melhor a qualidade real do modelo
- Modelos mais simples (KNN) já entregam resultados razoáveis, mas redes neurais mesmo pequenas capturam melhor os padrões em problemas de alta dimensão

---

Projeto desenvolvido como trabalho prático da disciplina de Fundamentos e Aplicações de Aprendizagem de Máquina.
