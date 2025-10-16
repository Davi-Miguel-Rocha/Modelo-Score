# Modelo-Score

# 📉 Modelo de Previsão de Score de Crédito com Deeplearning4j (DL4J)

Este projeto implementa um modelo de Rede Neural Densa (Deep Neural Network - DNN) utilizando a biblioteca **Deeplearning4j (DL4J)** para prever um **Score de Crédito** com base em 8 features financeiras e comportamentais de clientes.

O modelo é treinado usando Java e a arquitetura é otimizada para tarefas de regressão (previsão de um valor contínuo) em cenários de risco de crédito.

## ✨ Características Principais

* **Tecnologia:** Implementado em Java com **Deeplearning4j (DL4J)**.
* **Arquitetura:** Rede Densa Multicamadas (*Feedforward*) com 3 camadas ocultas.
* **Otimizador:** Adam, com taxa de aprendizado de `0.0005`.
* **Função de Perda:** Erro Quadrático Médio (`MSE`), ideal para regressão.
* **Pré-processamento:** Normalização Z-Score (`NormalizerStandardize`) aplicada aos dados.
* **Resultado:** Previsão de um score final que é "clipado" (limitado) entre **300 e 1000**.
* **Produção:** O modelo treinado e o objeto de normalização são salvos em `.zip` e `.bin` para uso em inferência em produção.
* **SDK:** Microsoft OpenJDK 21.0.8
* **Language Level:** 24 - Stream gatherers

## 🏗️ Arquitetura da Rede Neural

O modelo é composto por uma sequência de camadas densas:

| Camada | Tipo | Neurônios de Entrada (`nIn`) | Neurônios de Saída (`nOut`) | Função de Ativação |
| :---: | :---: | :---: | :---: | :---: |
| **Entrada** | `DenseLayer` | 8 | 128 | ReLU |
| **Oculta 1** | `DenseLayer` | 128 | 64 | LeakyReLU |
| **Oculta 2** | `DenseLayer` | 64 | 32 | ReLU |
| **Saída** | `OutputLayer` | 32 | 1 | Identity |

## 🔢 As 8 Features de Entrada

O modelo recebe 8 features numéricas como entrada. Elas representam os principais fatores de risco de crédito:

| Feature Index | Descrição Provável | Exemplo de Tipo |
| :---: | :---: | :---: |
| **1** | Número de Pendências / Inadimplências Ativas | Contagem (Integer) |
| **2** | Grau de Risco (Ex: Atrasos Graves Recentes) | Categórica Codificada (Integer) |
| **3** | Flag de Restrições Financeiras / Histórico de Falência | Binária (0 ou 1) |
| **4** | Número de Consultas de Crédito Recentes | Contagem (Integer) |
| **5** | Renda Mensal Bruta | Contínua (Double) |
| **6** | Taxa de Endividamento / Comprometimento de Renda | Contínua (0.0 a 1.0) |
| **7** | Meses de Histórico de Crédito (Antiguidade) | Contagem (Integer) |
| **8** | Flag Adicional (Ex: Estabilidade de Emprego/Moradia) | Binária (0 ou 1) |

## 🚀 Como Executar o Projeto

### Pré-requisitos

* JDK 8 ou superior
* Gradle (necessário para rodar o comando, se o projeto usar o `gradlew`)

### Execução

1.  Clone este repositório:
    ```bash
    git clone [LINK_DO_SEU_REPOSITORIO]
    cd nome-do-seu-repositorio
    ```
2.  Execute o arquivo principal usando o Gradle (conforme a instrução do código):
    ```bash
    ./gradlew run
    ```

O script irá:
1.  Inicializar e configurar a Rede Neural.
2.  Carregar o *dataset* (`Datasetexemple.createDataset()`).
3.  Dividir dados em treino (80%) e validação (20%).
4.  Calcular e aplicar a normalização.
5.  Treinar o modelo por `1000 epochs`.
6.  Imprimir o *Loss* de Treino e Validação a cada 10 epochs.
7.  Testar a inferência com 3 perfis de clientes (Alto, Médio e Baixo Risco).
8.  Salvar os arquivos de produção (`credit_score_model2.zip` e `credit_score_normalizer2.bin`).

## 💾 Artefatos de Produção

Após a execução, os seguintes arquivos serão gerados no diretório raiz:

* `credit_score_model2.zip`: O modelo DL4J treinado e pronto para inferência.
* `credit_score_normalizer2.bin`: O objeto `NormalizerStandardize` serializado, essencial para pré-processar corretamente novos dados antes de passá-los para o modelo.
