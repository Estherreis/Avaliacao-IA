import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import product

# Função de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe Rede Neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

    def feedforward(self, x):
        activations = [x]
        input = x
        for weight in self.weights:
            input = sigmoid(np.dot(input, weight))
            activations.append(input)
        return activations

    def backpropagation(self, activations, y_true):
        error = y_true - activations[-1]
        deltas = [error * sigmoid_derivative(activations[-1])]
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * sigmoid_derivative(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.atleast_2d(activations[i]).T.dot(np.atleast_2d(deltas[i]))

    def train(self, X, y):
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                activations = self.feedforward(xi)
                self.backpropagation(activations, yi)
            # Calcula e armazena o erro médio da época
            predictions = self.predict(X)
            loss = np.mean(np.square(y - predictions))
            self.loss_history.append(loss)

    def predict(self, X):
        return np.array([self.feedforward(xi)[-1] for xi in X])

# Dados XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
y = np.array([
    [0],
    [1],
    [1],
    [0],
])

# Parâmetros para testar
#neurons_list = [2, 4, 6]
#learning_rates = [0.01, 0.1, 0.5]
#epochs_list = [1000, 5000, 10000]

#todas_combinacoes = list(product(neurons_list, learning_rates, epochs_list))
#combinacoes_sorteadas = random.sample(todas_combinacoes, 5)

combinacoes_sorteadas = [(2, 0.01, 1000), (6, 0.1, 5000), (4, 0.5, 5000),
                         (4, 0.01, 10000), (4, 0.1, 5000), (6, 0.01, 10000)]

# Armazenar resultados
resultados = []

for neurons, lr, epochs in combinacoes_sorteadas:
    print(f"Treinando: Neurônios={neurons}, Taxa de Aprendizado={lr}, Épocas={epochs}")
    nn = NeuralNetwork(input_size=2, hidden_sizes=[neurons], output_size=1, learning_rate=lr, epochs=epochs)
    nn.train(X, y)
    predictions = nn.predict(X)
    predictions_rounded = np.round(predictions)
    accuracy = np.mean(predictions_rounded == y)
    loss = np.mean(np.square(y - predictions))

    # Salvar resultados
    resultados.append({
        'Neurônios Ocultos': neurons,
        'Taxa de Aprendizagem': lr,
        'Épocas': epochs,
        'Acurácia': round(accuracy, 2),
        'Erro Quadrático Médio': round(loss, 4)
    })

    # Gerar gráfico de convergência
    plt.figure()
    plt.plot(nn.loss_history)
    plt.title(f'Convergência - Neurônios: {neurons}, Taxa Aprendizado: {lr}, Épocas: {epochs}')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'convergencia_n{neurons}_lr{lr}_e{epochs}.png')
    plt.close()

# Mostrar tabela final
df_resultados = pd.DataFrame(resultados)
print(df_resultados)
