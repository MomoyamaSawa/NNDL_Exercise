import numpy as np


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = 0.00001  # 调整学习率

        # 使用LeCun初始化
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)

    # 使用Leaky ReLU替换ReLU
    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def train(self, x_train, y_train, epochs):
        for epoch in range(epochs):
            # 前向传播
            h1 = self.leaky_relu(np.dot(x_train, self.W1) + self.b1)
            h2 = self.leaky_relu(np.dot(h1, self.W2) + self.b2)
            y_pred = np.dot(h2, self.W3) + self.b3

            # 计算损失
            loss = np.mean((y_pred - y_train) ** 2)
            if np.isnan(loss):
                print("Loss is nan")
                break
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, loss {loss}")

            # 反向传播
            grad_y_pred = 2.0 * (y_pred - y_train)
            grad_W3 = np.dot(h2.T, grad_y_pred)
            grad_b3 = np.sum(grad_y_pred, axis=0)
            grad_h2 = np.dot(grad_y_pred, self.W3.T) * self.leaky_relu_derivative(h2)
            grad_W2 = np.dot(h1.T, grad_h2)
            grad_b2 = np.sum(grad_h2, axis=0)
            grad_h1 = np.dot(grad_h2, self.W2.T) * self.leaky_relu_derivative(h1)
            grad_W1 = np.dot(x_train.T, grad_h1)
            grad_b1 = np.sum(grad_h1, axis=0)

            # 更新权重和偏置
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W3 -= self.lr * grad_W3
            self.b3 -= self.lr * grad_b3


def generate_data(func, train_range, test_range, train_size, test_size):
    # 生成训练数据
    x_train = np.linspace(train_range[0], train_range[1], train_size)
    y_train = func(x_train)

    # 生成测试数据
    x_test = np.linspace(test_range[0], test_range[1], test_size)
    y_test = func(x_test)

    # 将数据reshape为神经网络所需的形状
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return x_train, y_train, x_test, y_test


# 定义函数
def func(x):
    return np.sin(x)


# 生成数据
x_train, y_train, x_test, y_test = generate_data(
    func, [-np.pi, np.pi], [-np.pi, np.pi], 2000, 400
)

x_train = x_train / np.pi
x_test = x_test / np.pi

# 创建并训练神经网络
nn = NeuralNetwork(1, 50, 1)
nn.train(x_train, y_train, 10000)

# 使用训练好的神经网络进行预测
h1 = nn.leaky_relu(np.dot(x_test, nn.W1) + nn.b1)
h2 = nn.leaky_relu(np.dot(h1, nn.W2) + nn.b2)
y_pred = np.dot(h2, nn.W3) + nn.b3
import matplotlib.pyplot as plt

# 绘制训练数据和预测数据的对比图
plt.figure(figsize=(10, 6))
plt.plot(x_test * np.pi, y_test, "b-", label="True")
plt.plot(x_test * np.pi, y_pred, "r-", label="Predicted")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison between true and predicted values")

# 保存图形
plt.savefig("./result.png")
plt.show()
