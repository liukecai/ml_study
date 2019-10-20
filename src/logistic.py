#-*-coding:utf-8-*-

import random
import sklearn.datasets
import numpy as np

import matplotlib.pyplot as plt


class Logistic(object):
    def __init__(self, feature_size, train_data,
                 train_label, test_data, test_label,
                 learn_rate=0.002,
                 learn_rate_decay=1,
                 reg_rate=0,
                 its=1000,
                 step=10,
                 plot=False):

        # self.weight = np.random.rand(feature_size)
        self.weight = np.zeros(feature_size)
        # self.bias = np.random.rand(1)
        self.bias = np.zeros(1)
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.reg_rate = reg_rate

        self.its = its
        self.step = step

        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

        self.max_acc = 0

        self.plot = plot

        if self.plot:
            plt.ion()
            fig = plt.figure()
            self.ax1 = fig.add_subplot(111)

            self.plot_x = []
            self.plot_y = []

            self.line, = self.ax1.plot(self.plot_x, self.plot_y, linestyle="-", color="r")


    def get_sample(self):
        index = random.randint(0, len(self.train_data) - 1)
        sample_data = self.train_data[index]
        sample_label = self.train_label[index]

        sample_data = sample_data - np.max(sample_data)
        # sample_data = sample_data / np.sum(sample_data)

        return sample_data, sample_label


    def train(self):
        loss = 0

        for it in range(self.its):
            if it % self.step == 0:
                acc = self.test()
                print("Train its: %d, loss: %f, acc: %f, learn_rate: %.10f" % (it, loss/self.step, acc, self.learn_rate))
                loss = 0
                if acc != 0 and self.plot:
                    self.update_plot(it, acc)
            if it != 0 and it % 2000 == 0 and self.learn_rate > 1e-10:
                self.learn_rate *= self.learn_rate_decay

            loss += self._train(*self.get_sample())


    def _train(self, x, y):
        f_ = np.dot(self.weight, x) + self.bias
        y_ = 1 / (1 + np.exp(-f_))

        if y_[0] != 0 and y_[0] != 1:
            # if y_ != 0 and y_ != 1:
            loss = - (y * np.log(y_) + (1 - y) * np.log(1 - y_))

            # L2
            # Euclidean distance
            # np.linalg.norm
            loss += self.reg_rate * np.sum(np.square(self.weight)) + self.reg_rate * np.square(self.bias)

        else:
            loss = 0

        weight_inc = (y_ - y) * x
        bias_inc = (y_ - y)

        self.weight = self.weight - self.learn_rate * weight_inc - self.reg_rate * 2 * np.sum(self.weight)
        self.bias = self.bias - self.learn_rate * bias_inc - self.reg_rate * 1

        return loss


    def test(self):
        correct = 0
        count = 0
        for data, label in zip(self.test_data, self.test_label):

            data = data - np.max(data)
            # data = data / np.sum(data)

            f_ = np.dot(self.weight, data) + self.bias
            y_ = 1 / (1 + np.exp(-f_))
            y_ = 1 if y_[0] > 0.5 else 0
            if y_ == label:
                correct += 1
            count += 1

        return correct / count


    def update_plot(self, its, acc):
        # https://blog.csdn.net/yc_1993/article/details/54933751
        #if len(self.plot_x) >= 20:
        #    self.plot_x.pop(0)
        #    self.plot_y.pop(0)
        self.plot_x.append(its)
        self.plot_y.append(acc)

        self.ax1.set_xlim(min(self.plot_x), max(self.plot_x) + 1)
        # self.ax1.set_ylim(min(self.plot_y), max(self.plot_y) + 0.2)
        self.ax1.set_ylim(0, 1)
        self.line.set_data(self.plot_x, self.plot_y)
        plt.pause(0.001)
        self.ax1.figure.canvas.draw()



def load_data():
    dataset = sklearn.datasets.load_breast_cancer(True)

    train_data = dataset[0][:-100]
    train_label = dataset[1][:-100]
    test_data = dataset[0][-100:]
    test_label = dataset[1][-100:]

    return train_data, train_label, test_data, test_label



def main():
    train_data, train_label, test_data, test_label = load_data()
    model = Logistic(train_data.shape[1], train_data, train_label, test_data, test_label,
                     learn_rate=5e-7, learn_rate_decay=0.8, reg_rate=1e-5, its=1000000000, step=100, plot=True)
    model.train()


if __name__ == "__main__":
    main()