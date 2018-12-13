import abc  # Abstract and Interface module
import numpy as np
import pickle
import os
import random

class ActivateFunction(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = "Activate Function"

    @abc.abstractmethod
    def function(self, input):
        pass

    @abc.abstractmethod
    def diff(self, input):
        pass


class Sigmoid(ActivateFunction):
    def __init__(self):
        self.name = "Sigmoid function"

    def function(self, input):
        return 1 / (1 + np.exp(-input))

    def diff(self, input):
        return self.function(input) * (1 - self.function(input))


class ReLU(ActivateFunction):
    def __init__(self):
        self.name = "ReLU function"

    def function(self, input):
        return np.maximum(0, input)

    def diff(self, input):
        array = np.copy(input)
        array[array>0] = 1
        array[array<=0] = 0
        return array


class Tanh(ActivateFunction):
    def __init__(self):
        self.name = "tanh function"

    def function(self, input):
        return np.tanh(input)

    def diff(self, input):
        return 1 - np.square(np.tanh(input))


def softmax(input):
    input = input - np.max(input)
    exps = np.exp(input)
    return exps / np.sum(exps)


def softmax_cross_entropy_with_logits(logit, label):
    # log(softmax(zi)) = zi - log(sum(exp(zj))) ~= zi - max(z)
    ls = logit - np.max(logit)
    ls_1 = (1-logit) - np.max(1-logit)

    # -np.sum(label * np.log(softmax(o)) + (1-label) * np.log(softmax(1-o)))
    return -np.sum(label * ls + (1 - label) * ls_1)


class MLP(object):
    def __init__(self,
                 input_size,
                 output_size,
                 size=1024,
                 batch_size=1,
                 learn_rate=0.1,
                 dropout_input=0.0,
                 dropout_hidden=0.0,
                 activate_function=None):

        self.step = 0

        self.input_size = input_size
        self.output_size = output_size
        self.size = size
        self.batch_size = batch_size
        self.rate = learn_rate

        if dropout_input == 0 and dropout_hidden == 0:
            self.dropout = False
        else:
            self.dropout = True
            self.dropout_input = dropout_input
            self.dropout_hidden = dropout_hidden

        self.activate = Tanh()
        if activate_function == 'sigmoid':
            self.activate = Sigmoid()
        elif activate_function == 'relu':
            self.activate = ReLU()

        # randn(): Gaussian distribution
        self.input_W = np.random.randn(size, input_size + 1)    # size * input_size
        # random_sample(): uniform distribution
        # self.input_W = np.random.random_sample((size, input_size + 1))    # size * input_size
        self.input_W[:, -1] = 0
        self.input_weight_delta = np.zeros(self.input_W.shape)

        self.output_W = np.random.randn(output_size, size + 1)     # output_size * size
        # self.output_W = np.random.random_sample((output_size, size + 1))     # output_size * size
        self.output_W[:, -1] = 0
        self.output_weight_delta = np.zeros(self.output_W.shape)

        self.dropout_vec_i = np.ones(self.input_size + 1, dtype=int)
        self.dropout_vec_h = np.ones(self.size + 1, dtype=int)


    def train(self, input, target):
        if len(input) != self.batch_size:
            raise ValueError("batch size error, %d != %d"
                             % (len(input), self.batch_size))
        if len(input[0]) != self.input_size:
            raise ValueError("input size error, %d != d"
                             % (len(input[0]), self.input_size))
        if len(target[0]) != self.output_size:
            raise ValueError("target size error, %d != d"
                             % (len(target[0]), self.output_size))
        self.input = input
        self.target = np.array(target)


        self.input_weight_delta[:] = 0
        self.output_weight_delta[:] = 0
        self.train_loss = 0
        for input_0, target_0 in zip(self.input, self.target):

            if self.dropout:
                # Dropout
                dropout_vec_i = [int(1) if random.random() > self.dropout_input else int(0) for _ in range(self.input_size)]
                dropout_vec_i.append(int(1))
                self.dropout_vec_i = np.array(dropout_vec_i)
                dropout_vec_h = [int(1) if random.random() > self.dropout_hidden else int(0) for _ in range(self.size)]
                dropout_vec_h.append(int(1))
                self.dropout_vec_h = np.array(dropout_vec_h)

            input_0 = np.append(np.array(input_0), 1)
            output_with_softmax, output_without_softmax = self.__forward(input_0)
            loss = softmax_cross_entropy_with_logits(output_without_softmax, self.target)
            i, o = self.__backward(input_0, output_with_softmax, target_0)
            self.train_loss += loss
            self.input_weight_delta += i
            self.output_weight_delta += o
        self.train_loss = self.train_loss / self.batch_size
        self.__update(self.rate,
                    self.input_weight_delta / self.batch_size,
                    self.output_weight_delta / self.batch_size)

        self.step += 1
        if self.step % 800 == 0:
            self.rate = self.rate * 0.6

        return self.train_loss


    def __forward(self, input):
        # input layer's output become hidden1
        hidden = np.dot(self.input_W, input * self.dropout_vec_i)
        self.hidden = self.activate.function(hidden)

        output = np.dot(self.output_W, np.append(self.hidden, 1) * self.dropout_vec_h)

        return softmax(output), output


    def __backward(self, input, output, target):
        # http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

        out_error = target - output    # output error, shape is output_size * 1


        # update output weight
        # loss = -np.sum(label * np.log(softmax(o)))
        # the loss derivative is: softmax(o) - label
        output_delta = out_error * (-out_error)
        hidden_tmp = np.append(self.hidden, 1) * self.dropout_vec_h
        output_weight_delta = np.dot(output_delta.reshape(-1, 1), hidden_tmp.reshape(1, -1))

        # in_error = np.dot(self.output_W.T, output_delta)
        in_error = np.dot(self.output_W.T, out_error)

        # update input weight
        input_delta = in_error[0:-1] * self.activate.diff(self.hidden)
        input_tmp = input * self.dropout_vec_i
        input_weight_delta = np.dot(input_delta.reshape(-1, 1), input_tmp.reshape(1, -1))

        return (input_weight_delta, output_weight_delta)


    def __update(self, rate, input_weight_delta, output_weight_delta):
        self.input_W = self.input_W + rate * input_weight_delta
        self.output_W = self.output_W + rate * output_weight_delta


    def predict(self, input):
        input = np.array(input)  # change list to np array
        return self.__forward(np.append(input, 1))


    def print_parameter(self):
        np.set_printoptions(threshold='nan')
        f = open(str(self.step)+'mlp.txt', mode="w", encoding='UTF-8')
        f.write("input weight\n")
        np.savetxt(f, self.input_W)
        f.write('\n')
        f.write("output weight\n")
        np.savetxt(f, self.output_W)
        f.write('\n\n')
        f.write("hidden\n")
        np.savetxt(f, self.hidden)
        f.write("output with softmax\n")
        np.savetxt(f, self.output_with_softmax)
        f.write("\n")
        f.write("output\n")
        np.savetxt(f, self.output_without_softmax)
        f.close()


    def save(self, model, pickle_file=None):
        if isinstance(model, MLP):
            print("Saving...")
            if pickle_file is None:
                pickle.dump(model, open("MLP.pkl", mode="wb"))
            else:
                pickle.dump(model, open(pickle_file, mode="wb"))
            print("Already saved")
        else:
            raise("Need a MLP object")


    def restore(self, pickle_file=None):
        if pickle_file is None:
            return pickle.load(open("MLP.pkl", mode="rb"))
        else:
            return pickle.load(open(pickle_file, mode="rb"))

def read_mnist_data():
    data_root = "/tmp/data/notmnist"
    pickle_file = os.path.join(data_root, 'notMNIST.pickle')
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        train_dataset = data['train_dataset']
        train_labels = data['train_labels']
        valid_dataset = data['valid_dataset']
        valid_labels = data['valid_labels']
        test_dataset = data['test_dataset']
        test_labels = data['test_labels']
        del data  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
    return (train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)


def train_next_batch(mnist, batch_size, cursor):
    batch_x = np.zeros((batch_size, mnist[0].shape[1], mnist[0].shape[2]))
    batch_y = np.zeros(batch_size)
    if len(cursor) < batch_size:
        cursor.extend(range(batch_size - len(cursor)))
    for i in range(batch_size):
        batch_x[i] = mnist[0][cursor.pop(),:]
        batch_y[i] = mnist[1][cursor.pop()]
    batch_x = batch_x.reshape(batch_size, -1)
    batch_y = (np.arange(10)==batch_y[:,None]).astype(np.integer)
    return batch_x, batch_y


def valid_or_test_next_batch(mnist):
    batch_x = mnist[0].reshape(len(mnist[0]), -1)
    batch_y = mnist[1]
    batch_y = (np.arange(10)==batch_y[:,None]).astype(np.integer)
    return (batch_x, batch_y)


def testMLP():
    image_size = 28
    num_labels = 10
    batch_size = 32

    mnist = read_mnist_data()
    mlp = MLP(image_size * image_size,
              num_labels,
              512,
              batch_size,
              learn_rate=0.01,
              dropout_input=0.8,
              dropout_hidden=0.5,
              activate_function='relu')

    valid_data = valid_or_test_next_batch((mnist[2], mnist[3]))
    test_data = valid_or_test_next_batch((mnist[4], mnist[5]))

    train_length = len(mnist[0])
    train_loss = 0
    train_cursor = [x for x in range(train_length)]
    random.shuffle(train_cursor)
    epoch = 0
    for step in range(10 * train_length):
        batch_x, batch_y = train_next_batch((mnist[0], mnist[1]), batch_size, train_cursor)

        if len(train_cursor) <= 0:
            epoch += 1
            print("Epoch %d finish\n" % epoch)
            train_cursor = [x for x in range(train_length)]
            random.shuffle(train_cursor)

        train_loss += mlp.train(batch_x, batch_y)
        # mlp.print_parameter()

        if step % 200 == 0:
            # Display logs per epoch step
            if step != 0:
                train_loss = train_loss / 200
            print("Step:", '%04d, ' % step, "learning_rate:", "%.9f, " % mlp.rate, "train_loss={:.9f}".format(train_loss))

            valid_correct_prediction = 0
            valid_count = 0
            for x, y in zip(valid_data[0], valid_data[1]):
                # Test model
                pred, _ = mlp.predict(x)
                valid_correct_prediction += 1 if np.argmax(pred) == np.argmax(y) else 0
                valid_count += 1
            # Calculate accuracy
            valid_accuracy = valid_correct_prediction / valid_count
            print("Valid accuracy: %f" % valid_accuracy, end=", ")

            test_correct_prediction = 0
            test_count = 0
            for x, y in zip(test_data[0], test_data[1]):
                # Test model
                pred, _ = mlp.predict(x)
                test_correct_prediction += 1 if np.argmax(pred) == np.argmax(y) else 0
                test_count += 1
                # Calculate accuracy
            test_accuracy = test_correct_prediction / test_count
            print("test accuracy: %f" % test_accuracy)

            train_loss = 0

    print("Optimization Finished!")

def testMLP_tf():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    batch_size = 32
    mlp = MLP(784, 10, 512, batch_size, learn_rate=0.01, dropout_input=0.8, dropout_hidden=0.5, activate_function='relu')

    train_loss = 0
    for step in range(20000):
        batch_x, batch_y = mnist.train.next_batch(batch_size, shuffle=True)
        train_loss += mlp.train(batch_x, batch_y)

        if step % 200 == 0:
            # Display logs per epoch step
            if step != 0:
                train_loss = train_loss / 200
            print("Step:", '%04d, ' % step, "learning_rate:", "%.9f, " % mlp.rate, "train_loss={:.9f}".format(train_loss))

            valid_correct_prediction = 0
            valid_count = 0
            for x, y in zip(mnist.validation.images, mnist.validation.labels):
                # Test model
                pred, _ = mlp.predict(x)
                valid_correct_prediction += 1 if np.argmax(pred) == np.argmax(y) else 0
                valid_count += 1
            # Calculate accuracy
            valid_accuracy = valid_correct_prediction / valid_count
            print("Valid accuracy: %f" % valid_accuracy, end=", ")

            test_correct_prediction = 0
            test_count = 0
            for x, y in zip(mnist.test.images, mnist.test.labels):
                # Test model
                pred, _ = mlp.predict(x)
                test_correct_prediction += 1 if np.argmax(pred) == np.argmax(y) else 0
                test_count += 1
                # Calculate accuracy
            test_accuracy = test_correct_prediction / test_count
            print("test accuracy: %f" % test_accuracy)

            train_loss = 0

    print("Optimization Finished!")

testMLP()