import abc  # Abstract and Interface module
import numpy as np
import pickle

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
                 activate_function=None):

        self.step = 0

        self.input_size = input_size
        self.output_size = output_size
        self.size = size
        self.batch_size = batch_size

        self.activate = Tanh()
        if activate_function == 'sigmoid':
            self.activate = Sigmoid()
        elif activate_function == 'relu':
            self.activate = ReLU()

        # randn(): Gaussian distribution
        self.input_W = np.random.randn(size, input_size + 1)    # size * input_size
        self.input_W[:, -1] = 0
        self.input_weight_delta = np.zeros(self.input_W.shape)

        self.output_W = np.random.randn(output_size, size + 1)     # output_size * size
        self.output_W[:, -1] = 0
        self.output_weight_delta = np.zeros(self.output_W.shape)


    def train(self, input, target, rate=0.1, M=0.8):
        if self.step == 0:
            self.rate = rate

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
            input_0 = np.append(np.array(input_0), 1)
            output_with_softmax, output_without_softmax = self.__forward(input_0)
            loss = softmax_cross_entropy_with_logits(output_without_softmax, self.target)
            i, o = self.backward(input_0, output_with_softmax, target_0, M)
            self.train_loss += loss
            self.input_weight_delta += i
            self.output_weight_delta += o
        self.train_loss = self.train_loss / self.batch_size
        self.update(self.rate,
                    self.input_weight_delta / self.batch_size,
                    self.output_weight_delta / self.batch_size)

        self.step += 1
        if self.step % 500 == 0:
            self.rate = self.rate * 0.9

        return self.train_loss


    def __forward(self, input):
        # input layer's output become hidden1
        hidden = np.dot(self.input_W, input)
        self.hidden = self.activate.function(hidden)

        output = np.dot(self.output_W, np.append(self.hidden, 1))

        return softmax(output), output


    def backward(self, input, output, target, M):
        # http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

        out_error = target - output    # output error, shape is output_size * 1

        in_error = np.dot(self.output_W.T, out_error)    # layer 1 errors

        # update input weight
        input_delta = in_error[0:-1] * self.activate.diff(self.hidden)
        input_weight_delta = np.dot(input_delta.reshape(-1, 1), input.reshape(1, -1))

        # update output weight
        # loss = -np.sum(label * np.log(softmax(o)))
        # the loss derivative is: softmax(o) - label
        output_delta = out_error * (-out_error)
        output_weight_delta = np.dot(output.reshape(1, -1), output_delta.reshape(-1, 1))

        return (input_weight_delta, output_weight_delta)


    def update(self, rate, input_weight_delta, output_weight_delta):
        self.input_W = self.input_W + rate * input_weight_delta
        self.output_W = self.output_W + rate * output_weight_delta


    def predict(self, input):
        input = np.array(input)  # change list to np array
        return self.__forward(np.append(input, 1))


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


def testMLP():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    batch_size = 64
    mlp = MLP(784, 10, 512, batch_size, activate_function='relu')

    train_loss = 0
    for step in range(20000):
        batch_x, batch_y = mnist.train.next_batch(batch_size, shuffle=True)
        train_loss += mlp.train(batch_x, batch_y)

        if step % 200 == 0:
            # Display logs per epoch step
            if step != 0:
                train_loss = train_loss / 200
            print("Step:", '%04d' % step, "train_loss={:.9f}".format(train_loss))

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