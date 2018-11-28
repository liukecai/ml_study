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
    return np.exp(input) / np.sum(np.exp(input))


def softmax_cross_entropy_with_logits(logit, label):
    s = softmax(logit)
    return -np.sum(label * np.log(s), axis=0)


class MLP(object):
    def __init__(self,
                 input_size,
                 output_size,
                 size=1024,
                 activate_function=None):

        self.step = 0

        self.input_size = input_size
        self.output_size = output_size
        self.size = size

        self.activate = Tanh()
        if activate_function == 'sigmoid':
            self.activate = Sigmoid()
        elif activate_function == 'relu':
            self.activate = ReLU()

        # randn(): Gaussian distribution
        self.input_W = np.random.randn(size, input_size)    # size * input_size

        self.output_W = np.random.randn(output_size, size)     # output_size * size


    def train(self, input, target, rate=0.1, M=0.8):
        if self.step == 0:
            self.rate = rate

        if len(input) != self.input_size:
            raise ValueError("input size error, %d != d"
                             % (len(input), self.input_size))
        if len(target) != self.output_size:
            raise ValueError("target size error, %d != d"
                             % (len(target), self.output_size))
        self.input = np.array(input)    # change list to np array
        self.target = np.array(target)

        self.output = self.__forward(self.input)
        loss = softmax_cross_entropy_with_logits(self.output, self.target)
        self.backward(self.target, self.rate, M)

        self.step += 1
        if self.step % 500 == 0:
            self.rate = self.rate * 0.9

        return loss


    def __forward(self, input):
        # input layer's output become hidden1
        hidden = np.dot(self.input_W, input)
        self.hidden = self.activate.function(hidden)

        output = np.dot(self.output_W, self.hidden)
        output = self.activate.function(output)

        return output


    def backward(self, target, rate, M):
        # http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

        out_error = target - self.output    # output error, shape is output_size * 1

        in_error = np.dot(self.output_W.T, out_error)    # layer 1 errors

        # update input weight
        input_delta = rate * in_error * self.activate.diff(self.hidden)
        input_weight_delta = np.dot(input_delta.reshape(-1, 1), self.input.reshape(1, -1))

        # update output weight
        output_delta = rate * out_error * self.activate.diff(self.output)
        output_weight_delta = np.dot(self.output.reshape(1, -1), output_delta.reshape(-1, 1))

        self.input_W = self.input_W + input_weight_delta

        self.output_W = self.output_W + output_weight_delta


    def predict(self, input):
        input = np.array(input)  # change list to np array
        return self.__forward(input)


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

    mlp = MLP(784, 10, 512, activate_function='sigmoid')

    for step in range(20000):
        batch_x, batch_y = mnist.train.next_batch(1)
        train_loss = mlp.train(batch_x[0], batch_y[0])

        if step % 200 == 0:
            # Display logs per epoch step
            print("Step:", '%04d' % step, "train_loss={:.9f}".format(train_loss))
            correct_prediction = 0
            count = 0
            for x, y in zip(mnist.validation.images, mnist.validation.labels):
                # Test model
                out = mlp.predict(x)
                pred = softmax(out)  # Apply softmax to logits
                correct_prediction += 1 if np.argmax(pred) == np.argmax(y) else 0
                count += 1
            # Calculate accuracy
            accuracy = correct_prediction / count
            print("Accuracy: %f" % accuracy)

    print("Optimization Finished!")

testMLP()