from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy
import math

TRAIN_BACKUP_DIR = '/tmp/model.ckpt'
MNIST_READ_DIR = '/tmp/tensorflow/mnist/input_data'
LOGS_DIR = '/tmp/log.txt'
LOGS_DIR_TB = '/tmp/logBoard1'
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0

layer1_patches_count = 16
layer2_patches_count = 32
layer3_patches_count = 64
connected_layer_size = 1024


def learningRate(i):
    return min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)


class NeuralNetworkMnist:
    def __init__(self, app):
        self.app = app
        self.mnist = input_data.read_data_sets(MNIST_READ_DIR, one_hot=True)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.correct_result = tf.placeholder(tf.float32, [None, 10])
            self.X = tf.placeholder(tf.float32, [None, 28 * 28])
            self.learning_rate = tf.placeholder(tf.float32)

            reshapedX = tf.reshape(self.X, shape=[-1, 28, 28, 1])
            weight1 = tf.Variable(tf.truncated_normal([6, 6, 1, layer1_patches_count], stddev=0.1), name="weight1")
            bias1 = tf.Variable(tf.ones([layer1_patches_count]) / 10, name="bias1")

            # layer 14*14*8
            weight2 = tf.Variable(tf.truncated_normal([5, 5, layer1_patches_count, layer2_patches_count], stddev=0.1), name="weight2")
            bias2 = tf.Variable(tf.ones([layer2_patches_count]) / 10, name="bias2")

            # layer 7*7*16
            weight3 = tf.Variable(tf.truncated_normal([4, 4, layer2_patches_count, layer3_patches_count], stddev=0.1),name="weight3")
            bias3 = tf.Variable(tf.ones([layer3_patches_count]) / 10, name="bias3")

            # fully connected
            weight4 = tf.Variable(tf.truncated_normal([7*7*layer3_patches_count, connected_layer_size], stddev=0.1),name="weight4")
            bias4 = tf.Variable(tf.ones([connected_layer_size]) / 10, name="bias4")

            # layer for softmax (as we have 10 digits. Bias for softmax starts from 0
            weight5 = tf.Variable(tf.truncated_normal([connected_layer_size, 10], stddev=0.1), name="weight5")
            bias5 = tf.Variable(tf.zeros([10]) / 10, name="bias5")

            Y1 = tf.nn.relu(tf.nn.conv2d(reshapedX, weight1, strides=[1, 1, 1, 1], padding='SAME') + bias1)
            Y2 = tf.nn.relu(tf.nn.conv2d(Y1, weight2, strides=[1, 2, 2, 1], padding='SAME') + bias2)
            Y3 = tf.nn.relu(tf.nn.conv2d(Y2, weight3, strides=[1, 2, 2, 1], padding='SAME') + bias3)

            Y3_one_hot = tf.reshape(Y3, shape=[-1, 7*7*layer3_patches_count])

            Y4 = tf.nn.relu(tf.matmul(Y3_one_hot, weight4) + bias4)

            self.Y = tf.nn.softmax(tf.matmul(Y4, weight5) + bias5)
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.correct_result)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

            correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.correct_result, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.saver = tf.train.Saver()

            # tf.scalar_summary("cost", self.cross_entropy)
            tf.scalar_summary("accuracy", self.accuracy)
            self.summary_op = tf.merge_all_summaries()

            assert self.cross_entropy.graph is self.graph
        self.writer = tf.train.SummaryWriter(LOGS_DIR_TB, graph=self.graph)
        self.session = tf.Session(graph=self.graph)

    def train(self, save=True, test=True):
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            for i in range(20000):
                # Data from MNIST
                correct_images, correct_labels = self.mnist.train.next_batch(50)
                self.app.updateInputImages(correct_images)
                self.app.updateIteration(i)
                train_data = {self.X: correct_images, self.correct_result: correct_labels, self.learning_rate: 0.0001}
                summary, _ = session.run([self.summary_op, self.train_step], train_data)
                self.writer.add_summary(summary, i)
                if test and i % 50000 == 0:
                    f = open(LOGS_DIR, "a")
                    acc_value = session.run(self.accuracy, feed_dict={self.X: self.mnist.test.images, self.correct_result: self.mnist.test.labels})
                    print("Step ", i, "Accuracy at ", str(acc_value), file=f)
                    f.close()

            if save:
                save_path = self.saver.save(session, TRAIN_BACKUP_DIR)
                print("Model saved in file: %s" % save_path)

    def continue_train(self, save=True, test=True):
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            self.saver.restore(session, TRAIN_BACKUP_DIR)
            print("Model restored.")
            for i in range(20000):
                # Data from MNIST
                correct_images, correct_labels = self.mnist.train.next_batch(50)
                self.app.updateInputImages(correct_images)
                self.app.updateIteration(i)
                train_data = {self.X: correct_images, self.correct_result: correct_labels, self.learning_rate: 0.0001}
                session.run(self.train_step, train_data)

    def resultForImage(self, pixels):
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, TRAIN_BACKUP_DIR)
            print("Model restored.")
            feed_dict = {self.X: pixels}
            #value = session.run(self.Y, feed_dict)
            value1 = session.run(tf.argmax(self.Y, 1), feed_dict)

            #self.app.resultNumberLabel.config(text=str(value))
            self.app.resultNumberLabel.config(text=str(value1))

            correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.correct_result, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(session.run(accuracy, feed_dict={self.X: self.mnist.test.images,
                                                   self.correct_result: self.mnist.test.labels}))


