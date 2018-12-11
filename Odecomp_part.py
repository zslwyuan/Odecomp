from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import time
tfe.enable_eager_execution()


mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

tf.set_random_seed(1)


verbose_interval = 10

# some global variables
global mask
global rank_def
global rank_fade
global logfile
global accuracy_g

logfile = open("output_incremental_cvx","w")
# firsr layer=28*28=784, hidden layer=150
b00 = tf.get_variable("b00", shape=[150])
W1 = tf.get_variable("W1", shape=[150, 10])
b1 = tf.get_variable("b1", shape=[10])


def finetune_mlp_decomp(step, x, y, test_x, test_y, is_train = True):

    global mask

    hidden1 = tf.matmul(x, tf.matmul(U_t, V_t)) + b00
    hidden1 = tf.nn.relu(hidden1)

    if is_train:
        hidden1 = tf.nn.dropout(hidden1, keep_prob = 0.8)
    logits = tf.matmul(hidden1, W1) + b1

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)

    if step % verbose_interval == 0:
        predict = tf.argmax(logits, 1).numpy()
        target = np.argmax(y, 1)
        accuracy = np.sum(predict == target)/len(target)
        print("TRAIN: \t step {}:\tloss = {}\taccuracy = {}".format(step, loss.numpy(), accuracy))
        test_mlp_decomp(test_x, test_y)
        end = time.time()
        print(end - start)
    return loss


def test_mlp_decomp(x, y):
    global mask
    global accuracy_g
    global rank_def
    hidden1 =  tf.matmul(x, tf.matmul(U_t, V_t)) + b00
    hidden1 = tf.nn.relu(hidden1)

    logits = tf.matmul(hidden1, W1) + b1

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)

    predict = tf.argmax(logits, 1).numpy()
    target = np.argmax(y, 1)
    accuracy = np.sum(predict == target)/len(target)
    accuracy_g = accuracy

    print("TEST: \tloss = {}\taccuracy = {}".format(loss.numpy(), accuracy))
    print("TEST: accuracy = {}".format(accuracy),file=logfile)


def test_mlp(x, y):
    global mask
    global accuracy_g
    global logfile

    hidden1 = tf.matmul(x, tf.matmul(U_t, V_t)) + b00
    hidden1 = tf.nn.relu(hidden1)

    logits = tf.matmul(hidden1, W1) + b1

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)

    predict = tf.argmax(logits, 1).numpy()
    target = np.argmax(y, 1)
    accuracy = np.sum(predict == target)/len(target)
    accuracy_g = accuracy

    print("TEST: \tloss = {}\taccuracy = {}".format(loss.numpy(), accuracy))
    print("TEST: accuracy = {}".format(accuracy),file=logfile)

def mlp(step, x, y, test_x, test_y, factor=1.0, is_train = True):

    if (factor>1.0):
        factor = 1.0

    global rank_def
    global rank_fade
    W00 = tf.matmul(U_t, V_t)
    # W00 = tf.matmul(U_t, tf.matmul(tf.diag(S_t), V_t))
    hidden1 = tf.matmul(x, W00) + b00
    hidden1 = tf.nn.relu(hidden1)

    if is_train:
        hidden1 = tf.nn.dropout(hidden1, keep_prob = 0.75)
    logits = tf.matmul(hidden1, W1) + b1

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss_diff = 0

   # s, u, v = tf.linalg.svd(W00)

    if (rank_fade>=rank_def):
        loss = tf.reduce_mean(loss)

    else:
        loss = (tf.reduce_mean(loss)+0.001*factor*tf.norm(tf.slice(U_t,[0,rank_fade],[784,rank_def-rank_fade]),1)+
                                   0.001*factor*tf.norm(tf.slice(V_t,[rank_def-rank_fade,0],[rank_fade,150]),1))

    if step % verbose_interval == 0:
        predict = tf.argmax(logits, 1).numpy()
        target = np.argmax(y, 1)
        accuracy = np.sum(predict == target)/len(target)

        print("TRAIN: \t step {}:\tloss = {}\taccuracy = {}".format(step, loss.numpy(), accuracy))
        test_mlp(test_x, test_y)
        end = time.time()       
        print(end - start)
    return loss


optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)

batch_data, batch_label = mnist.train.next_batch(60000)
start = time.time()

tmp_array = []
for i in range(30):
    tmp_array.append(1.0)
sess = tf.InteractiveSession()


print("Pre-adjust!!!!==============================")
print("Pre-adjust!!!!==============================",file=logfile)

rank_def = 150
rank_fade = rank_def

U_t = tf.get_variable("U_t", shape=[784,rank_def])
# S_t = tf.get_variable("S_t", shape=[rank_def])
V_t = tf.get_variable("V_t", shape=[rank_def,150])

# pre-adjust weights
for step in range(2000):
    which = ((step) % 5)*10000
    next_r = which + 10000
    optimizer.minimize(lambda: mlp(step, batch_data[which:next_r], batch_label[which:next_r], batch_data[50000:], batch_label[50000:]))

print("Fading!!!!==============================")
print("Fading!!!!==============================",file=logfile)

# rank fading
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
rank_fade = 15
for step in range(2000):
    which = ((step) % 5)*10000
    next_r = which + 10000
    optimizer.minimize(lambda: mlp(step, batch_data[which:next_r], batch_label[which:next_r], batch_data[50000:], batch_label[50000:],step/1000.0))


# rank truncation
rank_def = 15
U_t = tf.get_variable("U_t", initializer=tf.slice(U_t,[0,0],[784,rank_def]))
# S_t = tf.get_variable("S_t", initializer=S_t[0:rank_def])
V_t = tf.get_variable("V_t", initializer=tf.slice(V_t,[0,0],[rank_def,150]))

print("Truncation!!!!==============================")
print("Truncation!!!!==============================",file=logfile)

print("U_t===",U_t.shape)
# print("S_t===",S_t.shape)
print("V_t===",V_t.shape)

test_mlp_decomp(batch_data[50000:], batch_label[50000:])

optimizer = tf.train.AdamOptimizer(learning_rate = 2.5e-4)

for step in range(5000):
    which = ((step) % 5)*10000
    next_r = which + 10000
    mask = tf.constant(tmp_array)
    optimizer.minimize(lambda: finetune_mlp_decomp(step, batch_data[which:next_r], batch_label[which:next_r], batch_data[50000:], batch_label[50000:]))

end = time.time()
print(end - start)
