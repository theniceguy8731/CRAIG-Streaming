import tensorflow
# from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Input
# from tensorflow.keras.utils import np_util
import numpy as np
import time
from tensorflow.keras.regularizers import l2
import util

############################################
# Load and preprocess data
############################################
(X_train_full, Y_train_full), (X_test, Y_test) = tensorflow.keras.datasets.mnist.load_data()
# X_train_full = X_train_full.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
num_classes, smtk = 10, 0
# Y_train_full_nocat = Y_train_full
# Y_train_full = tensorflow.keras.utils.to_categorical(Y_train_full, num_classes)
Y_test = tensorflow.keras.utils.to_categorical(Y_test, num_classes)
granularity = 0.1

############################################
# Set Training Parameters
############################################
batch_size = 512
# subset, random = False, False  # all
subset, random = True, False  # greedy
# subset, random = True, True  # random
subset_size = .05 if subset else 1.0
epochs = 15
reg = 1e-4
runs = 1
save_subset = False
folder = f'./mnist'

############################################
# Define the model
############################################
model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(100, kernel_regularizer=l2(reg)))
model.add(Activation('sigmoid'))
model.add(Dense(10, kernel_regularizer=l2(reg)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

for i in range(int(100*granularity)):
    print(f'Iteration {i}')
    # loaded the unweighted new data
    nos_points=int(X_train_full.shape[0]*granularity)
    X_train = X_train_full[i*nos_points:(i+1)*nos_points]
    print(X_train.shape)
    Y_train = Y_train_full[i*nos_points:(i+1)*nos_points]
    W_train = np.ones((len(X_train), 1))
    X_train = X_train.reshape(nos_points, 784)
    Y_train_nocat = Y_train
    Y_train = tensorflow.keras.utils.to_categorical(Y_train, num_classes)

    # loading the data from store.npy
    try:
        store = np.load('store.npy')
        X_train = np.concatenate((X_train, store['X_train']))
        Y_train = np.concatenate((Y_train, store['Y_train']))
        W_train = np.concatenate((W_train, store['W_train']))
    except:
        pass
    # saving metrics
    train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    train_acc, test_acc = np.zeros((runs, epochs)), np.zeros((runs, epochs))
    train_time = np.zeros((runs, epochs))
    grd_time, sim_time, pred_time = np.zeros((runs, epochs)), np.zeros((runs, epochs)), np.zeros((runs, epochs))
    not_selected = np.zeros((runs, epochs))
    times_selected = np.zeros((runs, len(X_train)))
    best_acc = 0

    X_coreset =[]
    Y_coreset =[]
    W_coreset =[]

    for run in range(runs):
        grd = ''
        X_subset = X_train
        Y_subset = Y_train
        W_subset = W_train
        ordering_time, similarity_time, pre_time = 0, 0, 0
        loss_vec, acc_vec, time_vec = [], [], []
        for epoch in range(0, epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            num_batches = int(np.ceil(X_subset.shape[0] / float(batch_size)))

            for index in range(num_batches):
                X_batch = X_subset[index * batch_size:(index + 1) * batch_size]
                Y_batch = Y_subset[index * batch_size:(index + 1) * batch_size]
                W_batch = W_subset[index * batch_size:(index + 1) * batch_size]

                start = time.time()
                history = model.train_on_batch(X_batch, Y_batch, sample_weight=W_batch)
                train_time[run][epoch] += time.time() - start

            if subset:
                if random:
                    # indices = np.random.randint(0, len(X_train), int(subset_size * len(X_train)))
                    indices = np.arange(0, len(X_train))
                    np.random.shuffle(indices)
                    indices = indices[:int(subset_size * len(X_train))]
                    W_subset = np.ones(len(indices))
                else:
                    start = time.time()
                    _logits = model.predict(X_train)
                    pre_time = time.time() - start
                    features = _logits - Y_train
                    print(Y_train_nocat.shape)
                    print(int(subset_size * len(X_train)))
                    print(features.shape)

                    indices, W_subset, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                        int(subset_size * len(X_train)), features, 'euclidean', smtk, 0, False, Y_train_nocat)

                    W_subset = W_subset / np.sum(W_subset) * len(W_subset)  # todo

                if run == runs-1 and epoch == epochs-1:
                    X_coreset = np.asarray(X_train[indices])
                    Y_coreset = np.asarray(Y_train[indices])
                    W_coreset = np.asarray(W_subset)

                grd_time[run, epoch], sim_time[run, epoch], pred_time[run, epoch] = ordering_time, similarity_time, pre_time
                times_selected[run][indices] += 1
                not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
            else:
                pred_time = 0
                indices = np.arange(len(X_train))

            X_subset = X_train[indices, :]
            Y_subset = Y_train[indices]

            start = time.time()
            score = model.evaluate(X_test, Y_test, verbose=1)
            eval_time = time.time() - start

            start = time.time()
            score_loss = model.evaluate(X_train, Y_train, verbose=1)
            print(f'eval time on training: {time.time() - start}')

            test_loss[run][epoch], test_acc[run][epoch] = score[0], score[1]
            train_loss[run][epoch], train_acc[run][epoch] = score_loss[0], score_loss[1]
            best_acc = max(test_acc[run][epoch], best_acc)

            grd = 'random_wor' if random else 'grd_normw'
            print(f'run: {run}, {grd}, subset_size: {subset_size}, epoch: {epoch}, test_acc: {test_acc[run][epoch]}, '
                  f'loss: {train_loss[run][epoch]}, best_prec1_gb: {best_acc}, not selected %:{not_selected[run][epoch]}')

        print(f'Saving the results to {folder}_{subset_size}_{grd}_{runs}')
        np.savez(f'{folder}_{subset_size}_{grd}_{runs}_streaming',
                 # f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}',
                 train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                 train_time=train_time, grd_time=grd_time, sim_time=sim_time, pred_time=pred_time,
                 not_selected=not_selected, times_selected=times_selected)

    np.save('store.npy', {'X_train': X_coreset, 'Y_train': Y_coreset, 'W_train': W_coreset})
