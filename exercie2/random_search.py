import logging

logging.basicConfig(level=logging.WARNING)

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import argparse
import tensorflow as tf
import numpy as np

from cnn_mnist import mnist

def error(y_true, y_pred):
    return 1 - tf.keras.metrics.categorical_accuracy(y_true, y_pred)

class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist("./")

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        filter_size = config["filter_size"]
        epochs = budget

        # TODO: train and validate your convolutional neural networks here
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_filters, [filter_size, filter_size], padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(num_filters, [filter_size, filter_size], padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D([2, 2]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr), loss=tf.keras.losses.categorical_crossentropy, metrics=[error])
        model.fit(x=self.x_train, y=self.y_train, batch_size=batch_size, epochs=epochs)
        validation_error = model.evaluate(self.x_valid, self.y_valid)[1]
        # TODO: We minimize so make sure you return the validation error here
        return ({
            'loss': validation_error,  # this is the a mandatory field to run hyperband
            'info': {}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # TODO: Implement configuration space here. See https://github.com/automl/HpBandSter/blob/master/hpbandster/examples/example_5_keras_worker.py  for an example
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=0.0001, upper=0.1, default_value=0.1, log=True)
        num_filters = CSH.UniformIntegerHyperparameter('num_filters', lower=8, upper=64, default_value=16, log=True)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, default_value=64, log=True)
        filter_size = CSH.CategoricalHyperparameter('filter_size', [3,5])
        config_space.add_hyperparameters([learning_rate, num_filters, batch_size, filter_size])
        return config_space


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=6)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=50)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])


# Plots the performance of the best found validation error over time
all_runs = res.get_all_runs()
# Let's plot the observed losses grouped by budget,
import hpbandster.visualization as hpvis

hpvis.losses_over_time(all_runs)

import matplotlib.pyplot as plt
plt.savefig("random_search.png")

# TODO: retrain the best configuration (called incumbent) and compute the test error
x_train, y_train, x_valid, y_valid, x_test, y_test = mnist("./")
x_total = np.append(x_train, x_valid, axis=0)
y_total = np.append(y_train, y_valid, axis=0)
lr = id2config[incumbent]['config']['learning_rate']
filter_size = id2config[incumbent]['config']['filter_size']
num_filters = id2config[incumbent]['config']['num_filters']
batch_size = id2config[incumbent]['config']['batch_size']

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(num_filters, [filter_size, filter_size], padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(num_filters, [filter_size, filter_size], padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D([2, 2]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr), loss=tf.keras.losses.categorical_crossentropy, metrics=[error])
model.fit(x=x_total, y=y_total, batch_size=batch_size, epochs=6)
error = model.evaluate(x_test, y_test)[1]

print('test error: ' +str(error))
