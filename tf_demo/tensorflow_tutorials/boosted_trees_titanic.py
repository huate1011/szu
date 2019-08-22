import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

dftrain = pd.read_csv('titanic/titanic_train.csv')
dfeval = pd.read_csv('titanic/titanic_eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


def plot_titanic(df_data):
    df_data.age.hist(bins=20)
    plt.show()
    df_data.sex.value_counts().plot(kind="barh")
    plt.show()
    df_data['class'].value_counts().plot(kind="barh")
    plt.show()
    df_data.embark_town.value_counts().plot(kind="barh")
    plt.show()


# plot_titanic(dftrain)
# ax = (pd.concat([dftrain, y_train], axis=1)\
#   .groupby('sex')
#   .survived
#   .mean()
#   .plot(kind='barh'))
# ax.set_xlabel('% survive')
# plt.show()


num_examples = len(y_train)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat(n_epochs)
        dataset = dataset.batch(num_examples)
        return dataset
    return input_fn


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


def one_hot_cat_column(feature_name, vocab):
    return fc.indicator_column(
        fc.categorical_column_with_vocabulary_list(feature_name,
                                                   vocab))


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(fc.numeric_column(feature_name,
                                             dtype=tf.float32))


linear_est = tf.estimator.LinearClassifier(feature_columns)
linear_est.train(train_input_fn, max_steps=100)
results = linear_est.evaluate(eval_input_fn)

print("Accuracy: ", results['accuracy'])
print("Dummy model: ", results['accuracy_baseline'])


print("################ Boosted Trees")
n_batches = 1
params = {
  'n_trees': 50,
  'max_depth': 3,
  'n_batches_per_layer': 1,
  # You must enable center_bias = True to get DFCs. This will force the model to
  # make an initial prediction before using any features (e.g. use the mean of
  # the training labels for regression or log odds for classification when
  # using cross entropy loss).
  'center_bias': True
}

boosted_trees = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
boosted_trees.train(train_input_fn, max_steps=100)
results = boosted_trees.evaluate(eval_input_fn)
print("Accuracy: ", results['accuracy'])
print("Dummy model: ", results['accuracy_baseline'])


def make_inmemory_train_input_fn(X, y):
    def input_fn():
        return dict(X), y
    return input_fn


print("################ Boosted Trees in memory")
train_input_fn = make_inmemory_train_input_fn(dftrain, y_train)
estimator = tf.contrib.estimator.boosted_trees_classifier_train_in_memory(train_input_fn, feature_columns)
# eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)
print(estimator.evaluate(eval_input_fn)['accuracy'])