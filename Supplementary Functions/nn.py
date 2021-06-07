import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mean_averaged_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def create_model(num_cols):
    # Initialising the ANN
    model = Sequential()
    tf.random.set_seed(7)

    # Adding the input layer and the first hidden layer
    model.add(Dense(30, kernel_initializer='uniform', activation='relu', input_dim=num_cols))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(epsilon=1e-06, momentum=0.8, weights=None))

    # Adding the second hidden layer
    model.add(Dense(15, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(epsilon=1e-06, momentum=0.8, weights=None))

    # Adding the third hidden layer
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(epsilon=1e-06, momentum=0.8, weights=None))

    # Adding the output layer
    model.add(Dense(1, kernel_initializer='uniform'))

    return model


def train_model_rmse(X_train, y_train, num_cols):
    # Create model
    model = create_model(num_cols)
    tf.random.set_seed(7)

    # Define early stopping
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='min', verbose=1)
    checkpoint = ModelCheckpoint('model_best_weights.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)

    # Compiling the ANN
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss=root_mean_squared_error, optimizer=optimizer)

    # Fitting the ANN to the Training set
    model_history = model.fit(X_train.values.astype('float32'), y_train.values.astype('float32'),
                              validation_split=0.1, batch_size=64, epochs=1600, callbacks=[checkpoint])

    return model_history

def train_model_mae(X_train, y_train, num_cols):
    # Create model
    model = create_model(num_cols)
    tf.random.set_seed(7)

    # Define early stopping
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='min', verbose=1)
    checkpoint = ModelCheckpoint('model_best_weights_mae.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)

    # Compiling the ANN
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss=mean_averaged_error, optimizer=optimizer)

    # Fitting the ANN to the Training set
    model_history = model.fit(X_train.values.astype('float32'), y_train.values.astype('float32'),
                              validation_split=0.1, batch_size=64, epochs=1600, callbacks=[checkpoint])

    return model_history

def load_model(weights_path, num_cols):
    model = create_model(num_cols)
    model.load_weights(weights_path)

    return model