from .utils import logits_to_one_hot_encoding
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape

class CNNClassifier:
    """
    A Convolutional Neural Network (CNN) classifier using Keras, customized for binary classification tasks.

    This class wraps a Keras Sequential model with a specific architecture suitable for image classification tasks.
    It includes a custom `predict` method that outputs one-hot encoded predictions, and other standard Keras model
    methods are accessible as well. This was done to override the need for the SciKeras wrappers that is frequently
    incompatable with google collab versions of Keras & Tensorflow. Feel free to modify as needed.

    Attributes:
        num_epochs (int): The number of training epochs.
        layers (int): The number of convolutional layers in the model.
        dropout (float): The dropout rate used in dropout layers for regularization.
        model (keras.models.Sequential): The underlying Keras Sequential model.

    Methods:
        build_model(): Constructs the CNN model with the specified architecture and compiles it.

        fit(*args, **kwargs): Trains the model. Accepts arguments compatible with the Keras `fit` method.

        predict(*args, **kwargs): Predicts labels for the input data. Converts the softmax output of the model
                                  to one-hot encoded format using `logits_to_one_hot_encoding`. Necessary to match
                                  accuracy_score function expected arguments.

        predict_proba(*args, **kwargs): Predicts labels for the input data and returns the raw output of the softmax.
                                        Used when wanting the inspect the raw probabilistic scoring of the model.

    Usage:
        cnn_classifier = CNNClassifier(num_epochs=30, layers=4, dropout=0.5)
        cnn_classifier.fit(X_train, y_train)
        predictions = cnn_classifier.predict(X_test)

    Note:
        The `__getattr__` method is overridden to delegate attribute access to the underlying Keras model,
        except for the `predict` method which is customized.
    """
    def __init__(self, num_epochs=30, layers=4, dropout=0.5):
        self.num_epochs = num_epochs
        self.layers = layers
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Reshape((32, 32, 3)))

        for i in range(self.layers):
          model.add(Conv2D(32, (3, 3), padding='same'))
          model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        opt = keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, epochs=self.num_epochs, batch_size=10, verbose=2, **kwargs)

    #NOTE: WRITTEN TO RETURN ONE HOT ENCODINGS FOR ACCURACY
    def predict(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return logits_to_one_hot_encoding(predictions)

    def predict_proba(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return predictions

    def __getattr__(self, name):
        if name != 'predict' and name != 'predict_proba':
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
