from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

def get_mlp_model(
        hiddenLayerOne=784, 
        hiddenLayerTwo=256, 
        dropout=0.2, 
        learnRate=0.01
    ):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    model.add(Dense(hiddenLayerOne, activation='relu', input_shape=(hiddenLayerOne,)))
    model.add(Dropout(dropout))
    model.add(Dense(hiddenLayerTwo, activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learnRate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model