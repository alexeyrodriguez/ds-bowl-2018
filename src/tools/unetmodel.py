
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
import tensorflow as tf

def unet(channels, dropouts):
    
    def f(x):
        # downwards
        downwards = []
        for dropout, channel in zip(dropouts, channels):
            x = Conv2D(channel, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
            x = Dropout(dropout)(x)
            x = Conv2D(channel, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
            downwards.append(x)
            x = MaxPooling2D((2,2))(x)

        # We drop the last max pooling
        x = downwards[-1]
        downward_signals = downwards[:-1]
    
        # Iterate upwards
        for downward_signal, dropout, channel in list(zip(downward_signals, dropouts, channels))[::-1]:
            x = Conv2DTranspose(channel, (2, 2), strides=(2, 2), padding='same') (x)
            x = concatenate([x, downward_signal])
            x = Conv2D(channel, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)
            x = Dropout(dropout)(x)
            x = Conv2D(channel, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(x)

        return x
    
    return f

def u_net_model(width, height, channels):
    inputs = Input((height, width, channels))
    s = Lambda(lambda x: x / 255) (inputs)
    
    dropouts = [0.1, 0.1, 0.2, 0.2, 0.3]
    channels = [16, 32, 64, 128, 256]
    
    x = unet(channels, dropouts)(s)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


def mean_squared_error_masked(y_true, y_pred):
    mask = K.sign(K.abs(y_true))
    return K.sum(K.square(y_pred - y_true) * mask, axis=-1) / (K.sum(mask, axis=-1) + 0.1)

def u_net_model_ext(width, height, channels, centers_weight=None):
    inputs = Input((height, width, channels))
    s = Lambda(lambda x: x / 255) (inputs)
    
    dropouts = [0.1, 0.1, 0.2, 0.2, 0.3]
    #channels = [16, 32, 64, 128, 256]
    channels = [32, 64, 128, 256, 512]

    
    x = unet(channels[:2], dropouts[:2])(s)
    

    outputs = Conv2D(1, (1, 1), activation='sigmoid', name='mask') (x)

    # Predicting centers
    centers = Conv2D(1, (1, 1), activation='sigmoid', name='centers') (x)
    overlaps = Conv2D(1, (1, 1), activation='sigmoid', name='overlaps') (x)
    widths = Conv2D(1, (1, 1), activation='linear', name='widths') (x)
    heights = Conv2D(1, (1, 1), activation='linear', name='heights') (x)
    diffx = Conv2D(1, (1, 1), activation='linear', name='diffx') (x)
    diffy = Conv2D(1, (1, 1), activation='linear', name='diffy') (x)
    
    if centers_weight:
#        centers_loss = weighted_categorical_crossentropy([1.0, centers_weight])
        centers_loss = weighted_categorical_crossentropy([centers_weight, centers_weight])

    else:
        centers_loss = 'binary_crossentropy'
    centers_loss = weighted_categorical_crossentropy([1.0, 1.0])


    model = Model(inputs=[inputs], outputs=[outputs, centers, overlaps, widths, heights, diffx, diffy])
    model.compile(optimizer='adam',
                  loss={
                      'mask': 'binary_crossentropy',
                      'centers': 'binary_crossentropy',
                      'overlaps': 'binary_crossentropy',
#                      'centers': centers_loss,
#                      'overlaps': centers_loss,
                      'widths': mean_squared_error_masked,
                      'heights': mean_squared_error_masked,
                      'diffx': mean_squared_error_masked,
                      'diffy': mean_squared_error_masked
                  },
                  loss_weights={
                      'mask': 50,
                      'centers': 10000,
                      'overlaps': 50,
                      'widths': 150,
                      'heights': 150,
                      'diffx': 150,
                      'diffy': 150
                  })


    return model


#
# Reference: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
#
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def parametrized_weighted_loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # done to make the target label type compatible
        y_true = tf.cast(y_true, y_pred.dtype)
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return parametrized_weighted_loss

