import tensorflow as tf

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, name=None):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.), name=name)(input_layer)

    bn_name = f'{name}_BN' if name != None else None
    ac_name = f'{name}_AC' if name != None else None

    if bn: conv = tf.keras.layers.BatchNormalization(name=bn_name)(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1,name=ac_name)

    return conv

def Residual_network(input_layer, filters, i, Rname=None):
    prev = input_layer
    name = f'{Rname}/conv{i*2+1}' if Rname != None else None
    conv = convolutional(input_layer, filters_shape=(1, 1, filters // 2) , name=name)
    name = f'{Rname}/conv{i*2+2}' if Rname != None else None
    conv = convolutional(conv       , filters_shape=(3, 3, filters), name=name)
    Residual_ouput = tf.keras.layers.Add()([prev,conv])
    return Residual_ouput

def Residual_block(Residual_layer, filters, blocks, name=None):

    for i in range(blocks):
        Residual_layer = Residual_network(Residual_layer, filters, i, name)

    return Residual_layer

def upsample(input_layer, name=None):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest',name=name)
