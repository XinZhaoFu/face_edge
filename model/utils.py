from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D, \
    DepthwiseConv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras import Model, regularizers


class Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 padding='same',
                 strides=1,
                 activation='relu',
                 dilation_rate=1,
                 name=None,
                 kernel_regularizer=False):
        super(Con_Bn_Act, self).__init__()
        self.kernel_regularizer = kernel_regularizer
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.block_name = name

        if self.kernel_regularizer:
            self.con_regularizer = regularizers.l2()
        else:
            self.con_regularizer = None

        # kernel_initializer_special_cases = ['he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        self.con = Conv2D(filters=self.filters,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          strides=self.strides,
                          use_bias=False,
                          dilation_rate=(self.dilation_rate, self.dilation_rate),
                          name=self.block_name,
                          kernel_regularizer=self.con_regularizer,
                          kernel_initializer='glorot_uniform')
        self.bn = BatchNormalization()
        if self.activation is not None:
            self.act = Activation(self.activation)

    def call(self, inputs):
        con = self.con(inputs)
        bn = self.bn(con)
        if self.kernel_size != (1, 1) and self.activation is not None:
            out = self.act(bn)
        else:
            out = bn
        return out


class Sep_Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 padding='same',
                 strides=1,
                 activation='relu',
                 name=None):
        super(Sep_Con_Bn_Act, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.block_name = name

        self.con = SeparableConv2D(filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   strides=self.strides,
                                   use_bias=False,
                                   name=self.block_name)
        self.bn = BatchNormalization()
        self.act = Activation(self.activation)

    def call(self, inputs):
        con = self.con(inputs)
        bn = self.bn(con)
        if self.kernel_size != (1, 1):
            out = self.act(bn)
        else:
            out = bn
        return out


class DW_Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=1,
                 use_bias=False,
                 padding='same',
                 name=None,
                 activation='relu'):
        super(DW_Con_Bn_Act, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.padding = padding
        self.block_name = name
        self.activation = activation

        self.con_1x1 = Con_Bn_Act(filters=self.filters,
                                  kernel_size=(1, 1))

        self.dw_con = DepthwiseConv2D(kernel_size=self.kernel_size,
                                      strides=self.strides,
                                      use_bias=self.use_bias,
                                      padding=self.padding,
                                      name=self.block_name)
        self.bn = BatchNormalization()
        if self.activation is not None:
            self.act = Activation(self.activation)

    def call(self, inputs):
        con_1x1 = self.con_1x1(inputs)

        con = self.dw_con(con_1x1)
        bn = self.bn(con)

        if self.kernel_size != (1, 1) and self.activation is not None:
            out = self.act(bn)
        else:
            return bn
        return out


class Aspp(Model):
    def __init__(self, filters=256, dila_rate1=6, dila_rate2=12, dila_rate3=18, dila_rate4=24):
        super(Aspp, self).__init__()
        self.filters = filters

        self.con1x1 = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), activation=None, name='aspp_con1x1')

        self.dila_con1 = Con_Bn_Act(filters=self.filters, dilation_rate=dila_rate1, activation=None, padding='same',
                                name='aspp_dila_con1')
        self.dila_con2 = Con_Bn_Act(filters=self.filters, dilation_rate=dila_rate2, activation=None, padding='same',
                                name='aspp_dila_con2')
        self.dila_con3 = Con_Bn_Act(filters=self.filters, dilation_rate=dila_rate3, activation=None, padding='same',
                                name='aspp_dila_con3')
        self.dila_con4 = Con_Bn_Act(filters=self.filters, dilation_rate=dila_rate4, activation=None, padding='same',
                                    name='aspp_dila_con3')

        self.pooling_1 = MaxPooling2D(name='aspp_pooling_pooling')
        self.pooling_2 = Conv2D(filters=self.filters, kernel_size=(1, 1), padding='same',
                                name='aspp_pooling_con1x1')
        self.pooling_3 = UpSampling2D(name='aspp_pooling_upsampling')

        self.concat_2 = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), padding='same',
                                   name='aspp_concate_con1x1')

    def call(self, inputs):
        con1x1 = self.con1x1(inputs)

        dila_con6x6 = self.dila_con1(inputs)
        dila_con12x12 = self.dila_con2(inputs)
        dila_con18x18 = self.dila_con3(inputs)

        pooling_1 = self.pooling_1(inputs)
        pooling_2 = self.pooling_2(pooling_1)
        pooling_3 = self.pooling_3(pooling_2)

        concat_1 = concatenate([con1x1, dila_con6x6, dila_con12x12, dila_con18x18, pooling_3], axis=3)
        out = self.concat_2(concat_1)

        return out