import tensorflow as tf
from tensorflow.keras import Model
from model.utils import CBR_Block, SCBR_Block, Con_Bn_Act
from tensorflow.keras.layers import concatenate


class DenseNet(Model):
    def __init__(self, filters=32, num_class=1, activation='sigmoid'):
        super(DenseNet, self).__init__()
        self.filters = filters
        self.num_class = num_class
        self.activation = activation

        self.CBR_Block1 = CBR_Block(filters=self.filters, num_cbr=2, block_name='dense_cbr_block1')
        self.CBR_Block2 = CBR_Block(filters=self.filters, num_cbr=2, block_name='dense_cbr_block2')
        self.CBR_Block3 = CBR_Block(filters=self.filters, num_cbr=2, block_name='dense_cbr_block3')
        self.CBR_Block4 = CBR_Block(filters=self.filters, num_cbr=2, block_name='dense_cbr_block4')
        self.CBR_out = Con_Bn_Act(filters=self.num_class, activation=self.activation, name='dense_out')

    def call(self, inputs, training=None, mask=None):
        block1 = self.CBR_Block1(inputs)
        merge1 = concatenate([block1, inputs], axis=3)

        block2 = self.CBR_Block2(merge1)
        merge2 = concatenate([block2, block1, inputs], axis=3)

        block3 = self.CBR_Block3(merge2)
        merge3 = concatenate([block3, block2, block1, inputs], axis=3)

        block4 = self.CBR_Block4(merge3)
        out = self.CBR_out(block4)

        return out
