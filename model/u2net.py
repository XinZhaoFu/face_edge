from utils import Con_Bn_Act
from tensorflow.keras import Model
from unet import Up_CBR_Block
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, concatenate, add, UpSampling2D


class U2Net(Model):
    def __init__(self, rsu_middle_filters, rsu_out_filters, num_class, end_activation='softmax'):
        super(U2Net, self).__init__()
        self.rsu_middle_filters = rsu_middle_filters
        self.rsu_out_filters = rsu_out_filters
        self.num_class = num_class
        self.end_activation = end_activation

        self.pooling = MaxPooling2D(padding='same')
        self.down_rsu1 = RSU7(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)
        self.down_rsu2 = RSU6(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)
        self.down_rsu3 = RSU5(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)
        self.down_rsu4 = RSU4(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)
        self.down_rsu5 = RSU4F(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)

        self.rsu6 = RSU4F(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)

        self.up = UpSampling2D()
        self.up_rsu5 = RSU4F(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)
        self.up_rsu4 = RSU4(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)
        self.up_rsu3 = RSU5(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)
        self.up_rsu2 = RSU6(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)
        self.up_rsu1 = RSU7(middle_filters=self.rsu_middle_filters, out_filters=self.rsu_out_filters)

        self.rsu6_con = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)
        self.up_rsu5_con = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)
        self.up_rsu4_con = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)
        self.up_rsu3_con = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)
        self.up_rsu2_con = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)
        self.up_rsu1_con = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)

        self.rsu6_con_up = UpSampling2D(size=(32, 32))
        self.up_rsu5_con_up = UpSampling2D(size=(16, 16))
        self.up_rsu4_con_up = UpSampling2D(size=(8, 8))
        self.up_rsu3_con_up = UpSampling2D(size=(4, 4))
        self.up_rsu2_con_up = UpSampling2D(size=(2, 2))

        self.out_cbr = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)

    def call(self, inputs, training=None, mask=None):
        down_rsu1 = self.down_rsu1(inputs)
        down_rsu1_pooling = self.pooling(down_rsu1)

        down_rsu2 = self.down_rsu2(down_rsu1_pooling)
        down_rsu2_pooling = self.pooling(down_rsu2)

        down_rsu3 = self.down_rsu3(down_rsu2_pooling)
        down_rsu3_pooling = self.pooling(down_rsu3)

        down_rsu4 = self.down_rsu4(down_rsu3_pooling)
        down_rsu4_pooling = self.pooling(down_rsu4)

        down_rsu5 = self.down_rsu5(down_rsu4_pooling)
        down_rsu5_pooling = self.pooling(down_rsu5)

        rsu6 = self.rsu6(down_rsu5_pooling)
        rsu6_con = self.rsu6_con(rsu6)
        side6 = self.rsu6_con_up(rsu6_con)

        up_5 = self.up(rsu6)
        up_concat5 = concatenate([up_5, down_rsu5])
        up_rsu5 = self.up_rsu5(up_concat5)
        up_rsu5_con = self.up_rsu5_con(up_rsu5)
        side5 = self.up_rsu5_con_up(up_rsu5_con)

        up_4 = self.up(up_rsu5)
        up_concat4 = concatenate([up_4, down_rsu4])
        up_rsu4 = self.up_rsu4(up_concat4)
        up_rsu4_con = self.up_rsu4_con(up_rsu4)
        side4 = self.up_rsu4_con_up(up_rsu4_con)

        up_3 = self.up(up_rsu4)
        up_concat3 = concatenate([up_3, down_rsu3])
        up_rsu3 = self.up_rsu3(up_concat3)
        up_rsu3_con = self.up_rsu3_con(up_rsu3)
        side3 = self.up_rsu3_con_up(up_rsu3_con)

        up_2 = self.up(up_rsu3)
        up_concat2 = concatenate([up_2, down_rsu2])
        up_rsu2 = self.up_rsu2(up_concat2)
        up_rsu2_con = self.up_rsu2_con(up_rsu2)
        side2 = self.up_rsu2_con_up(up_rsu2_con)

        up_1 = self.up(up_rsu2)
        up_concat1 = concatenate([up_1, down_rsu1])
        up_rsu1 = self.up_rsu1(up_concat1)
        side1 = self.up_rsu1_con(up_rsu1)

        out_concat = concatenate([side1, side2, side3, side4, side5, side6], axis=3)
        out = self.out_cbr(out_concat)

        return tf.stack([out, side1, side2, side3, side4, side5, side6])


class RSU7(Model):
    def __init__(self, middle_filters, out_filters):
        super(RSU7, self).__init__()
        self.middle_filters = middle_filters
        self.out_filters = out_filters

        self.in_cbr = Con_Bn_Act(filters=self.out_filters, name='rsu7_in_cbr')

        self.pooling = MaxPooling2D(padding='same', name='rsu7_max_pooling')
        self.down_cbr_1 = Con_Bn_Act(filters=self.middle_filters, name='rsu7_down_cbr_1')
        self.down_cbr_2 = Con_Bn_Act(filters=self.middle_filters, name='rsu7_down_cbr_2')
        self.down_cbr_3 = Con_Bn_Act(filters=self.middle_filters, name='rsu7_down_cbr_3')
        self.down_cbr_4 = Con_Bn_Act(filters=self.middle_filters, name='rsu7_down_cbr_4')
        self.down_cbr_5 = Con_Bn_Act(filters=self.middle_filters, name='rsu7_down_cbr_5')
        self.down_cbr_6 = Con_Bn_Act(filters=self.middle_filters, name='rsu7_down_cbr_6')

        self.cbr_7 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=2, name='rsu7_cbr_7')

        self.up6 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu7_up_cbr_6')
        self.up5 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu7_up_cbr_5')
        self.up4 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu7_up_cbr_4')
        self.up3 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu7_up_cbr_3')
        self.up2 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu7_up_cbr_2')
        self.up1 = Con_Bn_Act(filters=self.out_filters, name='rsu7_up_cbr_1')

    def call(self, inputs, training=None, mask=None):
        in_cbr = self.in_cbr(inputs)

        down_cbr1 = self.down_cbr_1(in_cbr)
        down_pooling1 = self.pooling(down_cbr1)

        down_cbr2 = self.down_cbr_2(down_pooling1)
        down_pooling2 = self.pooling(down_cbr2)

        down_cbr3 = self.down_cbr_3(down_pooling2)
        down_pooling3 = self.pooling(down_cbr3)

        down_cbr4 = self.down_cbr_4(down_pooling3)
        down_pooling4 = self.pooling(down_cbr4)

        down_cbr5 = self.down_cbr_5(down_pooling4)
        down_pooling5 = self.pooling(down_cbr5)

        down_cbr6 = self.down_cbr_6(down_pooling5)

        cbr7 = self.cbr_7(down_cbr6)

        concat_6 = concatenate([cbr7, down_cbr6], axis=3)
        up_cbr6 = self.up6(concat_6)

        concat_5 = concatenate([up_cbr6, down_cbr5], axis=3)
        up_cbr5 = self.up5(concat_5)

        concat_4 = concatenate([up_cbr5, down_cbr4], axis=3)
        up_cbr4 = self.up4(concat_4)

        concat_3 = concatenate([up_cbr4, down_cbr3], axis=3)
        up_cbr3 = self.up3(concat_3)

        concat_2 = concatenate([up_cbr3, down_cbr2], axis=3)
        up_cbr2 = self.up2(concat_2)

        concat_1 = concatenate([up_cbr2, down_cbr1], axis=3)
        up_cbr1 = self.up1(concat_1)

        add_out = add([up_cbr1, in_cbr])

        return add_out


class RSU6(Model):
    def __init__(self, middle_filters, out_filters):
        super(RSU6, self).__init__()
        self.middle_filters = middle_filters
        self.out_filters = out_filters

        self.in_cbr = Con_Bn_Act(filters=self.out_filters, name='rsu6_in_cbr')

        self.pooling = MaxPooling2D(padding='same', name='rsu6_max_pooling')
        self.down_cbr_1 = Con_Bn_Act(filters=self.middle_filters, name='rsu6_down_cbr_1')
        self.down_cbr_2 = Con_Bn_Act(filters=self.middle_filters, name='rsu6_down_cbr_2')
        self.down_cbr_3 = Con_Bn_Act(filters=self.middle_filters, name='rsu6_down_cbr_3')
        self.down_cbr_4 = Con_Bn_Act(filters=self.middle_filters, name='rsu6_down_cbr_4')
        self.down_cbr_5 = Con_Bn_Act(filters=self.middle_filters, name='rsu6_down_cbr_5')

        self.cbr_6 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=2, name='rsu6_cbr_6')

        self.up5 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu6_up_cbr_5')
        self.up4 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu6_up_cbr_4')
        self.up3 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu6_up_cbr_3')
        self.up2 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu6_up_cbr_2')
        self.up1 = Con_Bn_Act(filters=self.out_filters, name='rsu6_up_cbr_1')

    def call(self, inputs, training=None, mask=None):
        in_cbr = self.in_cbr(inputs)

        down_cbr1 = self.down_cbr_1(in_cbr)
        down_pooling1 = self.pooling(down_cbr1)

        down_cbr2 = self.down_cbr_2(down_pooling1)
        down_pooling2 = self.pooling(down_cbr2)

        down_cbr3 = self.down_cbr_3(down_pooling2)
        down_pooling3 = self.pooling(down_cbr3)

        down_cbr4 = self.down_cbr_4(down_pooling3)
        down_pooling4 = self.pooling(down_cbr4)

        down_cbr5 = self.down_cbr_5(down_pooling4)

        cbr6 = self.cbr_6(down_cbr5)

        concat_5 = concatenate([cbr6, down_cbr5], axis=3)
        up_cbr5 = self.up5(concat_5)

        concat_4 = concatenate([up_cbr5, down_cbr4], axis=3)
        up_cbr4 = self.up4(concat_4)

        concat_3 = concatenate([up_cbr4, down_cbr3], axis=3)
        up_cbr3 = self.up3(concat_3)

        concat_2 = concatenate([up_cbr3, down_cbr2], axis=3)
        up_cbr2 = self.up2(concat_2)

        concat_1 = concatenate([up_cbr2, down_cbr1], axis=3)
        up_cbr1 = self.up1(concat_1)

        add_out = add([up_cbr1, in_cbr])

        return add_out


class RSU5(Model):
    def __init__(self, middle_filters, out_filters):
        super(RSU5, self).__init__()
        self.middle_filters = middle_filters
        self.out_filters = out_filters

        self.in_cbr = Con_Bn_Act(filters=self.out_filters, name='rsu5_in_cbr')

        self.pooling = MaxPooling2D(padding='same', name='rsu5_max_pooling')
        self.down_cbr_1 = Con_Bn_Act(filters=self.middle_filters, name='rsu5_down_cbr_1')
        self.down_cbr_2 = Con_Bn_Act(filters=self.middle_filters, name='rsu5_down_cbr_2')
        self.down_cbr_3 = Con_Bn_Act(filters=self.middle_filters, name='rsu5_down_cbr_3')
        self.down_cbr_4 = Con_Bn_Act(filters=self.middle_filters, name='rsu5_down_cbr_4')

        self.cbr_5 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=2, name='rsu5_cbr_5')

        self.up4 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu5_up_cbr_4')
        self.up3 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu5_up_cbr_3')
        self.up2 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu5_up_cbr_2')
        self.up1 = Con_Bn_Act(filters=self.out_filters, name='rsu5_up_cbr_1')

    def call(self, inputs, training=None, mask=None):
        in_cbr = self.in_cbr(inputs)

        down_cbr1 = self.down_cbr_1(in_cbr)
        down_pooling1 = self.pooling(down_cbr1)

        down_cbr2 = self.down_cbr_2(down_pooling1)
        down_pooling2 = self.pooling(down_cbr2)

        down_cbr3 = self.down_cbr_3(down_pooling2)
        down_pooling3 = self.pooling(down_cbr3)

        down_cbr4 = self.down_cbr_4(down_pooling3)

        cbr5 = self.cbr_5(down_cbr4)

        concat_4 = concatenate([cbr5, down_cbr4], axis=3)
        up_cbr4 = self.up4(concat_4)

        concat_3 = concatenate([up_cbr4, down_cbr3], axis=3)
        up_cbr3 = self.up3(concat_3)

        concat_2 = concatenate([up_cbr3, down_cbr2], axis=3)
        up_cbr2 = self.up2(concat_2)

        concat_1 = concatenate([up_cbr2, down_cbr1], axis=3)
        up_cbr1 = self.up1(concat_1)

        add_out = add([up_cbr1, in_cbr])

        return add_out


class RSU4(Model):
    def __init__(self, middle_filters, out_filters):
        super(RSU4, self).__init__()
        self.middle_filters = middle_filters
        self.out_filters = out_filters

        self.in_cbr = Con_Bn_Act(filters=self.out_filters, name='rsu4_in_cbr')

        self.pooling = MaxPooling2D(padding='same', name='rsu4_max_pooling')
        self.down_cbr_1 = Con_Bn_Act(filters=self.middle_filters, name='rsu4_down_cbr_1')
        self.down_cbr_2 = Con_Bn_Act(filters=self.middle_filters, name='rsu4_down_cbr_2')
        self.down_cbr_3 = Con_Bn_Act(filters=self.middle_filters, name='rsu4_down_cbr_3')

        self.cbr_4 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=2, name='rsu4_cbr_4')

        self.up3 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu4_up_cbr_3')
        self.up2 = Up_CBR_Block(filters=self.middle_filters, block_name='rsu4_up_cbr_2')
        self.up1 = Con_Bn_Act(filters=self.out_filters, name='rsu4_up_cbr_1')

    def call(self, inputs, training=None, mask=None):
        in_cbr = self.in_cbr(inputs)

        down_cbr1 = self.down_cbr_1(in_cbr)
        down_pooling1 = self.pooling(down_cbr1)

        down_cbr2 = self.down_cbr_2(down_pooling1)
        down_pooling2 = self.pooling(down_cbr2)

        down_cbr3 = self.down_cbr_3(down_pooling2)

        cbr4 = self.cbr_5(down_cbr3)

        concat_3 = concatenate([cbr4, down_cbr3], axis=3)
        up_cbr3 = self.up3(concat_3)

        concat_2 = concatenate([up_cbr3, down_cbr2], axis=3)
        up_cbr2 = self.up2(concat_2)

        concat_1 = concatenate([up_cbr2, down_cbr1], axis=3)
        up_cbr1 = self.up1(concat_1)

        add_out = add([up_cbr1, in_cbr])

        return add_out


class RSU4F(Model):
    def __init__(self, middle_filters, out_filters):
        super(RSU4F, self).__init__()
        self.middle_filters = middle_filters
        self.out_filters = out_filters

        self.in_cbr = Con_Bn_Act(filters=self.out_filters, name='rsu4f_in_cbr')
        self.left_cbr1 = Con_Bn_Act(filters=self.middle_filters, name='rsu4f_left_cbr1')
        self.left_cbr2 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=2, name='rsu4f_left_cbr2')
        self.left_cbr3 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=4, name='rsu4f_left_cbr3')

        self.cbr4 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=8, name='rsu4f_cbr4')

        self.right_cbr3 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=4, name='rsu4f_right_cbr3')
        self.right_cbr2 = Con_Bn_Act(filters=self.middle_filters, dilation_rate=2, name='rsu4f_right_cbr2')
        self.right_cbr1 = Con_Bn_Act(filters=self.out_filters, name='rsu4f_right_cbr1')

    def call(self, inputs, training=None, mask=None):
        in_cbr = self.in_cbr(inputs)

        left_cbr1 = self.left_cbr1(in_cbr)
        left_cbr2 = self.left_cbr2(left_cbr1)
        left_cbr3 = self.left_cbr3(left_cbr2)

        cbr4 = self.cbr4(left_cbr3)

        concat3 = concatenate([cbr4, left_cbr3], axis=3)
        right_cbr3 = self.right_cbr3(concat3)

        concat2 = concatenate([right_cbr3, left_cbr2], axis=3)
        right_cbr2 = self.right_cbr2(concat2)

        concat1 = concatenate([right_cbr2, left_cbr1], axis=3)
        right_cbr1 = self.right_cbr1(concat1)

        add_out = add([in_cbr, right_cbr1])

        return add_out
