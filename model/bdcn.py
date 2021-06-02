from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, MaxPooling2D, UpSampling2D
from model.utils import Con_Bn_Act


class BDCN(Model):
    def __init__(self,
                 filters_id1_block=64,
                 filters_id2_block=128,
                 filters_id3_block=256,
                 filters_sem=32,
                 num_class=1,
                 end_activation='sigmoid'):
        super(BDCN, self).__init__()
        self.filters_id1_block = filters_id1_block
        self.filters_id2_block = filters_id2_block
        self.filters_id3_block = filters_id3_block
        self.filters_sem = filters_sem
        self.num_class = num_class
        self.end_activation = end_activation

        self.id1_block = IDBlock12(filters_block=self.filters_id1_block, filters_sem=self.filters_sem)
        self.id2_block = IDBlock12(filters_block=self.filters_id2_block, filters_sem=self.filters_sem)
        self.id3_block = IDBlock3(filters_block=self.filters_id3_block, filters_sem=self.filters_sem)

        self.pooling = MaxPooling2D(padding='same')
        self.up_id2 = UpSampling2D(size=(2, 2))
        self.up_id3 = UpSampling2D(size=(4, 4))

        self.out = Con_Bn_Act(filters=self.num_class, activation=self.end_activation, name='bdcn_out')

    def call(self, inputs, training=None, mask=None):
        id1_block = self.id1_block(inputs)

        id1_block_pooling = self.pooling(id1_block)
        id2_block = self.id2_block(id1_block_pooling)

        id2_block_pooing = self.pooling(id2_block)
        id3_block = self.id3_block(id2_block_pooing)

        id2_block_up = self.up_id2(id2_block)
        id3_block_up = self.up_id3(id3_block)

        concat = concatenate([id1_block, id2_block_up, id3_block_up], axis=3)
        out = self.out(concat)
        return out


class SEM(Model):
    def __init__(self, filters_sem):
        super(SEM, self).__init__()
        self.filters_sem = filters_sem

        self.cbr1 = Con_Bn_Act(filters=filters_sem, name='sem_cbr1')
        self.dilate_cbr1 = Con_Bn_Act(filters=filters_sem, dilation_rate=4, name='sem_dilate_rate4')
        self.dilate_cbr2 = Con_Bn_Act(filters=filters_sem, dilation_rate=8, name='sem_dilate_rate8')
        self.dilate_cbr3 = Con_Bn_Act(filters=filters_sem, dilation_rate=12, name='sem_dilate_rate12')
        self.cbr_end = Con_Bn_Act(filters=filters_sem, kernel_size=(1, 1), name='sem_cbr_end')

    def call(self, inputs, training=None, mask=None):
        cbr1 = self.cbr1(inputs)

        dilate_cbr1 = self.dilate_cbr1(cbr1)
        dilate_cbr2 = self.dilate_cbr2(cbr1)
        dilate_cbr3 = self.dilate_cbr3(cbr1)

        concat = concatenate([dilate_cbr1, dilate_cbr2, dilate_cbr3, cbr1], axis=3)
        out = self.cbr_end(concat)

        return out


class IDBlock12(Model):
    def __init__(self, filters_block, filters_sem=32):
        super(IDBlock12, self).__init__()
        self.filters_block = filters_block
        self.filters_sem = filters_sem

        self.cbr1 = Con_Bn_Act(filters=filters_block, name='id_block12_cbr1')
        self.cbr2 = Con_Bn_Act(filters=filters_block, name='id_block12_cbr2')

        self.sem1 = SEM(filters_sem=self.filters_sem)
        self.sem2 = SEM(filters_sem=self.filters_sem)

        self.out = Con_Bn_Act(filters=1, kernel_size=(1, 1), name='id_block12_out')

    def call(self, inputs, training=None, mask=None):
        cbr1 = self.cbr1(inputs)
        cbr2 = self.cbr2(cbr1)
        sem1 = self.sem1(cbr1)
        sem2 = self.sem2(cbr2)

        concat = concatenate([sem1, sem2], axis=3)
        out = self.out(concat)

        return out


class IDBlock3(Model):
    def __init__(self, filters_block=256, filters_sem=32):
        super(IDBlock3, self).__init__()
        self.filters_block = filters_block
        self.filters_sem = filters_sem

        self.cbr1 = Con_Bn_Act(filters=filters_block, name='id_block3_cbr1')
        self.cbr2 = Con_Bn_Act(filters=filters_block, name='id_block3_cbr2')
        self.cbr3 = Con_Bn_Act(filters=filters_block, name='id_block3_cbr3')

        self.sem1 = SEM(filters_sem=self.filters_sem)
        self.sem2 = SEM(filters_sem=self.filters_sem)
        self.sem3 = SEM(filters_sem=filters_sem)

        self.out = Con_Bn_Act(filters=1, kernel_size=(1, 1), name='id_block3_out')

    def call(self, inputs, training=None, mask=None):
        cbr1 = self.cbr1(inputs)
        cbr2 = self.cbr2(cbr1)
        cbr3 = self.cbr3(cbr2)
        sem1 = self.sem1(cbr1)
        sem2 = self.sem2(cbr2)
        sem3 = self.sem3(cbr3)

        concat = concatenate([sem1, sem2, sem3], axis=3)
        out = self.out(concat)
        return out
