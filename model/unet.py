from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Sequential
from model.utils import Con_Bn_Act


class CBR_Block(Model):
    def __init__(self, filters, num_cbr=1, block_name=None):
        super(CBR_Block, self).__init__()
        self.filters = filters
        self.num_cbr = num_cbr
        self.block_name = None
        if block_name is not None and type(block_name) == str:
            self.block_name = block_name

        self.con_blocks = Sequential()
        for index in range(self.num_cbr):
            if self.block_name is not None:
                block = Con_Bn_Act(filters=self.filters, name=self.block_name + '_Con_Block_' + str(index+1))
            else:
                block = Con_Bn_Act(filters=self.filters, name='Con_Block_' + str(index + 1))
            self.con_blocks.add(block)

    def call(self, inputs):
        out = self.con_blocks(inputs)
        return out


class Up_CBR_Block(Model):
    def __init__(self, filters, num_cbr=1, block_name=None):
        super(Up_CBR_Block, self).__init__()
        self.filters = filters
        self.num_cbr = num_cbr
        self.block_name = None
        if block_name is not None and type(block_name) == str:
            self.block_name = block_name

        self.con_blocks = CBR_Block(filters=self.filters, num_cbr=self.num_cbr, block_name=self.block_name)
        self.up = UpSampling2D(name=self.block_name + '_up_sampling')

    def call(self, inputs):
        con = self.con_blocks(inputs)
        out = self.up(con)
        return out


class UNet(Model):
    def __init__(self, filters=32, num_class=2, semantic_num_cbr=1, detail_num_cbr=4):
        super(UNet, self).__init__()
        self.filters = filters
        self.num_class = num_class
        self.semantic_num_cbr = semantic_num_cbr
        self.detail_num_cbr = detail_num_cbr

        self.cbr_block1 = CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='down1')
        self.cbr_block2 = CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='down2')
        self.cbr_block3 = CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='down3')
        self.cbr_block4 = CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='down4')
        self.cbr_block5 = CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='down5')
        self.cbr_block6 = CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='down6')
        self.cbr_block7 = CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='down7')

        self.cbr_block_up7 = Up_CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='up7')
        self.cbr_block_up6 = Up_CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='up6')
        self.cbr_block_up5 = Up_CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='up5')
        self.cbr_block_up4 = Up_CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='up4')
        self.cbr_block_up3 = Up_CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='up3')
        self.cbr_block_up2 = Up_CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='up2')
        self.cbr_block_up1 = CBR_Block(filters=self.filters, num_cbr=self.semantic_num_cbr, block_name='up1')

        self.cbr_block_detail = CBR_Block(filters=self.filters, num_cbr=self.detail_num_cbr, block_name='detail')

        self.con_end = Con_Bn_Act(filters=self.num_class, activation='softmax')

        self.pool = MaxPooling2D(padding='same')

    def call(self, inputs):
        con1 = self.cbr_block1(inputs)
        detail = self.cbr_block_detail(con1)

        pool2 = self.pool(con1)
        con2 = self.cbr_block2(pool2)

        pool3 = self.pool(con2)
        con3 = self.cbr_block3(pool3)

        pool4 = self.pool(con3)
        con4 = self.cbr_block4(pool4)

        pool5 = self.pool(con4)
        con5 = self.cbr_block5(pool5)

        pool6 = self.pool(con5)
        con6 = self.cbr_block6(pool6)

        pool7 = self.pool(con6)
        con7 = self.cbr_block7(pool7)

        up7 = self.cbr_block_up7(con7)

        merge6 = concatenate([up7, con6], axis=3)
        up6 = self.cbr_block_up6(merge6)

        merge5 = concatenate([up6, con5], axis=3)
        up5 = self.cbr_block_up5(merge5)

        merge4 = concatenate([up5, con4], axis=3)
        up4 = self.cbr_block_up4(merge4)

        merge3 = concatenate([up4, con3], axis=3)
        up3 = self.cbr_block_up3(merge3)

        merge2 = concatenate([up3, con2], axis=3)
        up2 = self.cbr_block_up2(merge2)

        merge1 = concatenate([up2, con1, detail], axis=3)
        up1 = self.cbr_block_up1(merge1)

        out = self.con_end(up1)

        return out