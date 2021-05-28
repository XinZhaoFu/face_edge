from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate
from model.utils import Con_Bn_Act, CBR_Block


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

    def call(self, inputs, training=None, mask=None):
        con = self.con_blocks(inputs)
        out = self.up(con)
        return out


class DenseUNet(Model):
    def __init__(self,
                 semantic_filters=32,
                 detail_filters=32,
                 num_class=2,
                 semantic_num_cbr=1,
                 detail_num_cbr=2,
                 end_activation='softmax'):
        super(DenseUNet, self).__init__()
        self.semantic_filters = semantic_filters
        self.detail_filters = detail_filters
        self.num_class = num_class
        self.semantic_num_cbr = semantic_num_cbr
        self.detail_num_cbr = detail_num_cbr
        self.end_activation = end_activation

        self.cbr_block1 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down1')
        self.cbr_block2 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down2')
        self.cbr_block3 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down3')
        self.cbr_block4 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down4')
        self.cbr_block5 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down5')
        # self.cbr_block6 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down6')
        # self.cbr_block7 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='down7')

        # self.cbr_block_up7 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='up7')
        # self.cbr_block_up6 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='up6')
        self.cbr_block_up5 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='up5')
        self.cbr_block_up4 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='up4')
        self.cbr_block_up3 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='up3')
        self.cbr_block_up2 = Up_CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='up2')
        self.cbr_block_up1 = CBR_Block(filters=self.semantic_filters, num_cbr=self.semantic_num_cbr, block_name='up1')

        self.cbr_block_detail1 = CBR_Block(filters=self.detail_filters, num_cbr=self.detail_num_cbr, block_name='detail')
        self.cbr_block_detail2 = CBR_Block(filters=self.detail_filters, num_cbr=self.detail_num_cbr, block_name='detail')
        self.cbr_block_detail3 = CBR_Block(filters=self.detail_filters, num_cbr=self.detail_num_cbr, block_name='detail')
        self.cbr_block_detail4 = CBR_Block(filters=self.detail_filters, num_cbr=self.detail_num_cbr, block_name='detail')
        self.cbr_block_detail5 = CBR_Block(filters=self.detail_filters, num_cbr=self.detail_num_cbr, block_name='detail')

        self.cbr_block_concat = CBR_Block(filters=self.detail_filters, num_cbr=self.detail_num_cbr, block_name='concat')
        self.con_end = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)

        self.pool = MaxPooling2D(padding='same')

    def call(self, inputs, training=None, mask=None):
        con1 = self.cbr_block1(inputs)

        pool2 = self.pool(con1)
        con2 = self.cbr_block2(pool2)

        pool3 = self.pool(con2)
        con3 = self.cbr_block3(pool3)

        pool4 = self.pool(con3)
        con4 = self.cbr_block4(pool4)

        pool5 = self.pool(con4)
        con5 = self.cbr_block5(pool5)

        # pool6 = self.pool(con5)
        # con6 = self.cbr_block6(pool6)
        #
        # pool7 = self.pool(con6)
        # con7 = self.cbr_block7(pool7)
        #
        # up7 = self.cbr_block_up7(con7)
        #
        # merge6 = concatenate([up7, con6], axis=3)
        # up6 = self.cbr_block_up6(merge6)

        # merge5 = concatenate([up6, con5], axis=3)
        # up5 = self.cbr_block_up5(merge5)
        up5 = self.cbr_block_up5(con5)

        merge4 = concatenate([up5, con4], axis=3)
        up4 = self.cbr_block_up4(merge4)

        merge3 = concatenate([up4, con3], axis=3)
        up3 = self.cbr_block_up3(merge3)

        merge2 = concatenate([up3, con2], axis=3)
        up2 = self.cbr_block_up2(merge2)

        merge1 = concatenate([up2, con1], axis=3)
        up1 = self.cbr_block_up1(merge1)

        detail1 = self.cbr_block_detail1(con1)
        detail_merge1 = concatenate([detail1, con1], axis=3)

        detail2 = self.cbr_block_detail2(detail_merge1)
        detail_merge2 = concatenate([detail2, detail1], axis=3)

        detail3 = self.cbr_block_detail3(detail_merge2)
        detail_merge3 = concatenate([detail3, detail2], axis=3)

        detail4 = self.cbr_block_detail4(detail_merge3)
        detail_merge4 = concatenate([detail4, detail3], axis=3)

        detail5 = self.cbr_block_detail5(detail_merge4)
        detail_merge5 = concatenate([detail5, up1], axis=3)

        concat = self.cbr_block_concat(detail_merge5)

        out = self.con_end(concat)

        return out
