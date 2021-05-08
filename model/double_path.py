from tensorflow.keras import Model
from model.utils import Con_Bn_Act, Sep_Con_Bn_Act, Aspp
from tensorflow.keras.layers import MaxPooling2D, concatenate, UpSampling2D


class DPNet(Model):
    def __init__(self, filters=32, class_num=2):
        super(DPNet, self).__init__()
        self.filters = filters
        self.class_num = class_num

        """
            初始块  — — — — 细节纹理分支 - - - - -  融合块     CBR x 3 - - - - SCBR x3 - - - - - - - - - CBR x 2
                  |                          |                    |                          |
                  |                          |                    |                          |
                   - - - - 语义分支    - - - -                      - down x4 aspp up x4 CBR  -
        """
        self.unit_cbr1 = Con_Bn_Act(filters=self.filters, name='unit_cbr1', activation='relu')
        self.unit_cbr2 = Sep_Con_Bn_Act(filters=self.filters, name='unit_scbr2', activation='relu')
        self.unit_cbr3 = Sep_Con_Bn_Act(filters=self.filters, name='unit_scbr3', activation='relu')

        self.detail_scbr1 = Sep_Con_Bn_Act(filters=self.filters, name='detail_scbr1', activation='relu')
        self.detail_scbr2 = Sep_Con_Bn_Act(filters=self.filters, name='detail_scbr2', activation='relu')
        self.detail_scbr3 = Sep_Con_Bn_Act(filters=self.filters, name='detail_scbr3', activation='relu')

        # self.semantic_cbr1_down = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='semantic_cbr1')
        self.semantic_scbr1 = Sep_Con_Bn_Act(filters=self.filters, name='semantic_scbr1', activation='relu')
        # self.semantic_cbr2_down = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='semantic_cbr2')
        # self.semantic_scbr2_down = Sep_Con_Bn_Act(filters=self.filters, name='semantic_scbr2')
        # # self.semantic_cbr3_down = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='semantic_cbr3')
        # self.semantic_scbr3_down = Sep_Con_Bn_Act(filters=self.filters, name='semantic_scbr3')
        # # self.semantic_cbr4_down = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='semantic_cbr4')
        # self.semantic_scbr4_down = Sep_Con_Bn_Act(filters=self.filters, name='semantic_scbr4')
        self.semantic_aspp = Aspp(filters=self.filters, dila_rate1=6, dila_rate2=12, dila_rate3=18, dila_rate4=24)
        # self.semantic_cbr5_up = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='semantic_cbr5')
        # self.semantic_scbr5_down = Sep_Con_Bn_Act(filters=self.filters, name='semantic_scbr5')
        # # self.semantic_cbr6_up = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='semantic_cbr6')
        # self.semantic_scbr6_down = Sep_Con_Bn_Act(filters=self.filters, name='semantic_scbr6')
        # # self.semantic_cbr7_up = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='semantic_cbr7')
        # self.semantic_scbr7_down = Sep_Con_Bn_Act(filters=self.filters, name='semantic_scbr7')
        # self.semantic_cbr8_up = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='semantic_cbr8')
        self.semantic_scbr2 = Sep_Con_Bn_Act(filters=self.filters, name='semantic_scbr8', activation='relu')

        self.concat_cbr1 = Con_Bn_Act(filters=filters, kernel_size=(1, 1), name='concat_cbr1')
        self.concat_cbr2 = Con_Bn_Act(filters=self.class_num, activation='softmax', name='concat_cbr2')

        self.maxPooling = MaxPooling2D(padding='same')
        self.upSampling = UpSampling2D()

    def call(self, inputs):
        unit_cbr1 = self.unit_cbr1(inputs)
        unit_cbr2 = self.unit_cbr2(unit_cbr1)
        unit_cbr3 = self.unit_cbr3(unit_cbr2)

        detail_scbr1 = self.detail_scbr1(unit_cbr3)
        detail_scbr2 = self.detail_scbr1(detail_scbr1)
        detail_scbr3 = self.detail_scbr1(detail_scbr2)
        # detail_scbr4 = self.detail_scbr1(detail_scbr3)
        # detail_scbr5 = self.detail_scbr1(detail_scbr4)
        # detail_scbr6 = self.detail_scbr1(detail_scbr5)

        # semantic_cbr1 = self.semantic_cbr1_down(unit_cbr3)
        semantic_scbr1 = self.semantic_scbr1(unit_cbr3)
        # semantic_pool1 = self.maxPooling(semantic_scbr1)
        #
        # # semantic_cbr2 = self.semantic_cbr2_down(semantic_pool1)
        # semantic_scbr2 = self.semantic_scbr2_down(semantic_pool1)
        # semantic_pool2 = self.maxPooling(semantic_scbr2)
        #
        # # semantic_cbr3 = self.semantic_cbr3_down(semantic_pool2)
        # semantic_scbr3 = self.semantic_scbr3_down(semantic_pool2)
        # semantic_pool3 = self.maxPooling(semantic_scbr3)
        #
        # # semantic_cbr4 = self.semantic_cbr4_down(semantic_pool3)
        # semantic_scbr4 = self.semantic_scbr4_down(semantic_pool3)

        semantic_aspp = self.semantic_aspp(semantic_scbr1)

        # # semantic_cbr5 = self.semantic_cbr5_up(semantic_aspp)
        # semantic_scbr5 = self.semantic_scbr5_down(semantic_aspp)
        # semantic_up5 = self.upSampling(semantic_scbr5)
        #
        # # semantic_cbr6 = self.semantic_cbr6_up(semantic_up5)
        # semantic_scbr6 = self.semantic_scbr6_down(semantic_up5)
        # semantic_up6 = self.upSampling(semantic_scbr6)
        #
        # # semantic_cbr7 = self.semantic_cbr7_up(semantic_up6)
        # semantic_scbr7 = self.semantic_scbr7_down(semantic_up6)
        # semantic_up7 = self.upSampling(semantic_scbr7)
        #
        # # semantic_cbr8 = self.semantic_cbr8_up(semantic_up7)
        semantic_scbr8 = self.semantic_scbr2(semantic_aspp)

        merge = concatenate([detail_scbr3, semantic_scbr8], axis=3)
        concat_cbr1 = self.concat_cbr1(merge)
        out = self.concat_cbr2(concat_cbr1)

        return out
