from data_utils.utils import get_img_mask_list


class Data_Loader_File:
    def __init__(self,
                 batch_size,
                 data_augmentation=False,
                 train_file_path='./data/train/',
                 val_file_path='./data/val/'):
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path

    def load_train_data(self, load_file_number):
        print('正在载入训练集')
        train_dataset = get_img_mask_list(file_number=load_file_number,
                                          file_path=self.train_file_path,
                                          batch_size=self.batch_size,
                                          data_augmentation=self.data_augmentation)
        return train_dataset

    def load_val_data(self, load_file_number):
        print('正在载入验证集')
        val_dataset = get_img_mask_list(file_number=load_file_number,
                                        batch_size=self.batch_size,
                                        file_path=self.val_file_path)
        return val_dataset
