"""
对病人实现分层抽样
"""
from sklearn.model_selection import StratifiedKFold, train_test_split
import xlrd
import os

total_train_sample_list = os.listdir('D:\\manuscript_BAFNet\\manuscript\\MP\\R2\\new_data_split\\HNC\\train\\')
test_sample_list = os.listdir('D:\\manuscript_BAFNet\\manuscript\\MP\\R2\\new_data_split\\HNC\\test\\')

file = 'total_survival.xls'
wb = xlrd.open_workbook(filename=file)
sheet = wb.sheet_by_index(0)

id_cols = sheet.col_values(0)
total_train_time_list = []
total_train_status_list = []
for sample in total_train_sample_list:
    for j in range(len(id_cols)):
        if sample == id_cols[j]:
            time = int(sheet.cell(j, 8).value)
            status = int(sheet.cell(j, 9).value)
            total_train_time_list.append(time)
            total_train_status_list.append(status)

test_time_list = []
test_status_list = []
for sample in test_sample_list:
    for j in range(len(id_cols)):
        if sample == id_cols[j]:
            time = int(sheet.cell(j, 8).value)
            status = int(sheet.cell(j, 9).value)
            test_time_list.append(time)
            test_status_list.append(status)


# split the total train sample list into train and valid sample for cv
skf = StratifiedKFold(n_splits=5, random_state=2023, shuffle=True)
FOLD = 0
for train_index, valid_index in skf.split(total_train_sample_list, total_train_status_list):
    train_sample_list, train_time_list, train_status_list = [], [], []
    for index in train_index:
        train_sample_list.append(total_train_sample_list[index])
        train_time_list.append(total_train_time_list[index])
        train_status_list.append(total_train_status_list[index])

    valid_sample_list, valid_time_list, valid_status_list = [], [], []
    for index in valid_index:
        valid_sample_list.append(total_train_sample_list[index])
        valid_time_list.append(total_train_time_list[index])
        valid_status_list.append(total_train_status_list[index])

    # write txt dataloader file
    train_txt = 'new_split_txt/HNC_train_' + str(FOLD) + '.txt'
    valid_txt = 'new_split_txt/HNC_valid_' + str(FOLD) + '.txt'

    with open(train_txt, 'w') as text:
        for sample in train_sample_list:
            sample_index = train_sample_list.index(sample)
            sample_time = train_time_list[sample_index]
            sample_status = train_status_list[sample_index]

            pet_file = '/home/mijia/mijia/wuhuiqin/HNFuser_cox/Datas/train/' + sample + '/PET.nii.gz'
            ct_file = pet_file.replace('PET', 'CT')
            mask_file = pet_file.replace('PET', 'mask')

            pet_file_flip = '/home/mijia/mijia/wuhuiqin/HNFuser_cox/Datas/train/' + sample + '/PET_flip.nii.gz'
            ct_file_flip = pet_file_flip.replace('PET', 'CT')
            mask_file_flip = pet_file_flip.replace('PET', 'mask')

            pet_file_rotate = '/home/mijia/mijia/wuhuiqin/HNFuser_cox/Datas/train/' + sample + '/PET_rotate.nii.gz'
            ct_file_rotate = pet_file_rotate.replace('PET', 'CT')
            mask_file_rotate = pet_file_rotate.replace('PET', 'mask')

            pet_file_shear = '/home/mijia/mijia/wuhuiqin/HNFuser_cox/Datas/train/' + sample + '/PET_shear.nii.gz'
            ct_file_shear = pet_file_shear.replace('PET', 'CT')
            mask_file_shear = pet_file_shear.replace('PET', 'mask')

            pet_file_translate = '/home/mijia/mijia/wuhuiqin/HNFuser_cox/Datas/train/' + sample + '/PET_translate.nii.gz'
            ct_file_translate = pet_file_translate.replace('PET', 'CT')
            mask_file_translate = pet_file_translate.replace('PET', 'mask')

            text.write(pet_file + ' ' + ct_file + ' ' + mask_file + ' ' + str(sample_time) + ' ' + str(sample_status) + '\n')
            text.write(pet_file_flip + ' ' + ct_file_flip + ' ' + mask_file_flip + ' ' + str(sample_time) + ' ' + str(sample_status) + '\n')
            text.write(pet_file_rotate + ' ' + ct_file_rotate + ' ' + mask_file_rotate + ' ' + str(sample_time) + ' ' + str(sample_status) + '\n')
            text.write(pet_file_shear + ' ' + ct_file_shear + ' ' + mask_file_shear + ' ' + str(sample_time) + ' ' + str(sample_status) + '\n')
            text.write(pet_file_translate + ' ' + ct_file_translate + ' ' + mask_file_translate + ' ' + str(sample_time) + ' ' + str(sample_status) + '\n')

    with open(valid_txt, 'w') as text:
        for sample in valid_sample_list:
            sample_index = valid_sample_list.index(sample)
            sample_time = valid_time_list[sample_index]
            sample_status = valid_status_list[sample_index]

            pet_file = '/home/mijia/mijia/wuhuiqin/HNFuser_cox/Datas/train/' + sample + '/PET.nii.gz'
            ct_file = pet_file.replace('PET', 'CT')
            mask_file = pet_file.replace('PET', 'mask')

            text.write(
                pet_file + ' ' + ct_file + ' ' + mask_file + ' ' + str(sample_time) + ' ' + str(sample_status) + '\n')

    FOLD += 1


# write test sample path for dataloader
test_txt = 'new_split_txt/HNC_test.txt'
with open(test_txt, 'w') as text:
    for sample in test_sample_list:
        sample_index = test_sample_list.index(sample)
        sample_time = test_time_list[sample_index]
        sample_status = test_status_list[sample_index]

        pet_file = '/home/mijia/mijia/wuhuiqin/HNFuser_cox/Datas/test/' + sample + '/PET.nii.gz'
        ct_file = pet_file.replace('PET', 'CT')
        mask_file = pet_file.replace('PET', 'mask')

        text.write(
            pet_file + ' ' + ct_file + ' ' + mask_file + ' ' + str(sample_time) + ' ' + str(sample_status) + '\n')





