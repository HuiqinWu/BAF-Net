"""
对病人实现分层抽样
"""
from sklearn.model_selection import StratifiedKFold, train_test_split
import xlrd
from utils.eachFile import eachFile


ct_path = 'D:\\manuscript_BAFNet\\manuscript\\MP\\R2\\new_data_split\\LcPTB\\CT\\'
ct, _ = eachFile(ct_path)

id_list = []
for niiData in ct:
    total_name = niiData.split('\\')[8].split('.')[0]
    name_list = total_name.split('_')
    aug_flag = 'flip' in name_list or 'rotate' in name_list or 'shear' in name_list or 'translate' in name_list
    if aug_flag:
        aug_form = name_list[-1]
        delete_len = len(aug_form) + 1  # the 1 is '_'
        total_name = total_name[:-delete_len]
    if total_name not in id_list:
        id_list.append(total_name)

# load label excel
file = 'LcPTB.xls'
wb = xlrd.open_workbook(filename=file)
sheet = wb.sheet_by_index(0)

cols = sheet.col_values(0)
label_list = []
for patient_id in id_list:
    for j in range(len(cols)):
        if patient_id == cols[j]:
            label = int(sheet.cell(j, 1).value)
            label_list.append(label)


# select the test samples
train_test_sample_list = train_test_split(id_list, label_list, test_size=35, random_state=2023, stratify=label_list)
total_train_sample_list = train_test_sample_list[0]
test_sample_list = train_test_sample_list[1]
total_train_label_list = train_test_sample_list[2]
test_label_list = train_test_sample_list[3]


# split the total train sample list into train and valid sample for cv
skf = StratifiedKFold(n_splits=5, random_state=2023, shuffle=True)
FOLD = 0
for train_index, valid_index in skf.split(total_train_sample_list, total_train_label_list):
    train_sample_list, train_label_list = [], []
    for index in train_index:
        train_sample_list.append(total_train_sample_list[index])
        train_label_list.append(total_train_label_list[index])

    valid_sample_list, valid_label_list = [], []
    for index in valid_index:
        valid_sample_list.append(total_train_sample_list[index])
        valid_label_list.append(total_train_label_list[index])

    # write txt dataloader file
    train_txt = 'new_split_txt/LcPTB_train_' + str(FOLD) + '.txt'
    valid_txt = 'new_split_txt/LcPTB_valid_' + str(FOLD) + '.txt'

    with open(train_txt, 'w') as text:
        for sample in train_sample_list:
            sample_index = train_sample_list.index(sample)
            sample_label = train_label_list[sample_index]

            pet_file = './AugDatas/LcPTB/PET/' + sample.replace('ct', 'pet') + '.nii'
            ct_file = './AugDatas/LcPTB/CT/' + sample + '.nii'
            pet_mask_file = './AugDatas/LcPTB/PET_Mask/' + sample.replace('ct', 'petMask') + '.nii'
            ct_mask_file = './AugDatas/LcPTB/CT_Mask/' + sample .replace('ct', 'ctMask') + '.nii'

            pet_file_flip = './AugDatas/LcPTB/PET/' + sample.replace('ct', 'pet') + '_flip.nii'
            ct_file_flip = './AugDatas/LcPTB/CT/' + sample + '_flip.nii'
            pet_mask_file_flip = './AugDatas/LcPTB/PET_Mask/' + sample.replace('ct', 'petMask') + '_flip.nii'
            ct_mask_file_flip = './AugDatas/LcPTB/CT_Mask/' + sample .replace('ct', 'ctMask') + '_flip.nii'

            pet_file_rotate = './AugDatas/LcPTB/PET/' + sample.replace('ct', 'pet') + '_rotate.nii'
            ct_file_rotate = './AugDatas/LcPTB/CT/' + sample + '_rotate.nii'
            pet_mask_file_rotate = './AugDatas/LcPTB/PET_Mask/' + sample.replace('ct', 'petMask') + '_rotate.nii'
            ct_mask_file_rotate = './AugDatas/LcPTB/CT_Mask/' + sample .replace('ct', 'ctMask') + '_rotate.nii'

            pet_file_shear = './AugDatas/LcPTB/PET/' + sample.replace('ct', 'pet') + '_shear.nii'
            ct_file_shear = './AugDatas/LcPTB/CT/' + sample + '_shear.nii'
            pet_mask_file_shear = './AugDatas/LcPTB/PET_Mask/' + sample.replace('ct', 'petMask') + '_shear.nii'
            ct_mask_file_shear = './AugDatas/LcPTB/CT_Mask/' + sample .replace('ct', 'ctMask') + '_shear.nii'

            pet_file_translate = './AugDatas/LcPTB/PET/' + sample.replace('ct', 'pet') + '_translate.nii'
            ct_file_translate = './AugDatas/LcPTB/CT/' + sample + '_translate.nii'
            pet_mask_file_translate = './AugDatas/LcPTB/PET_Mask/' + sample.replace('ct', 'petMask') + '_translate.nii'
            ct_mask_file_translate = './AugDatas/LcPTB/CT_Mask/' + sample .replace('ct', 'ctMask') + '_translate.nii'

            # data augmentation for training
            text.write(pet_file + ' ' + ct_file + ' ' + pet_mask_file + ' ' + ct_mask_file + ' ' + str(sample_label) + '\n')
            text.write(pet_file_flip + ' ' + ct_file_flip + ' ' + pet_mask_file_flip + ' ' + ct_mask_file_flip + ' ' + str(sample_label) + '\n')
            text.write(pet_file_rotate + ' ' + ct_file_rotate + ' ' + pet_mask_file_rotate + ' ' + ct_mask_file_rotate + ' ' + str(sample_label) + '\n')
            text.write(pet_file_shear + ' ' + ct_file_shear + ' ' + pet_mask_file_shear + ' ' + ct_mask_file_shear + ' ' + str(sample_label) + '\n')
            text.write(pet_file_translate + ' ' + ct_file_translate + ' ' + pet_mask_file_translate + ' ' + ct_mask_file_translate + ' ' + str(sample_label) + '\n')

    with open(valid_txt, 'w') as text:
        for sample in valid_sample_list:
            sample_index = valid_sample_list.index(sample)
            sample_label = valid_label_list[sample_index]

            pet_file = './AugDatas/LcPTB/PET/' + sample.replace('ct', 'pet') + '.nii'
            ct_file = './AugDatas/LcPTB/CT/' + sample + '.nii'
            pet_mask_file = './AugDatas/LcPTB/PET_Mask/' + sample.replace('ct', 'petMask') + '.nii'
            ct_mask_file = './AugDatas/LcPTB/CT_Mask/' + sample .replace('ct', 'ctMask') + '.nii'

            # no data augmentation for validation
            text.write(pet_file + ' ' + ct_file + ' ' + pet_mask_file + ' ' + ct_mask_file + ' ' + str(sample_label) + '\n')

    FOLD += 1


# write test sample path for dataloader
test_txt = 'new_split_txt/LcPTB_test.txt'
with open(test_txt, 'w') as text:
    for sample in test_sample_list:
        sample_index = test_sample_list.index(sample)
        sample_label = test_label_list[sample_index]

        pet_file = './AugDatas/LcPTB/PET/' + sample.replace('ct', 'pet') + '.nii'
        ct_file = './AugDatas/LcPTB/CT/' + sample + '.nii'
        pet_mask_file = './AugDatas/LcPTB/PET_Mask/' + sample.replace('ct', 'petMask') + '.nii'
        ct_mask_file = './AugDatas/LcPTB/CT_Mask/' + sample.replace('ct', 'ctMask') + '.nii'

        # no data augmentation for test
        text.write(pet_file + ' ' + ct_file + ' ' + pet_mask_file + ' ' + ct_mask_file + ' ' + str(sample_label) + '\n')