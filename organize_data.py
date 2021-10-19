import pandas as pd
import os
import shutil

def organize_data(num_train, num_val_cat):
    num_train = 200*32
    num_train_cat = np.round(num_train/2).astype(int)
    num_val_cat = 20*32
    num_val = num_val_cat *2

    data_label = pd.read_csv('train.csv')

    if (num_train/2)+(num_val/2) > sum(data_label.sign==0):
    print('Not enough of sign 0, change load')
    if (num_train/2)+(num_val/2) > sum(data_label.sign==1):
    print('Not enough of sign 1, change load')

    # Splitting data into smol folder
    if 'train_set' not in os.listdir():
        os.mkdir('train_set')
        os.mkdir('val_set')


    # Find the appropriate number of training and validation sets for each condtion (0 and 1)
    filenames_train_0 = data_label[data_label.sign==0][0:num_train_cat].filename
    filenames_train_1 = data_label[data_label.sign==1][0:num_train_cat].filename
    filenames_val_0 = data_label[data_label.sign==0][-num_val_cat:].filename
    filenames_val_1 = data_label[data_label.sign==1][-num_val_cat:].filename

    for n in filenames_train_0:
    #print('avia_train/avia-train/{n}.png')
    if (os.path.exists(f'avia_train/avia-train/{n}.png')):
        shutil.move(f'avia_train/avia-train/{n}.png','train_set/')
    elif (os.path.exists(f'train_set/{n}.png')):
        shutil.move(f'train_set/{n}.png','train_set/')
    else:
        print('does not exist')

    for n in filenames_train_1:
    #print('avia_train/avia-train/{n}.png')
    if (os.path.exists(f'avia_train/avia-train/{n}.png')):
        shutil.move(f'avia_train/avia-train/{n}.png','train_set/')
    elif (os.path.exists(f'train_set/{n}.png')):
        shutil.move(f'train_set/{n}.png','train_set/')
    else:
        print('does not exist')

    for n in filenames_val_0:
    if (os.path.exists(f'avia_train/avia-train/{n}.png')):
        shutil.move(f'avia_train/avia-train/{n}.png','val_set/')
    else:
        print('does not exist')

    for n in filenames_val_1:
    if (os.path.exists(f'avia_train/avia-train/{n}.png')):
        shutil.move(f'avia_train/avia-train/{n}.png','val_set/')
    else:
        print('does not exist')

if __name__=='__main__':
    organize_data(6400, 640)
