import os
import pandas as pd
import cv2
import random
import tensorflow as tf
import numpy as np


def build_list_dict_nerves(path_dataset, patient_cases, only_images=False, nerve_layer_imgs=False, max_samples=None):
    list_cases = list()
    for pattient_case in patient_cases:        
        path_patient_case = os.path.join(path_dataset, pattient_case)
        images_path = os.path.join(path_patient_case, 'images')
        masks_path = os.path.join(path_patient_case, 'masks_nerves')
        list_all_imgs = os.listdir(images_path)
        list_all_masks = os.listdir(masks_path)
        all_files = os.listdir(path_patient_case)
        all_files = [f for f in all_files if f.endswith('csv')]
        annotations_file = all_files[-1]
        df_p_case = pd.read_csv(os.path.join(path_patient_case, annotations_file))
        if nerve_layer_imgs:
            list_imgs = df_p_case[df_p_case['nerve layer'] == 1]['image name'].tolist()
        else:
            list_imgs = df_p_case[~df_p_case['density-pre'].isna()]['image name'].tolist()
        for j, img_name in enumerate(list_imgs):
            name_image = ''.join([ img_name + '_.jpg'])
            name_mask = ''.join(['mask_' + img_name + '_.jpg'])
            if only_images:
                path_img = os.path.join(images_path, name_image)
                list_cases.append({'path_img': path_img,})   
                
            else:
                if name_image in list_all_imgs and name_mask in list_all_masks:
                    path_mask = os.path.join(masks_path, name_mask)
                    path_img = os.path.join(images_path, name_image)
                    list_cases.append({'path_img': path_img, 'path_mask':path_mask})    

    return list_cases

# TF dataset
def read_img(dir_image, img_size=(256, 256)):
    path_img = dir_image.decode()
    original_img = cv2.imread(path_img)
    img = cv2.resize(original_img, img_size, interpolation=cv2.INTER_AREA)
    img = img / 255.
    return img


def read_mask(path, img_size=(256, 256)):
    thresh = 127
    path = path.decode()
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, img_size, interpolation=cv2.INTER_AREA)
    thresh, x = cv2.threshold(x, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x


def tf_dataset_semi_sup(annotations_dict, batch_size=8, img_size=256, training_mode=False, analyze_dataset=False, include_labels=True):
    img_size = img_size
    def tf_parse(x, y):
        def _parse(x, y):
            x = read_img(x, (img_size, img_size))
            y = read_mask(y, (img_size, img_size))
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
        x.set_shape([img_size, img_size, 3])
        y.set_shape([img_size, img_size, 1])
        return x, y
    
    def tf_parse_v2(x):
        def _parse(x):
            x = read_img(x, (img_size, img_size))
            return x

        x = tf.numpy_function(_parse, [x], [tf.float64])
        #x.set_shape([img_size, img_size, 3])
        return x
    
    def configure_for_performance(dataset, batch_size):
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.batch(batch_size, drop_remainder=True)        
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


    path_imgs = list()
    path_masks = list()

    if training_mode:
        random.shuffle(annotations_dict)
    
    if include_labels:
        for img_id in annotations_dict:
            path_imgs.append(img_id.get('path_img'))
            path_masks.append(img_id.get('path_mask'))
        dataset = tf.data.Dataset.from_tensor_slices((path_imgs, path_masks))
        dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        for img_id in annotations_dict:
            path_imgs.append(img_id.get('path_img'))
        dataset = tf.data.Dataset.from_tensor_slices((path_imgs))
        dataset = dataset.map(tf_parse_v2, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    
    if analyze_dataset:
        filenames_ds = tf.data.Dataset.from_tensor_slices(path_imgs)
        dataset = tf.data.Dataset.zip(dataset, filenames_ds)

    if training_mode:
        dataset = configure_for_performance(dataset, batch_size=batch_size)
    else:
        dataset = dataset.batch(batch_size,  drop_remainder=True)

    print(f'TF dataset with {int(len(path_imgs)/batch_size)} elements and {len(path_imgs)} images')

    return dataset
