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


def read_mask(path, img_size=(256, 256), thresh_value=127):
    path = path.decode()
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, img_size, interpolation=cv2.INTER_AREA)
    thresh, x = cv2.threshold(x, thresh_value, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x


def tf_dataset_semi_sup(annotations_dict, batch_size=8, img_size=256, training_mode=False, analyze_dataset=False, include_labels=True):
    img_size = img_size
    def tf_parse(x, y):
        def _parse(x, y):
            x = read_img(x, (img_size, img_size))
            y = read_mask(y, (img_size, img_size))
            x, y = augment_img_and_mask(x, y)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
        x.set_shape([img_size, img_size, 3])
        y.set_shape([img_size, img_size, 1])
        return x, y
    
    def tf_parse_v2(x):
        def _parse(x):
            x = read_img(x, (img_size, img_size))
            # to do augment also data with image
            return x

        x = tf.numpy_function(_parse, [x], [tf.float64])
        #x.set_shape([img_size, img_size, 3])
        return x
    
    def configure_for_performance(dataset, batch_size):
        #dataset = dataset.repeat(1)
        dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.batch(batch_size, drop_remainder=True)        
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def adjust_brightness(image, gamma=1.0):
        image = image * 255.
        image = image.astype(np.uint8)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(image, table)/255
    
    def add_salt_and_pepper_noise(image, noise_ratio=0.05):
        noisy_image = image.copy()
        h, w, c = noisy_image.shape
        noisy_pixels = int(h * w * noise_ratio)

        for _ in range(noisy_pixels):
            row, col = np.random.randint(0, h), np.random.randint(0, w)
            if np.random.rand() < 0.5:
                noisy_image[row, col] = [0, 0, 0] 
            else:
                noisy_image[row, col] = [1, 1, 1]

        return noisy_image
        
    def add_gaussian_noise(image, mean=0, std=0.5):
        image = image * 255.
        image = image.astype(np.uint8)
        noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image/255 

    def augment_img_and_mask(img, mask, thresh_value=0.35):
        rows, cols, ch = img.shape
        choice = random.randint(0, 7)
        if choice == 0:
            # roatation 90
            rot1 = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)
            img = cv2.warpAffine(img, rot1, (cols, rows))
            mask = cv2.warpAffine(mask, rot1, (cols, rows))
            mask = np.expand_dims(mask,axis=-1)

        elif choice == 1:
            # roatation 180
            rot2 = cv2.getRotationMatrix2D((cols/2,rows/2), 180, 1)
            img = cv2.warpAffine(img, rot2, (cols, rows))
            mask = cv2.warpAffine(mask, rot2, (cols, rows))
            mask = np.expand_dims(mask,axis=-1)

        elif choice == 2:
            # rotation 270
            rot3 = cv2.getRotationMatrix2D((cols/2,rows/2), 270, 1)
            img = cv2.warpAffine(img, rot3, (cols, rows))
            mask = cv2.warpAffine(mask, rot3, (cols, rows))
            mask = np.expand_dims(mask,axis=-1)

        elif choice == 3:
            # horizontal flip 
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
            mask = np.expand_dims(mask,axis=-1)

        elif choice == 4:
            # vertical flip 
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            mask = np.expand_dims(mask,axis=-1)

        elif choice == 5:
            # no augmentation 
            img = img
            mask = mask

        elif choice == 6:
            # salt and paper noise 
            img = add_salt_and_pepper_noise(img)

        elif choice == 7:
            # gausian_noise 
            img = add_gaussian_noise(img)

        if choice in range(0,5):
            if random.random()<=thresh_value:
                # change brightness
                gamma = random.choice([0.85, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5])
                img = adjust_brightness(img, gamma)

        return img, mask

    path_imgs = list()
    path_masks = list()

    if training_mode:
        random.shuffle(annotations_dict)
    
    if include_labels:
        for img_id in annotations_dict:
            path_imgs.append(img_id.get('path_img'))
            path_masks.append(img_id.get('path_mask'))
        dataset = tf.data.Dataset.from_tensor_slices((path_imgs, path_masks))
        if training_mode:
            dataset = dataset.repeat(5)
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