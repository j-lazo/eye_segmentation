import os
import pandas as pd
import cv2
import random
import tensorflow as tf
import numpy as np


def build_list_dict_dendritic_cells(path_dataset, patient_cases, only_images=False, extra_data=False, max_extra_samples=None):
    list_cases = list()
    for pattient_case in patient_cases:        
        path_patient_case = os.path.join(path_dataset, pattient_case)
        images_path = os.path.join(path_patient_case, 'images')
        masks_path = os.path.join(path_patient_case, 'new_masks_dendritic_cells')
        list_all_imgs = os.listdir(images_path)
        list_all_masks = os.listdir(masks_path)

        all_files = os.listdir(path_patient_case)
        all_files = [f for f in all_files if f.endswith('csv')]
        annotations_file = all_files[-1]

        df_p_case = pd.read_csv(os.path.join(path_patient_case, annotations_file))
        list_imgs = df_p_case[~df_p_case['total num cells'].isna()]['image name'].tolist()
        list_extra_imgs = list()

        if extra_data == 'nerve layer':
            list_extra_imgs = df_p_case[df_p_case['nerve layer'] == 1]['image name'].tolist()
        elif extra_data == 'nerve density':
            list_extra_imgs = df_p_case[~df_p_case['density-pre'].isna()]['image name'].tolist()
        elif extra_data == 'All':
            list_extra_imgs = df_p_case['image name'].tolist()
        
        if max_extra_samples:
            random.shuffle(list_extra_imgs)
            list_extra_imgs = list_extra_imgs[:max_extra_samples]

        list_imgs += list_extra_imgs

        for j, img_name in enumerate(list_imgs):
            name_image = ''.join([img_name, '_.jpg'])
            name_mask = ''.join(['mask_dendritic_', img_name, '_.jpg'])
            
            if only_images:
                path_img = os.path.join(images_path, name_image)
                list_cases.append({'path_img': path_img,})   
                
            else:
                if name_image in list_all_imgs and name_mask in list_all_masks:
                    path_mask = os.path.join(masks_path, name_mask)
                    path_img = os.path.join(images_path, name_image)
                    list_cases.append({'path_img': path_img, 'path_mask':path_mask})     
    
    return list_cases


def build_list_dict_nerves(path_dataset, patient_cases, only_images=False, extra_data=False, max_extra_samples=None):
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
        list_imgs = df_p_case[~df_p_case['density-pre'].isna()]['image name'].tolist()
        list_extra_imgs = list()

        if extra_data == 'nerve layer':
            list_extra_imgs = df_p_case[df_p_case['nerve layer'] == 1]['image name'].tolist()
        elif extra_data == 'dendritic cells':
            list_extra_imgs = df_p_case[~df_p_case['total num cells'].isna()]['image name'].tolist()
        elif extra_data == 'All':
            list_extra_imgs = df_p_case['image name'].tolist()
        
        if max_extra_samples:
            random.shuffle(list_extra_imgs)
            list_extra_imgs = list_extra_imgs[:max_extra_samples]

        list_imgs += list_extra_imgs

        for j, img_name in enumerate(list_imgs):
            name_image = ''.join([ img_name, '_.jpg'])
            name_mask = ''.join(['mask_', img_name, '_.jpg'])
            if only_images:
                path_img = os.path.join(images_path, name_image)
                list_cases.append({'path_img': path_img,})   
                
            else:
                if name_image in list_all_imgs and name_mask in list_all_masks:
                    path_mask = os.path.join(masks_path, name_mask)
                    path_img = os.path.join(images_path, name_image)
                    list_cases.append({'path_img': path_img, 'path_mask':path_mask})    

    return list_cases


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


def adjust_brightness(image, gamma=1.0):
    image = image * 255.
    image = image.astype(np.uint8)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)/255

def add_salt_and_pepper_noise(image, noise_ratio=0.05):
    # salt & pepepr noise
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
    # gausian_noise 
    image = image * 255.
    image = image.astype(np.uint8)
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image/255 

def rotate_90(img, mask):
    # roatation 90
    rows, cols, ch = img.shape
    rot1 = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)
    img = cv2.warpAffine(img, rot1, (cols, rows))
    mask = cv2.warpAffine(mask, rot1, (cols, rows))
    mask = np.expand_dims(mask,axis=-1)

    return img, mask

def rotate_180(img, mask):
    # roatation 180
    rows, cols, ch = img.shape
    rot2 = cv2.getRotationMatrix2D((cols/2,rows/2), 180, 1)
    img = cv2.warpAffine(img, rot2, (cols, rows))
    mask = cv2.warpAffine(mask, rot2, (cols, rows))
    mask = np.expand_dims(mask,axis=-1)
    return img, mask

def rotate_270(img, mask):
    # rotation 270
    rows, cols, ch = img.shape
    rot3 = cv2.getRotationMatrix2D((cols/2,rows/2), 270, 1)
    img = cv2.warpAffine(img, rot3, (cols, rows))
    mask = cv2.warpAffine(mask, rot3, (cols, rows))
    mask = np.expand_dims(mask,axis=-1)
    return img, mask

def random_rotate(img, mask):
    choice = random.randint(0,2)
    if choice == 0:
        img, mask = rotate_90(img, mask)

    elif choice == 1:
        img, mask = rotate_180(img, mask)

    elif choice == 2:
        img, mask = rotate_270(img, mask)

    return img, mask

def flip_vertical(img, mask):
    # vertical flip 
    img = cv2.flip(img, 1)
    mask = cv2.flip(mask, 1)
    mask = np.expand_dims(mask,axis=-1)
    return img, mask

def flip_horizontal(img, mask):
    # horizontal flip 
    img = cv2.flip(img, 0)
    mask = cv2.flip(mask, 0)
    mask = np.expand_dims(mask,axis=-1)
    return img, mask


def roate_points(list_rois, angle, image_shape, radius_x, radius_y):
    rotated_points = list()
    w, h, c = image_shape
    cx = int(w/2)
    cy = int(h/2)
    for points in list_rois:
        x = points[0]
        y = points[1]
        xf = points[2]
        yf = points[3]
        x_new, y_new = rotate_point((x, y), angle, (cx, cy))
        qx1, qy1 = rotate_point((xf, yf), angle, (cx, cy))
        rotated_points.append([int(x_new), int(y_new), int(qx1), int(qy1)])
    return rotated_points


def rotate_point(point, angle, center):
    """Rotate a point around a given center by an angle in degrees."""
    angle_rad = np.radians(angle)
    x, y = point
    cx, cy = center

    # Translate point to origin
    x -= cx
    y -= cy

    # Apply rotation
    x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)

    # Translate back
    x_new += cx
    y_new += cy

    return int(x_new), int(y_new)


def augment_img_mask_points(img, mask, points, radius_x, radius_y, augmentation_functions=['None', 'rotate_90', 'rotate_180', 'rotate_270', 'flip_vertical', 'flip_horizontal',
                        'add_salt_and_pepper_noise', 'add_gaussian_noise', 'adjust_brightness'], choice = None):
    if not choice:
        choice = random.choice(augmentation_functions)
    if choice == 'None':
        mask = np.expand_dims(mask, -1)
    elif choice == 'rotate_90':
        img, mask = rotate_90(img, mask)
        points = roate_points(points, -90, img.shape, radius_x, radius_y)

    elif choice == 'rotate_180':
        img, mask = rotate_180(img, mask)
        points = roate_points(points, 180, img.shape, radius_x, radius_y)

    elif choice == 'rotate_270':
        img, mask = rotate_270(img, mask)
        points = roate_points(points, -270, img.shape, radius_x, radius_y)

    elif choice == 'flip_vertical':
        img, mask = flip_vertical(img, mask)
        points = flip_points_vertical(points, img.shape)

    elif choice == 'flip_horizontal':
        img, mask = flip_horizontal(img, mask)
        points = flip_points_horizonta(points, img.shape)

    elif choice == 'add_salt_and_pepper_noise':
         img = add_salt_and_pepper_noise(img)

    elif choice == 'add_gaussian_noise':
        img = add_gaussian_noise(img)

    elif choice == 'adjust_brightness':
        img, mask = random_rotate(img, mask)
        gamma = random.choice([0.85, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5])
        img = adjust_brightness(img, gamma)

    return img, mask, points


def flip_points_horizonta(list_rois, image_shape):
    fliped_points = list()
    w, h, c = image_shape
    for points in list_rois:
        px0 = points[0]
        py0 = points[1]
        px1 = points[2]
        py1 = points[3]
        py0_f = h - 1 - py0
        py1_f = h - 1 - py1
        fliped_points.append([px0, py0_f, px1, py1_f])
    
    return fliped_points


def augment_img_and_mask(img, mask, augmentation_functions=['None', 'rotate_90', 'rotate_180', 'rotate_270', 'flip_vertical', 'flip_horizontal',
                        'add_salt_and_pepper_noise', 'add_gaussian_noise', 'adjust_brightness']):
        
    choice = random.choice(augmentation_functions)
    if choice == 'None':
        pass
    elif choice == 'rotate_90':
        img, mask = rotate_90(img, mask)

    elif choice == 'rotate_180':
        img, mask = rotate_180(img, mask)

    elif choice == 'rotate_270':
        img, mask = rotate_270(img, mask)

    elif choice == 'flip_vertical':
        img, mask = flip_horizontal(img, mask)

    elif choice == 'flip_horizontal':
        img, mask = flip_horizontal(img, mask)

    elif choice == 'add_salt_and_pepper_noise':
         img = add_salt_and_pepper_noise(img)

    elif choice == 'add_gaussian_noise':
        img = add_gaussian_noise(img)

    elif choice == 'adjust_brightness':
        img, mask = random_rotate(img, mask)
        gamma = random.choice([0.85, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5])
        img = adjust_brightness(img, gamma)

    return img, mask


# TF dataset
def tf_dataset(annotations_dict, batch_size=8, img_size=256, training_mode=False, analyze_dataset=False, include_labels=True,
                        augment=False, num_repeats=1, augmentation_functions=['None', 'rotate_90', 'rotate_180', 'rotate_270', 'flip_vertical', 'flip_horizontal',
                        'add_salt_and_pepper_noise', 'add_gaussian_noise', 'adjust_brightness']):
    
    AUGMENT_FUNCTONS = augmentation_functions
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
    
    def tf_parse_augment(x, y):
        def _parse(x, y):
            x = read_img(x, (img_size, img_size))
            y = read_mask(y, (img_size, img_size))
            x, y = augment_img_and_mask(x, y, AUGMENT_FUNCTONS)
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
    
    path_imgs = list()
    path_masks = list()

    if training_mode:
        random.shuffle(annotations_dict)
    
    if include_labels:
        for img_id in annotations_dict:
            path_imgs.append(img_id.get('path_img'))
            path_masks.append(img_id.get('path_mask'))
        dataset = tf.data.Dataset.from_tensor_slices((path_imgs, path_masks))
        if augment:
            num_repeats = 5
            dataset = dataset.repeat(num_repeats)
            dataset = dataset.map(tf_parse_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
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

    print(f'TF dataset with {int(len(path_imgs*num_repeats)/batch_size)} batches and {len(path_imgs)} images')
    dataset = dataset.prefetch(1)

    return dataset