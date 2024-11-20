import tensorflow as tf
import os 
import numpy as np
import cv2
import random 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from datetime import datetime
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
import keras.backend as K
from glob import glob
from sklearn.model_selection import train_test_split
from absl import app, flags
from absl.flags import FLAGS
import yaml


# model U-Net

def res_conv_block(x, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    skip = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    skip = tf.keras.layers.Activation("relu")(skip)
    skip = tf.keras.layers.BatchNormalization()(skip)

    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.Activation("relu")(x)

    return x


def build_model(input_size, num_filters = [64, 128, 256, 512]):
    #num_filters = [16, 32, 48, 64]
    
    inputs = tf.keras.Input((input_size, input_size, 3))  
    skip_x = []
    x = inputs

    ## Encoder
    for f in num_filters:
        x = res_conv_block(x, f)
        skip_x.append(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)

    ## Bridge
    x = res_conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = tf.keras.layers.Concatenate()([x, xs])
        x = res_conv_block(x, f)

    ## Output
    x = tf.keras.layers.Conv2D(1, (1, 1), padding="same")(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return Model(inputs, x)


def build_list_dict(path_dataset, patient_cases):
    list_cases = list()
    for pattient_case in patient_cases:
        path_patient_case = os.path.join(path_dataset, pattient_case)
        images_path = os.path.join(path_patient_case, 'images')
        masks_path = os.path.join(path_patient_case, 'masks')
        list_imgs = os.listdir(images_path)
        list_masks = os.listdir(masks_path)
        list_img_names = [f.split('.')[0] for f in list_imgs]
        for j, img_name in enumerate(list_img_names):
            for mask_name in list_masks:
                if img_name in mask_name:
                    path_mask = os.path.join(masks_path, mask_name)
                    path_img = os.path.join(images_path, list_imgs[j])
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


def tf_dataset(annotations_dict, batch_size=8, img_size=256, training_mode=False, analyze_dataset=False):
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
    
    for img_id in annotations_dict:
        path_imgs.append(img_id.get('path_img'))
        path_masks.append(img_id.get('path_mask'))

    
    dataset = tf.data.Dataset.from_tensor_slices((path_imgs, path_masks))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if analyze_dataset:
        filenames_ds = tf.data.Dataset.from_tensor_slices(path_imgs)
        dataset = tf.data.Dataset.zip(dataset, filenames_ds)

    if training_mode:
        dataset = configure_for_performance(dataset, batch_size=batch_size)
    else:
        dataset = dataset.batch(batch_size,  drop_remainder=True)

    print(f'TF dataset with {len(path_imgs)} elements')


    return dataset


# Loss

def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3])
    return tf.keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def main(_argv):
    tf.keras.backend.clear_session()
    path_dataset = FLAGS.path_dataset
    path_annotations = FLAGS.path_annotations
    project_folder = FLAGS.project_folder
    name_model = FLAGS.name_model
    #  Hyperparameters 
    lr = FLAGS.learning_rate
    epochs = FLAGS.max_epochs
    img_size = FLAGS.image_size
    batch_size = FLAGS.batch_size
    num_filters = FLAGS.num_filers
    
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

    print("Num GPUs:", len(physical_devices))
    print(physical_devices)
    print('TF Version:', tf.__version__)

    list_patient_cases = os.listdir(path_dataset)

    list_patient_cases = [f for f in list_patient_cases if os.path.isdir(os.path.join(path_dataset, f))]
    print(len(list_patient_cases))
    seed = 10
    train_cases, val_test_cases = train_test_split(list_patient_cases, test_size=0.30, random_state=seed)
    val_cases, test_cases = train_test_split(val_test_cases, test_size=0.40, random_state=seed)

    print('train cases:', len(train_cases))
    print('val cases:', len(val_cases))
    print('test cases:', len(test_cases))
    list_train_cases = list()

    list_train_cases = build_list_dict(path_dataset, train_cases)
    list_val_cases = build_list_dict(path_dataset, val_cases)
    list_test_cases = build_list_dict(path_dataset, test_cases)

    train_ds = tf_dataset(list_train_cases, batch_size=batch_size, training_mode=True, img_size=img_size)
    val_ds = tf_dataset(list_val_cases, batch_size=batch_size, training_mode=True, img_size=img_size)
    
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), dice_coef]
    # Compile the model 
    model = build_model(img_size, num_filters=num_filters)
    #model.summary()
    print('Compiling model')
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=metrics)

    training_time = datetime.now()
    new_results_id = ''.join([name_model,
                            '_lr_',
                            str(lr),
                            '_bs_',
                            str(batch_size),
                            '_', training_time.strftime("%d_%m_%Y_%H_%M"),
                            ])

    results_directory = ''.join([project_folder, 'results/', new_results_id, '/'])
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # Save Training details in YAML

    patermets_traning = {'Model name':name_model, 
                         'training date': training_time.strftime("%d_%m_%Y_%H_%M"), 
                         'learning rate':lr, 
                         'batch size': batch_size, 
                         'image input size': img_size,
                         'num filters': num_filters, 
                         'dataset': os.path.split(path_dataset)[-1]}

    path_yaml_file = os.path.join(results_directory, 'parameters_training.yaml')
    with open(path_yaml_file, 'w') as file:
        yaml.dump(patermets_traning, file)


    checkpoint_filepath = os.path.join(results_directory, new_results_id + "_model.keras")
    callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
        CSVLogger(results_directory + new_results_id + "_data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    model_history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=epochs,
                            callbacks=callbacks,verbose=1)
    
    acc_hist = model_history.history['acc']
    dice_coef_hist = model_history.history['dice_coef']
    loss_hist = model_history.history['loss']
    precision_hist = model_history.history['precision']
    recall_hist = model_history.history['recall']
    val_acc_hist = model_history.history['val_acc']
    val_dice_coef_hist = model_history.history['val_dice_coef']
    val_loss_hist = model_history.history['val_loss']
    val_precision_hist = model_history.history['val_precision']
    val_recall_hist = model_history.history['val_recall']
    learning_rate_hist = model_history.history['learning_rate']

    df_hist = pd.DataFrame(list(zip(acc_hist, dice_coef_hist, loss_hist, precision_hist, recall_hist, 
                                    val_acc_hist, val_dice_coef_hist, val_loss_hist, val_precision_hist, val_recall_hist, 
                                    learning_rate_hist)),
                  columns =list(model_history.history.keys()))
    training_history_path = os.path.join(results_directory, new_results_id + "_training_history.csv")
    df_hist.to_csv(training_history_path, index=False)
    
    new_test_ds = tf_dataset(list_test_cases, batch_size=1, training_mode=False, img_size=img_size, analyze_dataset=True)

    # run the test 

    name_files = list()
    dsc_val_list = list()
    plot_figs = False

    predictions_dir = os.path.join(results_directory, 'predictions')
    os.mkdir(predictions_dir)

    for x, file_path in tqdm(new_test_ds, desc='Analyzing test dataset'):
        img_batch = x[0]
        label_batch = x[1]
        predeictions = model.predict(img_batch, verbose=0)
        pred_mask = predeictions[0]
        #resize_prediction = cv2.resize()
        #compare resized image 
        dsc_value = dice_coef(label_batch, predeictions)
        dsc_val_list.append(dsc_value.numpy())
        name_file = file_path.numpy()[0].decode("utf-8")
        name_files.append(name_file)
        only_name = os.path.split(name_file)[-1]
        name_output_image = os.path.join(predictions_dir, only_name)
        cv2.imwrite(name_output_image, pred_mask*255)

        # save predictions independently 
        if plot_figs is True:
            plt.figure()
            plt.subplot(131)
            plt.title('Original Image')
            plt.imshow(img_batch[0])
            plt.subplot(132)
            plt.title('Original Mask')
            plt.imshow(label_batch.numpy()[0])
            plt.subplot(133)
            plt.title('Predicted Mask')
            plt.imshow(pred_mask)

    # save results in csv
    df_preds = pd.DataFrame(list(zip(name_files, dsc_val_list)),
                            columns =['file name', 'DSC'])
    name_predictions_file = os.path.join(results_directory, f'predictions_test_ds_{str(img_size)}x{str(img_size)}_.csv')
    df_preds.to_csv(name_predictions_file, index=False)


    print(f'Mean DSC test dataset: {np.mean(dsc_val_list)}')
    print('Experiment finished')

if __name__ == '__main__':

    flags.DEFINE_string('path_dataset', os.path.join(os.getcwd(), 'dataset'), 'directory dataset')
    flags.DEFINE_string('path_annotations', '', 'path annotations')
    flags.DEFINE_string('project_folder', os.getcwd(), 'path project folder')
    flags.DEFINE_string('results_directory', os.path.join(os.getcwd(), 'results'), 'path where to save results')
    flags.DEFINE_string('name_model', 'Res-UNet', 'name of the model')

    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_integer('max_epochs', 4, 'epochs')
    flags.DEFINE_integer('image_size', 64, 'input impage size')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_list('num_filers', [32,64,128,256,512], 'mumber of filters per layer')

    flags.DEFINE_string('type_training', '', 'eager_train or custom_training')
    flags.DEFINE_string('results_dir', os.path.join(os.getcwd(), 'results'), 'directory to save the results')

    try:
        app.run(main)
    except SystemExit:
        pass