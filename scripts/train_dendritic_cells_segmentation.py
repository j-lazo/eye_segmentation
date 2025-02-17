import tensorflow as tf
import os 
import numpy as np
import cv2
import random 
from tqdm import tqdm
import pandas as pd
from tensorflow.python.client import device_lib
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
from utils import data_loaders as dl
from models import Res_UNet as res_unet
from utils import loss_functions as lf


def custome_train(model, train_dataset, results_directory, new_results_id, epochs=2, learning_rate=0.001, 
                   val_dataset=None,patience=30):
    
    train_loss_hist = list()
    val_loss_hist = list()
    train_dsc_hist = list()
    val_dsc_hist = list()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = lf.dice_coef_loss(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss_val = train_loss(loss)
        train_dsc_val = lf.dice_coef(labels, predictions)
        train_dsc = train_dice_coef(train_dsc_val)
        return train_loss_val, train_dsc

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = lf.dice_coef_loss(labels, predictions)

        val_loss = valid_loss(v_loss)
        val_dsc_value = lf.dice_coef(labels, predictions)
        val_dsc = val_dice_coef(val_dsc_value)
        return val_loss, val_dsc

    @tf.function
    def prediction_step(images):
        predictions = model(images, training=False)
        return predictions

    # define loss and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    train_dice_coef = tf.keras.metrics.Mean(name='train_dsc')
    val_dice_coef = tf.keras.metrics.Mean(name='val_dsc')   

    patience = patience
    wait = 0
    best = 0
    metrics_names = ['train_dsc','train_loss', 'train_acc', 'val_dsc', 'val_loss', 'val_acc'] 
    # start training
    num_training_samples = [i for i,_ in enumerate(train_dataset)][-1] + 1
    checkpoint_filepath = os.path.join(results_directory, new_results_id + "_model_flly_supervised.h5")

    for epoch in range(epochs):
        print("\nepoch {}/{}".format(epoch+1,epochs))
        progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=metrics_names)
        for idX, batch_ds in enumerate(train_dataset):
            train_images = batch_ds[0]
            train_labels = batch_ds[1]
            train_loss_value, training_metric = train_step(train_images, train_labels)
            values=[('train_dsc', training_metric), ('train_loss',train_loss_value)]
            progBar.update(idX+1, values=values) 
        
        train_loss_hist.append(train_loss_value.numpy())
        train_dsc_hist.append(training_metric.numpy())
            
        if val_dataset:
            for valid_images, valid_labels in val_dataset:
                v_loss, val_metric = valid_step(valid_images, valid_labels)
                values=[('val_dsc',val_metric),('val_loss',v_loss)]
                progBar.update(idX+1, values=values)
        
            val_loss_hist.append(v_loss.numpy())
            val_dsc_hist.append(val_metric.numpy())
        
        wait += 1
        if epoch == 0:
            best = v_loss
        if v_loss < best:
            best = v_loss
            wait = 0
            model.save(checkpoint_filepath)

        if wait >= patience:
            print('Early stopping triggered: wait time > patience')
            model.save(checkpoint_filepath)
            break

    model.save(checkpoint_filepath)
    print('Model saved at: ', checkpoint_filepath)     
    return [train_loss_hist, val_loss_hist, train_dsc_hist, val_dsc_hist]



def get_training_dictionary(path_training_dataset):
    list_dict_annoations = list()
    path_image_annotations = os.path.join(path_training_dataset, 'images')
    path_mask_annotations = os.path.join(path_training_dataset, 'masks')
    list_imgs = os.listdir(path_image_annotations)
    for img_name in list_imgs:
        mask_name = 'mask_SAM' + img_name
        path_img = os.path.join(path_image_annotations, img_name)
        path_mask = os.path.join(path_mask_annotations, mask_name)
        if os.path.isfile(path_img) and os.path.isfile(path_mask):
            list_dict_annoations.append({'path_img': path_img, 
                                        'path_mask': path_mask})
    
    return list_dict_annoations




def main(_argv):
    tf.keras.backend.clear_session()
    path_dataset = FLAGS.path_dataset
    path_annotations = FLAGS.path_annotations
    project_folder = FLAGS.project_folder
    name_model = FLAGS.name_model
    augmentation_functions = FLAGS.augmentation_functions

    if augmentation_functions == ['all']:
        augmentation_functions = ['None', 'rotate_90', 'rotate_180', 'rotate_270', 'flip_vertical', 'flip_horizontal',
                        'add_salt_and_pepper_noise', 'add_gaussian_noise', 'adjust_brightness']
        
    #  Hyperparameters 
    lr = FLAGS.learning_rate
    epochs = FLAGS.max_epochs
    img_size = FLAGS.image_size
    batch_size = FLAGS.batch_size
    num_filters = FLAGS.num_filers
    
    list_names_gpus = list()
    devices = device_lib.list_local_devices()
    for device in devices:
        if device.device_type == 'GPU':
            desci = device.physical_device_desc
            list_names_gpus.append(desci.split('name: ')[-1].split(',')[0])

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

    print('List names GPUs:', list_names_gpus)
    print("Num GPUs:", len(physical_devices))
    print(physical_devices)
    version_tf = tf.__version__
    print('TF Version:', version_tf)
    list_all_patient_cases = os.listdir(path_dataset)
    list_all_patient_cases = [f for f in list_all_patient_cases if os.path.isdir(os.path.join(path_dataset, f))]
    # select patient cases that have annotations of dendritic cells
    list_patient_cases = list()
    for p_case in list_all_patient_cases:
        list_sub_dir = os.listdir(os.path.join(path_dataset, p_case))
        if 'new_masks_dendritic_cells' in list_sub_dir: 
            list_patient_cases.append(p_case)

    print('num patients:', len(list_patient_cases))
    seed = 10
    test_cases = ['70']
    list_patient_cases.remove('70')
    train_cases, val_test_cases = train_test_split(list_patient_cases, test_size=0.20, random_state=seed)
    val_cases, _ = train_test_split(val_test_cases, test_size=0.20, random_state=seed)

    print('train cases:', len(train_cases))
    print('val cases:', len(val_cases))
    print('test cases:', len(test_cases))

    list_train_cases = dl.build_list_dict_dendritic_cells(path_dataset, train_cases, only_images=False)
    random.shuffle(list_train_cases)
    list_val_cases = dl.build_list_dict_dendritic_cells(path_dataset, val_cases, only_images=False)
    list_test_cases = dl.build_list_dict_dendritic_cells(path_dataset, test_cases)

    train_ds = dl.tf_dataset(list_train_cases, batch_size=batch_size, training_mode=True, augment=True, img_size=img_size, augmentation_functions=augmentation_functions)
    val_ds = dl.tf_dataset(list_val_cases, batch_size=batch_size, training_mode=True, img_size=img_size, augmentation_functions=augmentation_functions)
    
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), lf.dice_coef]
    # Compile the model 
    model = res_unet.build_model(img_size, num_filters=num_filters)
    #model.summary()
    print('Compiling model')
    model.compile(optimizer=opt, loss=lf.dice_coef_loss, metrics=metrics)

    training_time = datetime.now()
    new_results_id = ''.join(['dendritic_cells_segmentation_',
                            name_model, 
                            '_lr_',
                            str(lr),
                            '_bs_',
                            str(batch_size),
                            '_', training_time.strftime("%d_%m_%Y_%H_%M"),
                            ])

    results_directory = ''.join([project_folder, 'results/', new_results_id, '/'])
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    # Save Training details in YAMLß

    patermets_traning = {'Model name':name_model, 
                         'Type of training': 'custome-training', 
                         'GPUs names': list_names_gpus, 
                         'training date': training_time.strftime("%d_%m_%Y_%H_%M"), 
                         'learning rate':lr, 
                         'batch size': batch_size, 
                         'image input size': img_size,
                         'num filters': num_filters, 
                         'dataset': os.path.split(path_dataset)[-1], 
                         'augmentation operations': augmentation_functions,
                         'train_cases':train_cases, 
                         'val_cases':val_cases, 
                         'test_cases':test_cases}

    path_yaml_file = os.path.join(results_directory, 'parameters_training.yaml')
    with open(path_yaml_file, 'w') as file:
        yaml.dump(patermets_traning, file, sort_keys=False)

    train_history = custome_train(model, train_ds, results_directory=results_directory, new_results_id=new_results_id, epochs=epochs, val_dataset=val_ds)

    train_loss_hist = train_history[0]
    val_loss_hist = train_history[1]
    train_dsc_hist = train_history[2]
    val_dsc_hist = train_history[3]
 
    df_hist = pd.DataFrame(list(zip(train_loss_hist, val_loss_hist, train_dsc_hist, val_dsc_hist)),
                  columns =['train loss', 'val loss', 'train DSC', 'val DSC'])
    training_history_path = os.path.join(results_directory, new_results_id + "_training_history.csv")
    df_hist.to_csv(training_history_path, index=False)
    
    try:
        path_history_plot = os.path.join(results_directory, new_results_id + "_training_history.jpg")
        print('History plot saaved at: ', path_history_plot)
        plt.figure()
        plt.subplot(211)
        plt.title('DSC History')
        plt.plot(train_dsc_hist, '-o', color='blue', label='train')
        plt.plot(val_dsc_hist, '-o', color='orange', label='val')
        plt.subplot(212)
        plt.title('Loss History')   
        plt.plot(train_loss_hist, '-o', color='blue', label='train')
        plt.plot(val_loss_hist, '-o', color='orange', label='val')
        plt.legend(loc='best')
        plt.savefig(path_history_plot)
        plt.close()

    except:
        print('Not possible to print history')

    # run the test 

    new_test_ds = dl.tf_dataset(list_test_cases, batch_size=1, training_mode=False, img_size=img_size, analyze_dataset=True)
    name_files = list()
    dsc_val_list = list()
    plot_figs = False

    predictions_dir = os.path.join(results_directory, 'predictions')
    os.mkdir(predictions_dir)

    for x, file_path in tqdm(new_test_ds, desc='Analyzing test dataset'):
        img_batch = x[0]
        label_batch = x[1]
        predictions = model.predict(img_batch, verbose=0)
        pred_mask = predictions[0]
        #resize_prediction = cv2.resize()
        #compare resized image 
        dsc_value = lf.dice_coef(label_batch, predictions)
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
    flags.DEFINE_list('num_filers', [32,64,128,256,512, 1024], 'mumber of filters per layer')
    flags.DEFINE_list('augmentation_functions', ['all'], 'agumentation functions used')

    flags.DEFINE_string('type_training', '', 'eager_train or custom_training')
    flags.DEFINE_string('results_dir', os.path.join(os.getcwd(), 'results'), 'directory to save the results')

    try:
        app.run(main)
    except SystemExit:
        pass
