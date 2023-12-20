import json
from .data_utils.data_loader import *
import glob
import six
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
#
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from .models.config import IMAGE_ORDERING
import pandas as pd

# VGG_UNet Batch = 2, Epoch = 100, Acc. = 98.20, Error = 0.0415
# VGG_UNet Batch = 2, Epoch = 1000, Acc = 99.88, Error = 0.0038, fr_w_IoU = 0.84
# VGG_ UNet Batch = 8, Epoch = 100, Acc = 99.91, Error = 0.0027, fr_w_IoU = 0.84
# VGG_Segnet Batch = 8, Epoch = 100, Acc = 99.90, Error = 0.0028
#, Error = 0.0029, fr_w_IoU = 0.

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    if len(all_checkpoint_files) == 0:
        all_checkpoint_files = glob.glob(checkpoints_path + "*.*")
        
    all_checkpoint_files = [ ff.replace(".index" , "" ) for ff in all_checkpoint_files ] # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number

    #all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
    
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files)
    return latest_epoch_checkpoint


def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))

class CustomCallBack(Callback):
    def __init__(self, checkpoints_path, inp_images, annotations, n_classes):
        self.checkpoints_path = checkpoints_path
        self.inp_images = inp_images
        self.annotations = annotations
        self.n_classes = n_classes
        self.max_mIU = -1
        self.arr = []
 
    def on_epoch_end(self, epoch, logs=None):
        tp = np.zeros(self.n_classes)
        fp = np.zeros(self.n_classes)
        fn = np.zeros(self.n_classes)
        _, big, _  = self.model.output_shape
        _, input_width,input_height, _ = self.model.input_shape
        for inp, ann in tqdm(zip(self.inp_images, self.annotations)):
            if isinstance(inp, six.string_types):
                inp = cv2.imread(inp)
            x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
            pr = self.model.predict(np.array([x]))[0]
            pr = pr.reshape((int(np.sqrt(big)),  int(np.sqrt(big)), self.n_classes)).argmax(axis=2)
            pr = pr.flatten()
            
            gt = get_segmentation_array(ann, self.n_classes, int(np.sqrt(big)), int(np.sqrt(big)), no_reshape=True)
            gt = gt.argmax(-1)
            gt = gt.flatten()
            
            for cl_i in range(self.n_classes):
                tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
                fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
                fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
                
        cl_wise_score = tp / (tp + fp + fn + 0.000000000001)   
        mIU = np.mean(cl_wise_score)   
        #print(mean_IU)  
        # only save model if there is an improvement
        self.arr.append(mIU)
        if mIU > self.max_mIU:
            self.max_mIU = mIU
            if self.checkpoints_path is not None:
                self.model.save("{}checkpoint-{}-{:.04f}-.hdf5".format(self.checkpoints_path,str(epoch), mIU))
                print("saved ", self.checkpoints_path + "." + str(epoch))

    def on_train_end(self, logs=None):
        print(self.arr)
        plt.plot(self.arr)
        plt.show()

def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=100,
          batch_size=2,
          validate=True,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=5,
          val_steps_per_epoch=5,
          gen_use_multiprocessing=True,
          ignore_zero_class=False,
          optimizer_name='adam',
          do_augment=False,
          augmentation_name="aug_all"):

    from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width
    validate=True
    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:

        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = 'categorical_crossentropy'

        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        print(validate)
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)
        paths = get_pairs_from_paths(val_images, val_annotations)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1]) 

    callbacks = [
        CheckpointsCallback(checkpoints_path)
    ]
    # edit
    #checkpoint = ModelCheckpoint(filepath=checkpoints_path+"/checkpoint-{epoch:02d}-{loss:.04f}.hdf5", monitor='accuracy', save_best_only=False, mode='auto')
    if not validate:
        model.fit_generator(train_gen, steps_per_epoch,
                            epochs=epochs, callbacks=[callbacks])
    else:
        test = [CustomCallBack(checkpoints_path, inp_images, annotations, n_classes)]
        history = model.fit_generator(train_gen,
                            steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=[test])
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = 'history.csv'
        with open(hist_csv_file, mode='w') as f:
             hist_df.to_csv(f)
             
def train_continue(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=100,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          model_path=None,
          steps_per_epoch=5,
          val_steps_per_epoch=5,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adam',
          do_augment=False,
          augmentation_name="aug_all"):
          

    assert model_path is not None
    print("Loading weights from ", model_path)
    model = load_model(model_path)
    validate=True
    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:

        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = 'categorical_crossentropy'

        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['accuracy']) 

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        print(validate)
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, 512, 512,
        do_augment=do_augment, augmentation_name=augmentation_name)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, 512, 512)
        paths = get_pairs_from_paths(val_images, val_annotations)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1]) 

    callbacks = [
        CheckpointsCallback(checkpoints_path)
    ]
    # edit
    checkpoint = ModelCheckpoint(filepath=checkpoints_path+"/checkpoint-{epoch:02d}-{loss:.04f}.hdf5", monitor='accuracy', save_best_only=False, mode='auto')
    if not validate:
        model.fit_generator(train_gen, steps_per_epoch,
                            epochs=epochs, callbacks=callbacks)
    else:
        test = [CustomCallBack(checkpoints_path, inp_images, annotations, n_classes)]
        model.fit_generator(train_gen,
                            steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=[test])
                            
