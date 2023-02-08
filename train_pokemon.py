import os
if os.system('nvidia-smi') == 0:
    import setGPU
import tensorflow as tf
import glob
import sys
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, confusion_matrix
import resnet_v1_eembc
import yaml
import csv
import json
import datetime
# from keras_flops import get_flops # (different flop calculation)
import kerop
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.layers.experimental.preprocessing import RandomCrop
#random_crop_model = tf.keras.models.Sequential()
#random_crop_model.add(RandomCrop(32, 32, input_shape=(32, 32, 3,)))
# /home/bhawks/.conda/envs/dune-1dcnn/lib/python3.9/site-packages/qkeras/src/qkeras/qkeras
if os.system('nvidia-smi') == 0:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    device = "/GPU:0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    device = "/CPU:0"


#def random_crop(x):
#    return random_crop_model.predict(x)


def get_lr_schedule_func(initial_lr, lr_decay):

    def lr_schedule_func(epoch):
        return initial_lr * (lr_decay ** epoch)

    return lr_schedule_func


def main(args):

    # parameters
    input_shape = [32, 32, 3]
    num_classes = 151#10
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    num_filters = config['model']['filters']
    kernel_sizes = config['model']['kernels']
    strides = config['model']['strides']
    l1p = float(config['model']['l1'])
    l2p = float(config['model']['l2'])
    skip = bool(config['model']['skip'])
    avg_pooling = bool(config['model']['avg_pooling'])
    batch_size = config['fit']['batch_size']
    num_epochs = config['fit']['epochs']
    verbose = config['fit']['verbose']
    patience = config['fit']['patience']
    save_dir = config['save_dir']
    model_name = config['model']['name']
    loss = config['fit']['compile']['loss']
    model_file_path = os.path.join(save_dir, 'model_best.h5')

    # quantization parameters
    if 'quantized' in model_name:
        logit_total_bits = config["quantization"]["logit_total_bits"]
        logit_int_bits = config["quantization"]["logit_int_bits"]
        activation_total_bits = config["quantization"]["activation_total_bits"]
        activation_int_bits = config["quantization"]["activation_int_bits"]
        alpha = config["quantization"]["alpha"]
        use_stochastic_rounding = config["quantization"]["use_stochastic_rounding"]
        logit_quantizer = config["quantization"]["logit_quantizer"]
        activation_quantizer = config["quantization"]["activation_quantizer"]
        final_activation = bool(config['model']['final_activation'])

    # optimizer
    optimizer = getattr(tf.keras.optimizers, config['fit']['compile']['optimizer'])
    initial_lr = config['fit']['compile']['initial_lr']
    lr_decay = config['fit']['compile']['lr_decay']

    # load dataset
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
   # X_train, X_test = X_train/256., X_test/256.

    #y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    #y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    #if loss == 'squared_hinge':
    #    y_train = y_train * 2 - 1  # -1 or 1 for hinge loss
    #    y_test = y_test * 2 - 1

    # define data generator
    
    datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip = True,
        validation_split=0.25
        # preprocessing_function=random_crop,
        #brightness_range=(0.9, 1.2),
        #contrast_range=(0.9, 1.2)
    )
    
    
    train_generator = datagen.flow_from_directory('./data/PokemonData/',
                                target_size=(32,32),
                                batch_size=batch_size,
                                color_mode='rgb',
                                class_mode='categorical',
                                interpolation='bicubic',
                                subset='training',
                                shuffle=True
                                #keep_aspect_ratio=True,                                             
                                )
    
    validation_generator = datagen.flow_from_directory('./data/PokemonData/',
                                target_size=(32,32),
                                batch_size=batch_size,
                                color_mode='rgb',
                                class_mode='categorical',
                                interpolation='bicubic',
                                subset='validation',
                                shuffle=True
                                #keep_aspect_ratio=True
                                )       

    # run preprocessing on training dataset
    #datagen.fit(X_train)
    #print(validation_generator.classes)
    kwargs = {'input_shape': input_shape,
              'num_classes': num_classes,
              'num_filters': num_filters,
              'kernel_sizes': kernel_sizes,
              'strides': strides,
              'l1p': l1p,
              'l2p': l2p,
              'skip': skip,
              'avg_pooling': avg_pooling}

    # pass quantization params
    if 'quantized' in model_name:
        kwargs["logit_total_bits"] = logit_total_bits
        kwargs["logit_int_bits"] = logit_int_bits
        kwargs["activation_total_bits"] = activation_total_bits
        kwargs["activation_int_bits"] = activation_int_bits
        kwargs["alpha"] = None if alpha == 'None' else alpha
        kwargs["use_stochastic_rounding"] = use_stochastic_rounding
        kwargs["logit_quantizer"] = logit_quantizer
        kwargs["activation_quantizer"] = activation_quantizer
        kwargs["final_activation"] = final_activation

    # define model
    model = getattr(resnet_v1_eembc, model_name)(**kwargs)

    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')

    # analyze FLOPs (see https://github.com/kentaroy47/keras-Opcounter)
    layer_name, layer_flops, inshape, weights = kerop.profile(model)

    # visualize FLOPs results
    total_flop = 0
    for name, flop, shape in zip(layer_name, layer_flops, inshape):
        print("layer:", name, shape, " MFLOPs:", flop/1e6)
        total_flop += flop
    print("Total FLOPs: {} MFLOPs".format(total_flop/1e6))
    '''
    tf.keras.utils.plot_model(model,
                              to_file="model.png",
                              show_shapes=True,
                              show_dtype=False,
                              show_layer_names=False,
                              rankdir="TB",
                              expand_nested=False)
    '''
    # Alternative FLOPs calculation (see https://github.com/tokusumi/keras-flops), ~same answer
    #total_flop = get_flops(model, batch_size=1)
    #print("FLOPS: {} GLOPs".format(total_flop/1e9))

    # compile model with optimizer
    model.compile(optimizer=optimizer(learning_rate=initial_lr),
                  loss=loss,
                  metrics=['accuracy'])

    # callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

    lr_schedule_func = get_lr_schedule_func(initial_lr, lr_decay)
    
    log_dir = "logs/fit_151/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=verbose, save_best_only=True),
                 EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose, restore_best_weights=True),
                 LearningRateScheduler(lr_schedule_func, verbose=verbose),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
                 ]

    # train
    with tf.device(device):
        history = model.fit_generator(train_generator,
                                      steps_per_epoch = train_generator.samples // batch_size,
                                      validation_data = validation_generator, 
                                      validation_steps = validation_generator.samples // batch_size,
                                      epochs = num_epochs,
                                      callbacks=callbacks,
                                      verbose=verbose)
    '''
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=num_epochs,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks,
                        verbose=verbose)
    '''
    
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(os.join(log_dir,"history.json"), 'w'))
    
    
    # restore "best" model
    model.load_weights(model_file_path)

    # get predictions
    y_pred = model.predict(validation_generator)

    # evaluate with test dataset and share same prediction results
    evaluation = model.evaluate(validation_generator)

    auc = roc_auc_score(validation_generator.classes, y_pred, average='weighted', multi_class='ovr')

    print('Model test accuracy = %.3f' % evaluation[1])
    print('Model test weighted average AUC = %.3f' % auc)
    
    all1 = confusion_matrix(validation_generator.classes, y_pred>0.5)
    sns.heatmap(all1, annot=True)
    plt.title("Confusion Matrix - RN07 All Gen 1 Pokemon")
    plt.xlabel("Predicted Class (>0.5)")
    plt.ylabel("True Class")
    plt.show()
    plt.savefig('RN07_151_Gen1_ConfMat.png')
    
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(validation_generator.classes, y_pred)
    plt.plot(fpr_keras, tpr_keras)
    plt.title("ROC Curve - RN07 All Gen 1 Pokemon - AUC: %.3f" % auc)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('RN07_151_Gen1_ROC.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
