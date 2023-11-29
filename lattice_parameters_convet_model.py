import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, layers, models, metrics
import random
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt

#File_Path = "Test_data.h5"
weights_PATH = "/home/gridsan/tmackey/XRD_is_All_You_Need/ICSD/1. FCN/FCN_CS.ckpt"

#ICSD = h5py.File(File_Path, 'r')
#stack = ICSD['data'][:,:]

#data = stack[:,:-1813]/100

#x_data = data.reshape(-1,data.shape[1],1)
#y_data_cs = stack[:,-5]-1
#y_data_cs = tf.keras.utils.to_categorical(y_data_cs, num_classes=7)

drop_rate= 0.2
drop_rate_2= 0.4
initializer = tf.keras.initializers.GlorotUniform()

FCNcs_inputs = keras.Input(shape=(8192, 1))
c1=layers.Conv1D(filters=16, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(FCNcs_inputs)
p1=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c1)
p1=layers.Dropout(rate=drop_rate)(p1)
c2=layers.Conv1D(filters=16, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(p1)
p2=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c2)
p2=layers.Dropout(rate=drop_rate)(p2)
c3=layers.Conv1D(filters=32, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(p2)
p3=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c3)
p3=layers.Dropout(rate=drop_rate)(p3)
c4=layers.Conv1D(filters=32, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(p3)
p4=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c4)
p4=layers.Dropout(rate=drop_rate)(p4)
c5=layers.Conv1D(filters=64, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(p4)
p5=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c5)
p5=layers.Dropout(rate=drop_rate)(p5)
c6=layers.Conv1D(filters=64, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(p5)
p6=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c6)
p6=layers.Dropout(rate=drop_rate)(p6)
c7=layers.Conv1D(filters=128, kernel_size=3, strides=1,padding='same',activation=tf.keras.layers.LeakyReLU(), 
                         kernel_initializer=initializer)(p6)
p7=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c7)
p7=layers.Dropout(rate=drop_rate)(p7)
c8=layers.Conv1D(filters=128, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(p7)
p8=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c8)
p8=layers.Dropout(rate=drop_rate)(p8)
c9=layers.Conv1D(filters=256, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(p8)
p9=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c9)
p9=layers.Dropout(rate=drop_rate)(p9)
c10=layers.Conv1D(filters=256, kernel_size=3, strides=1,padding='same', activation=tf.keras.layers.LeakyReLU(), 
                        kernel_initializer=initializer)(p9)
p10=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c10)
p10=layers.Dropout(rate=drop_rate)(p10)
c11=layers.Conv1D(filters=512, kernel_size=3, strides=1,padding='same',activation=tf.keras.layers.LeakyReLU(), 
                         kernel_initializer=initializer)(p10)
p11=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c11)
p11=layers.Dropout(rate=drop_rate)(p11)
c12=layers.Conv1D(filters=512, kernel_size=3, strides=1,padding='same',activation=tf.keras.layers.LeakyReLU(), 
                         kernel_initializer=initializer)(p11)
p12=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c12)
p12=layers.Dropout(rate=drop_rate_2)(p12)
c13=layers.Conv1D(filters=64, kernel_size=3, strides=1,padding='same', kernel_initializer=initializer)(p12)
p13=layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(c13)
p13=layers.Dropout(rate=drop_rate_2)(p13)

c14=layers.Conv1D(filters=7, kernel_size=1, strides=1,padding='same', kernel_initializer=initializer)(p13)
f1= layers.Flatten()(c14)

FCNcs_outputs = layers.Dense(3)(f1)
# FCNcs_outputs = layers.Dense(20)(f1)
# FCNcs_outputs = layers.Dense(1)(FCNcs_outputs)

FCN = keras.Model(inputs=[FCNcs_inputs], outputs=[FCNcs_outputs], name="FCN")
opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
FCN.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
FCN.load_weights(weights_PATH)

FCN.summary()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder_dir = "/home/gridsan/tmackey/cdvae/data/mp_20/"
train_df = pd.read_csv(folder_dir + 'train.csv')
val_df = pd.read_csv(folder_dir + 'val.csv')
test_df = pd.read_csv(folder_dir + 'test.csv')

train_df['xrd_peak_locations'] = train_df['xrd_peak_locations'].apply(
    lambda x: [float(i) for i in x.strip('[]').split(',')]
)

#make sure we can read in the diffraction patterns
train_df['xrd_peak_intensities'] = train_df['xrd_peak_intensities'].apply(
    lambda x: [float(i) for i in x.strip('[]').split(',')]
)

from tqdm import tqdm

import numpy as np

def pseudo_voigt(x, center, amplitude, fwhm, eta):
    """
    Pseudo-Voigt function.
    x: array-like, the independent variable
    center: float, the center of the peak
    amplitude: float, the height of the peak
    fwhm: float, full-width at half-maximum
    eta: float, the fraction of the Lorentzian component (0 <= eta <= 1)
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma for Gaussian
    lorentzian = amplitude * (fwhm**2 / ((x - center)**2 + fwhm**2))
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    return eta * lorentzian + (1 - eta) * gaussian

def superimposed_pseudo_voigt(x, locations, intensities, fwhm, eta):
    """
    Superimpose multiple pseudo-Voigt functions.
    x: array-like, the independent variable
    locations: list, the centers of the peaks
    intensities: list, the heights of the peaks
    fwhm: float, full-width at half-maximum (same for all peaks)
    eta: float, the fraction of the Lorentzian component (0 <= eta <= 1, same for all peaks)
    """
    total = np.zeros_like(x)
    for center, amplitude in zip(locations, intensities):
        total += pseudo_voigt(x, center, amplitude, fwhm, eta)
    total = total / max(total)
    return total

import pandas as pd
from multiprocessing import Pool

# Assuming train_df is already defined and simulate_xrd function is defined

# Function to simulate XRD for each row
def simulate_pv_xrd_for_row(row_tuple):
    index, row = row_tuple  # Unpack the tuple
    
    x = np.linspace(5, 85, 8192) 
    fwhm = 0.005  # Full-width at half-maximum (common for all peaks)
    eta = 0.75  # Fraction of Lorentzian component (common for all peaks)

    sim_xrd = superimposed_pseudo_voigt(x, row['xrd_peak_locations'], row['xrd_peak_intensities'], fwhm, eta)

    return sim_xrd

# Function to apply simulation to each row
def apply_simulation(data):
    with Pool() as pool:
        results = list(tqdm(pool.imap(simulate_pv_xrd_for_row, data.iterrows()), total=len(data)))
    return results

# Apply the simulation
train_df['sim_pv_xrd_intensities'] = pd.Series(apply_simulation(train_df), dtype=object)

train_xrd_mp_20 = np.array(train_df['sim_pv_xrd_intensities'])

train_xrd_mp_20 = np.stack(train_xrd_mp_20)

train_xrd_mp_20 = train_xrd_mp_20.reshape(27136, 8192, 1)

import ast
train_df['lattice parameters'] = train_df['lattice parameters'].apply(ast.literal_eval)

train_df['lattice parameters'] = train_df['lattice parameters'].apply(np.array)

y_lattice_params = np.array(train_df['lattice parameters'])

y_lattice_params = np.stack(y_lattice_params)

y_lattice_params.shape

train_xrd_mp_20.shape


class MARD(tf.keras.metrics.Metric):
    def __init__(self, name='mard', **kwargs):
        super(MARD, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_true - y_pred) / tf.maximum(tf.abs(y_true), tf.keras.backend.epsilon())
        error = tf.reduce_mean(error)  # Average over the batch

        self.total.assign_add(error)
        self.count.assign_add(1)

    def result(self):
        return (self.total / self.count) * 100  # To convert to percentage

    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)

        
# Usage in model compilation
FCN.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error', MARD()])

history = FCN.fit(train_xrd_mp_20, y_lattice_params, epochs=1000, batch_size=256, validation_split=0.2)
