#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import argparse
import tensorflow as tf
tf.config.optimizer.set_jit(False)
import zipfile

def build_model(conv_filters=64, dense_units=512, dropout_rate=0.3, num_conv_layers=3):
    model = models.Sequential()
    model.add(layers.Input(shape=(250, 250, 3)))
    for i in range(num_conv_layers):
        model.add(layers.Conv2D(conv_filters * (2**i), (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def unzip_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    parser.add_argument('--conv_filters', type=int, default=64)
    parser.add_argument('--dense_units', type=int, default=512)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--num_conv_layers', type=int, default=3)
    return parser.parse_args()

args = parse_args()

# Unzip the data3.zip file
zip_path = os.path.join(args.data_dir, 'data3.zip')
unzip_data(zip_path, args.data_dir)

# Now point to the actual folders
train_dir = os.path.join(args.data_dir, 'data3/train')
val_dir = os.path.join(args.data_dir, 'data3/test')


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',  # for binary classification
    batch_size=args.batch_size,
    image_size=(250, 250),
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    batch_size=args.batch_size,
    image_size=(250, 250),
    shuffle=False
)


# In[2]:


# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Augmentation block
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomContrast(0.1),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[4]:


from tensorflow.keras import layers, models

model = build_model(
    conv_filters=args.conv_filters,
    dense_units=args.dense_units,
    dropout_rate=args.dropout_rate,
    num_conv_layers=args.num_conv_layers
)

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[5]:


callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
]

history = model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  callbacks=callbacks,
                  verbose=2)

# saving the model
model.save('/opt/ml/model')
val_accuracy = max(history.history['val_accuracy'])
print(f"val_accuracy: {val_accuracy}")

