## Kütüphanelerin yüklenmesi
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


## Veri Önişleme

#Eğitim setinin ön işlenmesi

train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
training_set=train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64,64),
                                              batch_size=32,
                                              class_mode='binary')

## Test setinin ön işlenmesi

test_datagen = ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('dataset/test_set',
                                         target_size=(64,64),
                                         batch_size=32,
                                         class_mode='binary')

### CNN Mimarisinin oluşturulması ###

cnn = tf.keras.models.Sequential()

# Adım 1 - Convolution

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

# Burada kullanılan Conv2D fonksiyonu sayesinde eldeki datalar 64x64 pixel boyutunda ele alınacaktır.
# Buradaki 3 parametresi ise resimler renkli olduğu için 3 adet katman oluşturulacaktır.


## Adım 2 - Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))




## Adım 3 - İkinci Convolutional Katman

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=3))


## Adım 4 - Flattening

cnn.add(tf.keras.layers.Flatten())

## Adım 5 - Full Connection

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

## Yapay sinir ağında gizli katmanlar ve neronlar oluşturulur.Aktivasyon fonksiyonu için en sık kullanılan
## yöntemlreden relu fonksiyonu kullanılmıştır.Kaç katman olacağı ise units metodu ile 128 olarak belirlenmiştir.
## Buradaki parametre resimler 64x64 boyutunda olduğu için 128 olarak belirlenmiştir.

## Adım 6 - Çıkış Katmanı

cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

## Çıkış katmanı ise sınıflama işlemi yapılacağı için sigmoid olarak belirlenmiştir.Eğer çıktı 1 olarak çıkarsa Temas YOK,0 olarak çıkarsa Temas VAR çıktısı verecektir.

## Derleme İşlemi

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Son adımda çıkış katmanından optimizer için adam fonksiyonu kullanılmıştır.Metric olarak accuracy seçilmiştir.


## Eğitim İşlemi

cnn.fit(x=training_set,validation_data=test_set,epochs=30)

## Tahmin oluşturma

import numpy as np
from tensorflow.keras.preprocessing import image

def tahmin_yap(resim_yolu):
    test_image = image.load_img(resim_yolu,target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result=cnn.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'Temas YOK'
    else:
        prediction = 'Temas VAR'
    return(prediction)