import numpy as np
from models import generator, discriminator
from utils import generate_noisy_image, display_img, save_img, early_stopping, oneHotEncodding, divide_dataset
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from path import paths
import argparse
from telegram_bot.sender import send_msg_telegram, send_image
from sklearn.utils import shuffle

#https://www.tensorflow.org/guide/checkpoint?hl=pt-br

def start_training(x_train,target,epochs,learning_rate,tries):

  model_generator, model_discriminator  = generator(), discriminator()
  optimizer_generator, optimizer_discriminator  = tf.keras.optimizers.Adam(learning_rate=learning_rate), tf.keras.optimizers.Adam(learning_rate=learning_rate)

  smallest_lost, count, epoch_loss_all = float('inf'), 0, []

  noisy_imgs_generated = np.reshape(generate_noisy_image(),(1,784))
  #noisy_imgs_generated[0][784+target[0]] = 1
  
  for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    epoch_loss = 0
#    display_img(np.reshape(noisy_imgs_generated[0][0:784],(28,28)),y_label='epoch:'+str(epoch),title='epoch'+str(epoch)+'.png',save_fig=True,show_img=False)    
    send_msg_telegram(f'start of epoch: {epoch}....')
    send_image('./imgs/'+'epoch'+str(epoch)+'.png')

    generator_row, discriminator_row = [], []
    dataset_to_train = x_train[0 if epoch%2 else 1]

    for index_img,img in enumerate(dataset_to_train):
      
        generator_loss_total, discriminator_loss_total = 0, 0

        if not index_img%2:
          
          with tf.GradientTape() as tape:
            
            model_discriminator_result_real_img = model_discriminator(img, training=True)
            model_discriminator_result_noisy_img = model_discriminator(noisy_imgs_generated, training=True)
            discriminator_loss = tf.math.log(model_discriminator_result_real_img)
            generator_loss = tf.math.log(tf.math.subtract(1,model_discriminator_result_noisy_img))
            discriminator_loss_total = (-1)*tf.math.add(discriminator_loss, generator_loss)

          #print(f"loss of discriminator is: {discriminator_loss_total}")
          grads_discriminator = tape.gradient(discriminator_loss_total, model_discriminator.trainable_weights)
          optimizer_discriminator.apply_gradients(zip(grads_discriminator, model_discriminator.trainable_weights))
          discriminator_row.append(discriminator_loss_total.numpy()[0][0])
          epoch_loss += discriminator_loss_total.numpy()[0][0]

        else:
          target_part = [0 for _ in range(10)]
          noisy_img_794 = np.reshape(np.concatenate([noisy_imgs_generated[0], target_part], axis=0), (1,794)) 
          noisy_img_794[0][784+selected_numbers[0 if epoch%2 else 1]] = 1

          with tf.GradientTape() as tape:
            
            noisy_imgs_generated = model_generator(noisy_img_794, training=True)
            generator_loss_total = tf.math.log(tf.subtract(1,model_discriminator(noisy_imgs_generated, training=True)))

          #print(f"loss generator: {generator_loss_total}")
          grads_generator = tape.gradient(generator_loss_total, model_generator.trainable_weights)
          optimizer_generator.apply_gradients(zip(grads_generator, model_generator.trainable_weights))
          generator_row.append(generator_loss_total.numpy()[0][0])
          epoch_loss += generator_loss_total.numpy()[0][0]
                      
    save_img(generator_row,title='generator'+str(epoch),path=paths['generator_loss'],label='generator')
    save_img(discriminator_row,title='discriminator'+str(epoch),path=paths['discriminator_loss'],label='discriminator')
    
    if epoch%10 == 0: model_generator.save(paths["model"]+f"/generator{epoch}.keras")
    
    epoch_loss_all.append(epoch_loss)
    smallest_lost, count, keep_learning, feedback = early_stopping(epoch_loss,smallest_lost,count,tries)
    
    send_msg_telegram(f"loss of epoch {epoch}: {epoch_loss}. {feedback}")
    print(f"loss of epoch: {epoch_loss}. {feedback}")
    if keep_learning: break

  x_train[0 if epoch%2 else 1] = shuffle(x_train[0 if epoch%2 else 1])
  model_generator.save(paths["model"]+"/generator.keras")
  send_msg_telegram("end of the traning")
  save_img(epoch_loss_all,title='loss_over_epoch',path=paths['imgs'])

def parse_opt():

  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs',type=int,default=200)
  parser.add_argument('--tries',type=int,default=20)
  parser.add_argument('--learning_rate',type=float,default=3e-5)
  
  opt = parser.parse_args()
  print(vars(opt)) 
  send_msg_telegram("The model is going to be trained with the following parameters: "+str(vars(opt))) 
  return opt
  
if __name__ == '__main__':

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  dataset, target = [*x_train,*x_test], [*y_train,*y_test]
  
  selected_numbers = [0,5]
  send_msg_telegram(f"The numbers selected were {selected_numbers}")
  print(f"The numbers selected were {selected_numbers}")
  
  #dataset = [img for index, img in enumerate(dataset) if target[index] in selected_numbers ]
  #target = [number for number in target if number in selected_numbers ]
  #print(dataset)
  #target_one_hot_encodding = oneHotEncodding(target)
  dataset = np.array([np.reshape(img/255,(1,784)) for img in dataset])
  dataset = divide_dataset(dataset,target,selected_numbers)
  #print(len(dataset),len(target))
  opt = parse_opt()
  start_training(dataset,target,**vars(opt))
    
