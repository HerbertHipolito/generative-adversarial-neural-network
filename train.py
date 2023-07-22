import numpy as np
from models import generator, discriminator
from utils import generate_noisy_image, display_img, save_img, early_stopping, oneHotEncodding
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from path import paths
import argparse
from telegram_bot.sender import send_msg_telegram, send_image
from sklearn.utils import shuffle

#https://www.tensorflow.org/guide/checkpoint?hl=pt-br

def start_training(x_train,epochs,learning_rate,tries):

  model_generator, model_discriminator  = generator(), discriminator()
  optimizer_generator, optimizer_discriminator  = tf.keras.optimizers.Adam(learning_rate=learning_rate), tf.keras.optimizers.Adam(learning_rate=learning_rate)

  noisy_imgs_generated = np.reshape(generate_noisy_image(),(1,794))

  smallest_lost, count, epoch_loss_all  = float('inf'), 0, []

  for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    epoch_loss = 0
    display_img(np.reshape(noisy_imgs_generated[0][0:784],(28,28)),y_label='epoch:'+str(epoch),title='epoch'+str(epoch)+'.png',save_fig=True,show_img=False)    
    send_msg_telegram(f'start of epoch: {epoch}....')
    send_image('./imgs/'+'epoch'+str(epoch)+'.png')

    generator_row, discriminator_row = [], []

    for index_img,img in enumerate(x_train):
      
        generator_loss_total, discriminator_loss_total = 0, 0

        if not index_img%2:
          with tf.GradientTape() as tape:
            model_discriminator_result_real_img = model_discriminator(img, training=True)
            model_discriminator_result_noisy_img = model_discriminator(noisy_imgs_generated, training=True)
            discriminator_loss = tf.math.log(model_discriminator_result_real_img)
            generator_loss = tf.math.log(tf.math.subtract(1,model_discriminator_result_noisy_img))
            discriminator_loss_total = (-1)*tf.math.add(discriminator_loss, generator_loss)

   #       print(f"loss of discriminator is: {loss_batch_sum}")
          grads_discriminator = tape.gradient(discriminator_loss_total, model_discriminator.trainable_weights)
          optimizer_discriminator.apply_gradients(zip(grads_discriminator, model_discriminator.trainable_weights))
          discriminator_row.append(discriminator_loss_total.numpy()[0][0])
          epoch_loss += discriminator_loss_total.numpy()[0][0]

        else:

          with tf.GradientTape() as tape:

            noisy_imgs_generated = model_generator(noisy_imgs_generated, training=True)
            generator_loss_total = tf.math.log(tf.subtract(1,model_discriminator(noisy_imgs_generated, training=True)))

   #       print(f"loss generator: {loss}")
          grads_generator = tape.gradient(generator_loss_total, model_generator.trainable_weights)
          optimizer_generator.apply_gradients(zip(grads_generator, model_generator.trainable_weights))
          generator_row.append(generator_loss_total.numpy()[0][0])
          epoch_loss += generator_loss_total.numpy()[0][0]
    
    save_img(generator_row,title='generator'+str(epoch),path=paths['generator_loss'],label='generator')
    save_img(discriminator_row,title='discriminator'+str(epoch),path=paths['discriminator_loss'],label='discriminator')
    
    epoch_loss_all.append(epoch_loss)
    smallest_lost, count, keep_learning, feedback = early_stopping(epoch_loss,smallest_lost,count,tries)
    
    send_msg_telegram(f"loss of epoch {epoch}: {epoch_loss}. {feedback}")
    print(f"loss of epoch: {epoch_loss}. {feedback}")
    if keep_learning: break

  x_train = shuffle(x_train)
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
  
  dataset = [img for index, img in enumerate(dataset) if target[index] in selected_numbers ]
  target = oneHotEncodding(target,selected_numbers)
  dataset = np.array([ np.reshape([ *np.reshape(img/255,(1,784))[0],*target[index] ],(1,794)) for index, img in enumerate(dataset)])
  print(len(dataset),len(target))
  opt = parse_opt()
  start_training(dataset,**vars(opt))
    
