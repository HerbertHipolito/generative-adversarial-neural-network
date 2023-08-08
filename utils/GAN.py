from telegram_bot.sender import send_msg_telegram
import matplotlib.pyplot as plt
import numpy as np
from path import paths
import keras
from models import generator, discriminator
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import math
from telegram_bot.sender import send_msg_telegram, send_image
import tensorflow as tf
from tqdm import tqdm
from utils.data import generate_noisy_image
from utils.plot_print import display_img, save_img

def early_stopping(current_loss, smallest_loss, count, tries = 10, e = 0.005):
  
  feedback = ''
  
  if current_loss + e <= smallest_loss: 
    count = 0
    smallest_loss = current_loss
  else:
    count +=1
    feedback = f"No improviment found: {count}"
  
  return smallest_loss, count, True if count >= tries else False, feedback

def generate_img(selected_numbers,mean,std,img_number,show_img,show_discriminator_output):
  
    model_generator = keras.models.load_model(paths['model_generator']+'/'+'generator.keras')
    model_discriminator = keras.models.load_model(paths['model_discriminator']+'/'+'discriminator.keras')
    discriminator_plot = []
    noisy_img = np.reshape(generate_noisy_image(mean,std),(1,784))
    portion_for_each_selected_number = img_number/len(selected_numbers)
    
    for index in range(img_number):
      
        target_one_hot_encoding = [selected_numbers[int(index/portion_for_each_selected_number)] for _ in range(10)]
        noisy_img = np.reshape(np.concatenate([noisy_img[0],target_one_hot_encoding], axis=0),(1,794))
        noisy_img = model_generator(noisy_img)
        discriminator_result = model_discriminator(noisy_img)[0][0] 
        print(f"discrminator result:\n prob: {discriminator_result}\n class: {'real' if discriminator_result > 0.5 else 'fake' }")
        
        generated_img = np.reshape(noisy_img[0],(28,28))
        
        display_img(generated_img,show_img=show_img,path = paths['generated_imgs'],save_fig=True,title="generated_"+str(index)+"_img.png")
        print(f"image {index+1} generated")
        #print(f"Number to be generated {selected_numbers[random_number_index]}")
        if  int(index/portion_for_each_selected_number) != int((index+1)/portion_for_each_selected_number):
          noisy_img = np.reshape(generate_noisy_image(mean,std),(1,784))
        discriminator_plot.append(discriminator_result) 
        

    if show_discriminator_output:
      plt.plot(discriminator_plot)
      plt.show()
    
def start_training(dataset,selected_numbers,target,epochs,learning_rate_discriminator,learning_rate_generator,tries,telegram_information):

  model_generator, model_discriminator  = generator(), discriminator()
  optimizer_generator, optimizer_discriminator  = tf.keras.optimizers.Adam(learning_rate=learning_rate_generator), tf.keras.optimizers.Adam(learning_rate=learning_rate_discriminator)

  smallest_lost, count, epoch_loss_all, acc_over_epochs = float('inf'), 0, [], []

  noisy_imgs_generated = np.reshape(generate_noisy_image(),(1,784))
  
  for epoch in range(epochs):
    
    print("\nStart of epoch %d" % (epoch,))
    epoch_loss = 0
    display_img(np.reshape(noisy_imgs_generated[0][0:784],(28,28)),y_label='epoch:'+str(epoch),title='epoch'+str(epoch)+'.png',save_fig=True,show_img=False)    
    
    if telegram_information: 
      send_msg_telegram(f'start of epoch: {epoch}....')
      send_image('./imgs/'+'epoch'+str(epoch)+'.png')

    generator_row, discriminator_row, prediction_discriminator_result_in_real,prediction_discriminator_result_in_generated = [], [], [], []
    #dataset_to_train = dataset[epoch%amount_of_selected_numbers]

    for index_img in tqdm(range(len(dataset))):
      
        discriminator_loss, generator_loss_total  = 0, 0

        if not index_img%2:
          
          with tf.GradientTape() as tape:
            
            model_discriminator_result_real_img = model_discriminator(dataset[index_img])
            
            prediction_discriminator_result_in_real.append( 1 if model_discriminator_result_real_img > 0.5 else 0) 
            
            discriminator_loss = (-1)*tf.math.log(model_discriminator_result_real_img)

          #print(f"loss of discriminator is: {discriminator_loss_total}")
          grads_discriminator = tape.gradient(discriminator_loss, model_discriminator.trainable_weights)
          optimizer_discriminator.apply_gradients(zip(grads_discriminator, model_discriminator.trainable_weights))
          discriminator_row.append(discriminator_loss.numpy()[0][0])
          epoch_loss += discriminator_loss.numpy()[0][0]
 
          discriminator_loss, model_discriminator_result_real_img  = 0, 0
          
          with tf.GradientTape() as tape:
            
            model_discriminator_result_noisy_img = model_discriminator(noisy_imgs_generated, training=True)
            
            prediction_discriminator_result_in_generated.append( 1 if model_discriminator_result_noisy_img > 0.5 else 0) 
            
            discriminator_loss = (-1)*tf.math.log(tf.math.subtract(1,model_discriminator_result_noisy_img))

          #print(f"loss of discriminator is: {discriminator_loss_total}")
          grads_discriminator = tape.gradient(discriminator_loss, model_discriminator.trainable_weights)
          optimizer_discriminator.apply_gradients(zip(grads_discriminator, model_discriminator.trainable_weights))
          discriminator_row.append(discriminator_loss.numpy()[0][0])
          epoch_loss += discriminator_loss.numpy()[0][0]

        else:
          
          generator_loss_total = 0
          target_part = [target[index_img] for _ in range(10)]
          noisy_img_794 = np.reshape(np.concatenate([noisy_imgs_generated[0], target_part], axis=0), (1,794)) 
          
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
    
    if epoch%10 == 0 and epoch != 0: 
      model_generator.save(paths["model_generator"]+f"/generator{epoch}.keras")
      model_discriminator.save(paths["model_discriminator"]+f"/discriminator{epoch}.keras")
    
    epoch_loss_all.append(epoch_loss)
    smallest_lost, count, keep_learning, feedback = early_stopping(epoch_loss,smallest_lost,count,tries)
    
    acc_of_epoch_1 = accuracy_score(prediction_discriminator_result_in_real,[1 for _ in range(len(prediction_discriminator_result_in_real))]) 
    acc_of_epoch_2 = accuracy_score(prediction_discriminator_result_in_generated,[0 for _ in range(len(prediction_discriminator_result_in_generated))]) 
    
    print(f"Accucary of discriminator in real img: {acc_of_epoch_1}")
    print(f"Accucary of discriminator in generated img: {acc_of_epoch_2}")
    acc_over_epochs.append((acc_of_epoch_1,acc_of_epoch_2))
    
    if telegram_information: send_msg_telegram(f"Loss of epoch {epoch}: {epoch_loss}\n Accuracy of dis. in real img: {acc_of_epoch_1}\n Accuracy of dis. in generated img: {acc_of_epoch_2} \n {feedback}")
    print(f"loss of epoch: {epoch_loss}. {feedback}")
    
    assert not math.isnan(epoch_loss), "Nan  Found"
    
    if keep_learning: break
    dataset, target = shuffle(dataset,target)
   
  acc_over_epochs = np.array(acc_over_epochs)  
  
  model_generator.save(paths["model_generator"]+"/generator.keras")
  model_discriminator.save(paths["model_discriminator"]+"/discrminator.keras")
  send_msg_telegram("end of the traning")
  save_img(epoch_loss_all,title='loss_over_epoch',path=paths['imgs'])
  save_img(acc_over_epochs[:,0],title='accucary_over_epoch_in_real',path=paths['imgs'])
  save_img(acc_over_epochs[:,1],title='accucary_over_epoch_in_generated',path=paths['imgs'])
