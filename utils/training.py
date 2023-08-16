import numpy as np
from path import paths
from utils.models import generator, discriminator
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

def update_discriminator_weights(model,img,target,is_real_img,optimizer):
  
  target_array = [target for _ in range(10)]
  input_model = tf.reshape(tf.concat([img[0],target_array],axis=0),(1,794))
  
  with tf.GradientTape() as tape:
    
    model_result = model(input_model)
    model_loss = (-1)*tf.math.log(model_result) if is_real_img else (-1)*tf.math.log(tf.math.subtract(1,model_result))
  
  grads_discriminator = tape.gradient(model_loss, model.trainable_weights)
  optimizer.apply_gradients(zip(grads_discriminator, model.trainable_weights))
  
  return model, model_loss.numpy()[0][0], model_result
    
def update_generator_weights(model_generator,model_discriminator,img,optimizer,target):
 
  target_array = [target for _ in range(10)] 
  noisy_img_794 = tf.reshape(tf.concat([img[0],target_array], axis=0),(1,794))

  with tf.GradientTape() as tape:
    
    generator_model_result = model_generator(noisy_img_794)
    noisy_img_794 = tf.reshape(tf.concat([generator_model_result[0],target_array], axis=0),(1,794))
    discriminator_model_result = model_discriminator(noisy_img_794)
    model_loss = tf.math.log(tf.math.subtract(1,discriminator_model_result))
  
  grads_discriminator = tape.gradient(model_loss, model_generator.trainable_weights)
  optimizer.apply_gradients(zip(grads_discriminator, model_generator.trainable_weights))

  return model_generator, model_loss.numpy()[0][0], generator_model_result

def start_training(dataset,target,epochs,learning_rate_discriminator,learning_rate_generator,tries,telegram_information):

  model_generator, model_discriminator  = generator(), discriminator()
  optimizer_generator, optimizer_discriminator  = tf.keras.optimizers.Adam(learning_rate=learning_rate_generator), tf.keras.optimizers.Adam(learning_rate=learning_rate_discriminator)

  smallest_lost, count, epoch_loss_all, acc_over_epochs = float('inf'), 0, [], []

  noisy_imgs_generated = tf.reshape(generate_noisy_image(),(1,784))
  
  for epoch in range(epochs):
    
    print("\nStart of epoch %d" % (epoch,))
    epoch_loss = 0
    display_img(tf.reshape(noisy_imgs_generated[0][0:784],(28,28)),y_label='epoch:'+str(epoch),title='epoch'+str(epoch)+'.png',save_fig=True,show_img=False)    
    
    if telegram_information: 
      send_msg_telegram(f'start of epoch: {epoch}....')
      send_image('./imgs/'+'epoch'+str(epoch)+'.png')

    generator_row, discriminator_row, prediction_discriminator_result_in_real,prediction_discriminator_result_in_generated = [], [], [], []

    for index_img in tqdm(range(len(dataset))):
      
      #generator model training
    
      model_generator, generator_loss, noisy_imgs_generated = update_generator_weights(model_generator,model_discriminator,noisy_imgs_generated,optimizer_generator,target[index_img])
      
      epoch_loss += generator_loss
      generator_row.append(generator_loss)
      
      #discriminator model training in real image
      model_discriminator, discriminator_loss, model_output = update_discriminator_weights(model_discriminator,dataset[index_img],target[index_img],True,optimizer_discriminator)
      
      epoch_loss += discriminator_loss
      discriminator_row.append(discriminator_loss)
      prediction_discriminator_result_in_real.append(1 if model_output > 0.5 else 0 )
      
      #discriminator model training in fake image
      model_discriminator, discriminator_loss, model_output = update_discriminator_weights(model_discriminator,noisy_imgs_generated,target[index_img],False,optimizer_discriminator)
      
      epoch_loss += discriminator_loss
      discriminator_row.append(discriminator_loss)
      prediction_discriminator_result_in_generated.append(1 if model_output > 0.5 else 0 )
                  
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
    
    if math.isnan(epoch_loss):
      print("Nan  Found")
      if telegram_information:
        send_msg_telegram("Nan Found")
      break
    
    if keep_learning: break
    dataset, target = shuffle(dataset,target)
   
  acc_over_epochs = np.array(acc_over_epochs)  
  
  model_generator.save(paths["model_generator"]+"/generator.keras")
  model_discriminator.save(paths["model_discriminator"]+"/discrminator.keras")
  
  if telegram_information: send_msg_telegram("end of the traning")
  
  save_img(epoch_loss_all,title='loss_over_epoch',path=paths['imgs'])
  save_img(acc_over_epochs[:,0],title='accucary_over_epoch_in_real',path=paths['imgs'])
  save_img(acc_over_epochs[:,1],title='accucary_over_epoch_in_generated',path=paths['imgs'])
  