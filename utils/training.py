import numpy as np
from path import paths
from utils.models import generator, discriminator
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import math
from telegram_bot.sender import send_msg_telegram, send_image
import tensorflow as tf
from tqdm import tqdm
from utils.data import generate_noisy_image,reshape_and_concat_794,calculate_batch_loss_generator, calculate_batch_loss_discriminator
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

def update_discriminator_weights(model,imgs,targets,is_real_img,optimizer,batch):  
  
  target_arrays = [[target for _ in range(10)] for target in targets]
  input_discriminator = [reshape_and_concat_794(imgs[i],target_arrays[i]) for i in range(len(imgs))] 
  model_results = []
  
  with tf.GradientTape() as tape:
    
    for i in range(batch):
      model_results.append(model(input_discriminator[i]))
      
    model_loss = calculate_batch_loss_discriminator(model_results,is_real_img)
    
  grads_discriminator = tape.gradient(model_loss, model.trainable_weights)
  optimizer.apply_gradients(zip(grads_discriminator, model.trainable_weights))
  
  return model, model_loss.numpy(), model_results
    
def update_generator_weights(model_generator,model_discriminator,imgs,optimizer,targets,batch): 
 
  target_arrays = [[target for _ in range(10)] for target in targets]
  noisy_imgs_794 = [reshape_and_concat_794(imgs[i],target_arrays[i]) for i in range(len(imgs))]
  
  batch_loss, discriminator_model_results, generator_model_results = 0, [], []

  with tf.GradientTape() as tape:
    
    for i in range(batch):
      
      generator_model_result = model_generator(noisy_imgs_794[i])
      generated_img_794 = reshape_and_concat_794(generator_model_result,target_arrays[i])
      discriminator_model_results.append(model_discriminator(generated_img_794))
      generator_model_results.append(generator_model_result)
      
    batch_loss = calculate_batch_loss_generator(discriminator_model_results)
  
  grads_discriminator = tape.gradient(batch_loss, model_generator.trainable_weights)
  optimizer.apply_gradients(zip(grads_discriminator, model_generator.trainable_weights))
  
  return model_generator, batch_loss.numpy(), generator_model_results

def start_training(dataset,target,epochs,learning_rate_discriminator,learning_rate_generator,tries,telegram_information,batch,sleep_discriminator,sleep_discriminator_acc):

  model_generator, model_discriminator  = generator(), discriminator()
  optimizer_generator, optimizer_discriminator  = tf.keras.optimizers.Adam(learning_rate=learning_rate_generator), tf.keras.optimizers.Adam(learning_rate=learning_rate_discriminator)

  smallest_lost, count, epoch_loss_all, acc_over_epochs, dataset_size, count_discriminator = float('inf'), 0, [], [], len(dataset), 0
  acc_of_epoch_1, acc_of_epoch_2 = 0, 0
  
  noisy_imgs_generated = [tf.reshape(generate_noisy_image(),(1,784)) for _ in range(batch)]
  
  for epoch in range(epochs):
    
    print("\nStart of epoch %d" % (epoch,))
    epoch_loss = 0
    display_img(tf.reshape(noisy_imgs_generated[0][0:784],(28,28)),y_label='epoch:'+str(epoch),title='epoch'+str(epoch)+'.png',save_fig=True,show_img=False)    
    
    if telegram_information: 
      send_msg_telegram(f'start of epoch: {epoch}....')
      send_image('./imgs/'+'epoch'+str(epoch)+'.png')

    generator_row, discriminator_row, prediction_discriminator_result_in_real,prediction_discriminator_result_in_generated = [], [], [], []

    for index_img in tqdm(range(0,dataset_size-batch,batch)):
      
      real_img_batch, target_batch = dataset[index_img:index_img+batch], target[index_img:index_img+batch]
      
      #generator model training      
      model_generator, generator_loss, noisy_imgs_generated = update_generator_weights(model_generator,model_discriminator,noisy_imgs_generated,optimizer_generator,target_batch,batch)
      epoch_loss += generator_loss
      generator_row.append(generator_loss) 
      
      # discriminator model training
      
      if sleep_discriminator == -1 and sleep_discriminator_acc >= (acc_of_epoch_1+acc_of_epoch_2)/2: 
        
        model_discriminator, discriminator_loss_real_img, model_output_real_img = update_discriminator_weights(model_discriminator,real_img_batch,target_batch,True,optimizer_discriminator,batch)
        model_discriminator, discriminator_loss_noisy_img, model_output_noisy_img = update_discriminator_weights(model_discriminator,noisy_imgs_generated,target_batch,False,optimizer_discriminator,batch)
        
        epoch_loss += (discriminator_loss_noisy_img + discriminator_loss_real_img)
        discriminator_row.append(discriminator_loss_noisy_img)
        discriminator_row.append(discriminator_loss_real_img)
        
        prediction_discriminator_result_in_real = tf.concat([prediction_discriminator_result_in_real,[1 if output > 0.5 else 0 for output in model_output_real_img]],axis=0)
        prediction_discriminator_result_in_generated = tf.concat([prediction_discriminator_result_in_generated,[1 if output > 0.5 else 0 for output in model_output_noisy_img]],axis=0)
      
      elif count_discriminator >= sleep_discriminator and sleep_discriminator != -1:
        
        model_discriminator, discriminator_loss_real_img, model_output_real_img = update_discriminator_weights(model_discriminator,real_img_batch,target_batch,True,optimizer_discriminator,batch)
        model_discriminator, discriminator_loss_noisy_img, model_output_noisy_img = update_discriminator_weights(model_discriminator,noisy_imgs_generated,target_batch,False,optimizer_discriminator,batch)
        
        epoch_loss += (discriminator_loss_noisy_img + discriminator_loss_real_img)
        discriminator_row.append(discriminator_loss_noisy_img)
        discriminator_row.append(discriminator_loss_real_img)
        
        prediction_discriminator_result_in_real = tf.concat([prediction_discriminator_result_in_real,[1 if output > 0.5 else 0 for output in model_output_real_img]],axis=0)
        prediction_discriminator_result_in_generated = tf.concat([prediction_discriminator_result_in_generated,[1 if output > 0.5 else 0 for output in model_output_noisy_img]],axis=0)

        count_discriminator = 0
    
      count_discriminator+=1  
      
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
  
