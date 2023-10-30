import matplotlib.pyplot as plt
import numpy as np
from path import paths
import keras
import tensorflow as tf
from utils.data import generate_noisy_image
from utils.plot_print import display_img
import os

def generate_img(selected_numbers,mean,std,img_number,show_img,show_discriminator_output):
  
    model_generator = keras.models.load_model(paths['model_generator']+'/'+'generator.keras')
    model_discriminator = keras.models.load_model(paths['model_discriminator']+'/'+'discriminator.keras')
    discriminator_plot = []
    noisy_img = tf.reshape(generate_noisy_image(mean,std),(1,784))
    
    path_folder_img, iteration_per_img = os.path.join(paths['generated_imgs'],str(selected_numbers[0])), 10 

    for selected_number in selected_numbers:
      
      print(f"\n Selected number: {selected_number} \n")
      
      path_folder_img = os.path.join(paths['generated_imgs'],str(selected_number))
      
      try:
          os.mkdir(path_folder_img)
      except FileExistsError:
          print(f'A folder named simulation{str(selected_number)} already exist. The filers inside will be overwritten.')
      
      for i in range(img_number):
        
        path_folder_img_iteration = os.path.join(path_folder_img,str(i))
        
        try:
            os.mkdir(path_folder_img_iteration)
        except FileExistsError:
            print(f'A folder named {i} already exist. The filers inside will be overwritten.')
          
        current_number_representation = [ selected_number for _ in range(10) ]
        
        noisy_img = np.reshape(generate_noisy_image(mean,std),(1,784))
      
        for j in range(iteration_per_img):
      
          generator_input = tf.reshape(tf.concat([noisy_img[0],current_number_representation],axis=0),(1,794))
          noisy_img = model_generator(generator_input)
          generator_output_794 = tf.reshape(tf.concat([noisy_img[0],current_number_representation], axis=0),(1,794))
          discriminator_result = model_discriminator(generator_output_794)[0][0] 
          print(f"discriminator result:\n prob: {discriminator_result}\n class: {'real' if discriminator_result > 0.5 else 'fake' }")
          
          generated_img = np.reshape(noisy_img[0],(28,28))
          
          print(f"{j} generated")
            
          display_img(generated_img,show_img=show_img,path = path_folder_img_iteration,save_fig=True,title="generated_"+str(i)+"_"+str(j)+"_img.png")

          discriminator_plot.append(discriminator_result) 
        
    if show_discriminator_output:
      plt.plot(discriminator_plot)
      plt.show()
