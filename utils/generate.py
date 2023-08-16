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
    
    portion_for_each_selected_number = img_number/len(selected_numbers)
    
    path_folder_img = os.path.join(paths['generated_imgs'],str(selected_numbers[0]))

    for index in range(img_number):
        
        current_number_representation = [selected_numbers[int(index/portion_for_each_selected_number)] for _ in range(10)]
        
        generator_input = tf.reshape(tf.concat([noisy_img[0],current_number_representation],axis=0),(1,794))
        noisy_img = model_generator(generator_input)
        generator_output_794 = tf.reshape(tf.concat([noisy_img[0],current_number_representation], axis=0),(1,794))
        discriminator_result = model_discriminator(generator_output_794)[0][0] 
        print(f"discriminator result:\n prob: {discriminator_result}\n class: {'real' if discriminator_result > 0.5 else 'fake' }")
        
        generated_img = np.reshape(noisy_img[0],(28,28))
        
        print(f"image {index+1} generated")
        
        if  (int(index/portion_for_each_selected_number) != int((index+1)/portion_for_each_selected_number) or index == 0) and index != img_number-1:
          
          noisy_img = np.reshape(generate_noisy_image(mean,std),(1,784))
          path_folder_img = os.path.join(paths['generated_imgs'],str( selected_numbers[int((index+1)/portion_for_each_selected_number)] ))
          
          try:
            os.mkdir(path_folder_img)
          except FileExistsError:
            print(f'A folder named simulation{int(index/portion_for_each_selected_number)} already exist. The filers inside will be overwritten.')
            
          display_img(np.reshape(noisy_img[0],(28,28)),show_img=show_img,path = path_folder_img ,save_fig=True,title="generated_"+str(index)+"_img.png")
          
        else: display_img(generated_img,show_img=show_img,path = path_folder_img ,save_fig=True,title="generated_"+str(index)+"_img.png")
        discriminator_plot.append(discriminator_result) 
        
    if show_discriminator_output:
      plt.plot(discriminator_plot)
      plt.show()
