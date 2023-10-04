import numpy as np
import os
import tensorflow as tf
from PIL import Image

def divide_dataset(dataset,target,selected_numbers):
  
  new_dataset = {}
  
  for index,selected_number in enumerate(selected_numbers):
    new_dataset[index] = [ img for index_img,img in enumerate(dataset) if target[index_img] == selected_number]
  
  return new_dataset

def generate_noisy_image(mean=0.5,std=0.2):

  noisy_image = np.zeros(784)

  for index in range(784):

    noisy_image[index] = np.random.normal(mean,std) 
  
  return noisy_image

def oneHotEncodding(target):
  
  new_target = []

  for number in target:
    row = [0,0,0,0,0,0,0,0,0,0]
    row[number] = 1
    new_target.append(row)

  return new_target

def is_valid_array(user_array):

  return all(0 <= element <= 9 and isinstance(element, int) for element in user_array)

def create_frames(folder_path): 
 
  if not os.path.exists(folder_path): raise Exception(f'Path {folder_path} not exists')
  if not os.path.isdir(folder_path): raise Exception(f'Path {folder_path} is not folder')
   
  subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
  
  for index, subfolder in enumerate(subfolders):
  
    img_folder, frames = os.path.join(folder_path,subfolder), []

    for img_name in os.listdir(img_folder):
      
      img_path = os.path.join(img_folder,img_name)
      frames.append(Image.open(img_path))

    # Fix the error when a gif already exist!!!
    
    frames[0].save(os.path.join(img_folder,f'gif{index}.gif'), save_all=True, append_images=frames[1:], loop=0, duration=300) 
    
  return frames 

@tf.function
def reshape_and_concat_794(img,target_array):
  return tf.reshape(tf.concat([img[0],target_array],axis=0),(1,794))

@tf.function
def calculate_batch_loss_generator(discriminator_model_results):
  return tf.reduce_mean([tf.math.log(tf.math.subtract(tf.cast(1,tf.float32),discriminator_model_result)) for discriminator_model_result in discriminator_model_results])

@tf.function
def calculate_batch_loss_discriminator(model_results,is_real_img):
  return tf.reduce_mean([(-1)*tf.math.log(model_result) if is_real_img else (-1)*tf.math.log(tf.math.subtract(tf.cast(1,tf.float32),model_result)) for model_result in model_results])