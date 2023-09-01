import numpy as np
import os
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