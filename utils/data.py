import numpy as np

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

      