import numpy as np
from utils import generate_noisy_image, display_img, save_img, early_stopping, divide_dataset, print_dataset_according_to_keys
from tensorflow.keras.datasets import mnist
import argparse
from telegram_bot.sender import send_msg_telegram
from utils import start_training
#https://www.tensorflow.org/guide/checkpoint?hl=pt-br

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

  opt = parse_opt()
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  dataset, target = [*x_train,*x_test], [*y_train,*y_test]
  
  selected_numbers = [1,4,5,7]
  send_msg_telegram(f"The numbers selected were {selected_numbers}")
  #print(f"The numbers selected were {selected_numbers}")
  
  dataset = np.array([np.reshape(img/255,(1,784)) for img in dataset])
  dataset = divide_dataset(dataset,target,selected_numbers)
  print_dataset_according_to_keys(dataset,selected_numbers)
  
  start_training(dataset,selected_numbers,**vars(opt))
    
