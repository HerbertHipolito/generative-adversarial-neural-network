import numpy as np
from utils.plot_print import print_dataset_according_to_keys
from utils.data import divide_dataset
from tensorflow.keras.datasets import mnist
import argparse
from telegram_bot.sender import send_msg_telegram
from utils.GAN import start_training

#https://www.tensorflow.org/guide/checkpoint?hl=pt-br
#https://stackoverflow.com/questions/46974047/generative-adversarial-network-generating-image-with-some-random-pixels
#https://arxiv.org/pdf/1511.06434v2.pdf

def parse_opt():

  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs',type=int,default=200)
  parser.add_argument('--tries',type=int,default=20)
  parser.add_argument('--learning_rate_discriminator',type=float,default=3e-5)
  parser.add_argument('--learning_rate_generator',type=float,default=3e-5)
  parser.add_argument('--telegram_information',type=bool,default=False)  
  opt = parser.parse_args()
  print(vars(opt)) 
  return opt
  
if __name__ == '__main__':

  opt = parse_opt()
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  dataset, target = [*x_train,*x_test], [*y_train,*y_test]
  
  selected_numbers, parameters = [1,4,7], vars(opt)
  
  if parameters['telegram_information']: 
    send_msg_telegram("The model is going to be trained with the following parameters: "+str(vars(opt))) 
    send_msg_telegram(f"The numbers selected were {selected_numbers}")
  #print(f"The numbers selected were {selected_numbers}")
  
  dataset = np.array([np.reshape(img/255,(1,784)) for img in dataset])
  dataset = divide_dataset(dataset,target,selected_numbers)
  print_dataset_according_to_keys(dataset,selected_numbers)
  
  start_training(dataset,selected_numbers,**parameters)
    
