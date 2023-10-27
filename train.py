import numpy as np
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.datasets import mnist
import argparse
from telegram_bot.sender import send_msg_telegram
from utils.training import start_training
from utils.plot_print import print_parameters
from utils.data import is_valid_array
from utils.log import new_action

def parse_opt():

  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs',type=int,default=200)
  parser.add_argument('--tries',type=int,default=20,help="Early Stopping parameter.\n The number of attempts that the model will keep training with no improvement")
  parser.add_argument('--sleep_discriminator',type=int,default=0,help="Number of loops that Discriminator will not update the weights")
  parser.add_argument('--batch',type=int,default=1,help="Mini-batch size")
  parser.add_argument('--selected_numbers',type=int,default=[i for i in range(10)],help="The numbers in the images to be generated. More numbers will make the training take more time",nargs="+")
  parser.add_argument('--learning_rate_discriminator',type=float,default=3e-5)
  parser.add_argument('--learning_rate_generator',type=float,default=3e-5)
  parser.add_argument('--telegram_information',type=bool,default=False,help="Send information via the Telegram bot about the training process, including an image at the end of each epoch.\n It's needed to set the key and chat_id: Just go to telegram_bot => variables and replace them ")  
  opt = parser.parse_args()
  
  return opt
  
if __name__ == '__main__':

  opt = parse_opt()
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  dataset, target = [*x_train,*x_test], [*y_train,*y_test]
  
  parameters =  vars(opt)
  
  if not is_valid_array(parameters['selected_numbers']):
    raise ValueError("The select_numbers input passed is not valid. Check if all numbers are ranging from 0 to 9")
  
  selected_numbers = parameters['selected_numbers']
  
  if parameters['telegram_information']: 
    send_msg_telegram(f"The model is going to be trained with the following parameters: {print_parameters(parameters)}") 
    send_msg_telegram(f"The numbers selected were: {selected_numbers}")
  
  print(f"The model is going to be trained with the following parameters: {print_parameters(parameters)}")
  print(f"The numbers selected were {selected_numbers}")
  
  dataset = np.array([np.reshape(img/255,(1,784)) for index, img in enumerate(dataset) if target[index] in selected_numbers])
  target = [number for number in target if number in selected_numbers]
  print(len(dataset),len(target))
 
  new_action("TRAINING",parameters)
  del parameters["selected_numbers"]
  start_training(dataset,target,**parameters)
  new_action("action completed in")
