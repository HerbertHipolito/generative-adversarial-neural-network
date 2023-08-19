import numpy as np
from tensorflow.keras.datasets import mnist
import argparse
from telegram_bot.sender import send_msg_telegram
from utils.training import start_training

def parse_opt():

  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs',type=int,default=200)
  parser.add_argument('--tries',type=int,default=20,help="Early Stopping parameter.\n The number of attempts that the model will keep training with no improvement")
  parser.add_argument('--batch',type=int,default=1,help="Mini-batch size")
  parser.add_argument('--selected_numbers',type=int,default=[i for i in range(10)],help="The numbers in the images to be generated. More number will make the training to take more time",nargs="+")
  parser.add_argument('--learning_rate_discriminator',type=float,default=3e-5)
  parser.add_argument('--learning_rate_generator',type=float,default=3e-5)
  parser.add_argument('--telegram_information',type=bool,default=False,help="Send information via the Telegram bot about the training process, including an image at the end of each epoch.\n It's needed to set the key and chat_id: Just go to telegram_bot => variables and replace them ")  
  opt = parser.parse_args()
  print(vars(opt)) 
  return opt
  
if __name__ == '__main__':

  opt = parse_opt()
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  dataset, target = [*x_train,*x_test], [*y_train,*y_test]
  
  parameters =  vars(opt)
  selected_numbers = parameters['selected_numbers']
  
  if parameters['telegram_information']: 
    send_msg_telegram("The model is going to be trained with the following parameters: "+str(vars(opt))) 
    send_msg_telegram(f"The numbers selected were {selected_numbers}")
  
  print(f"The numbers selected were {selected_numbers}")
  
  dataset = np.array([np.reshape(img/255,(1,784)) for index, img in enumerate(dataset) if target[index] in selected_numbers])
  target = [number for number in target if number in selected_numbers]
  print(len(dataset),len(target))
  
  start_training(dataset,target,**parameters)
    
