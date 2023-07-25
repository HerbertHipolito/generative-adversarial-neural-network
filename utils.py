import matplotlib.pyplot as plt
import numpy as np

def display_img(image,title='Image',x_label=None,y_label=None,show_axis=True,colorBar=False,size=(3,2),save_fig=False,path='./imgs',show_img=True):

  plt.gray()
  plt.matshow(image)
  if save_fig: plt.savefig(path+'/'+title,format='png')
  if colorBar: plt.colorbar()
  if not show_axis: plt.axis('off')
  if show_img:plt.show()
  plt.close()
  plt.clf()


def save_img(y,title=None,x_label='epoch',y_label='loss',path='./imgs',label=None,grid=True):

  plt.plot([i for i in range(len(y))],y,label=label)
  if label is not None: plt.legend()
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if grid: plt.grid()
  plt.savefig(path+'/'+title+'.png')
  plt.clf()
  plt.close()

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

def early_stopping(current_loss, smallest_loss, count, tries = 10, e = 0.005):
  
  feedback = ''
  
  if current_loss + e <= smallest_loss: 
    count = 0
    smallest_loss = current_loss
  else:
    count +=1
    feedback = f"No improviment found: {count}"
  
  return smallest_loss, count, True if count >= tries else False, feedback

def divide_dataset(dataset,target,selected_numbers):
  
  new_dataset = {}
  
  for index,selected_number in enumerate(selected_numbers):
    
    new_dataset[index] = [ img for index_img,img in enumerate(dataset) if target[index_img] == selected_number]
  
  return new_dataset