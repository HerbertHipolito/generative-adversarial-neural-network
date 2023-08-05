import matplotlib.pyplot as plt

def print_dataset_according_to_keys(dataset,selected_numbers):
 
  sum = 0 
  
  for index,key in enumerate(dataset.keys()):
      
    print(f"{key} => {selected_numbers[index]} => {len(dataset[key])}")
    sum += len(dataset[key])
    
  print(f"Total dataset size is {sum}")

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