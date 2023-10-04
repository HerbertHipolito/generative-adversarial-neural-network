from path import current_directory
import os 
from datetime import datetime

def new_action(action,parameters=None):
    
    file_path = 'log.txt'

    current_time_str = datetime.now().strftime("%H:%M:%S")

    with open(os.path.join(current_directory,file_path), 'a') as file:
        
        file.write(f'\n---------{action} {current_time_str}---------\n')
        if parameters != None: 
            
            file.write('---------Parameters used---------\n')
            for key in parameters.keys():
                file.write(f'{key}: {parameters[key]}\n')

