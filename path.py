import os

current_directory = os.getcwd()

paths = {
    "imgs": os.path.join(current_directory,"imgs"),
    "model_discriminator":os.path.join(current_directory,"model","discriminator"),
    "model_generator":os.path.join(current_directory,"model","generator"),
    "discriminator_loss":os.path.join(current_directory,"imgs","discriminator_loss"),
    "generator_loss":os.path.join(current_directory,"imgs","generator_loss"),
    "generated_imgs":os.path.join(current_directory,"generated_imgs")
}

