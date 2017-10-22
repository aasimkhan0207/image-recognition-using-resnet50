import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

model = resnet50.ResNet50() #  pre-trained model

target = image.load_img('wallet.jpg',target_size=(224,224)) # resizing it to 224x224 pixels (required by this model)

x = image.img_to_array(target) # image to np

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

pred = model.predict(x)

pred_classes = resnet50.decode_predictions(pred, top=2) # top 2 results

print(pred_classes[0]) # here 0 is index of first image (we have only one)
print(type(pred_classes[0]))
print('most matched item :',pred_classes[0][0][1])

