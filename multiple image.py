import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

model = resnet50.ResNet50() #  pre-trained model


target1 = image.load_img('wallet.jpg',target_size=(224,224)) # resizing it to 224x224 pixels (required by this model)
target2 = image.load_img('mouse.jpg',target_size=(224,224))

x1 = image.img_to_array(target1) # image to np
x2 = image.img_to_array(target2)
x = np.array([x1,x2])

# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

pred = model.predict(x)

pred_classes = resnet50.decode_predictions(pred, top=1) # top 1 results

print('IMAGE 1 :most matched item :',pred_classes[0][0][1])

print('IMAGE 2 :most matched item :',pred_classes[1][0][1])

