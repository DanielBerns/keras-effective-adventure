import numpy as np

from Classifiers import get_custom_image_context, Driver

context = get_custom_image_context((64, 64, 3), ["/home/dberns/Datasets/dl4cv/animals/"])

train_X, train_Y, validation_X, validation_Y, test_X, test_Y, data_shape, set_of_labels = context.get_dataset()

images = context.images
onehots = context.onehots

print('images', len(images))
print('onehots', len(onehots))

for this_onehot in onehots:
    print(this_onehot, context.get_label(this_onehot))

print('train_X.shape', train_X.shape)
print('train_Y.shape', train_Y.shape)
print('validation_X.shape', validation_X.shape)
print('validation_Y.shape', validation_Y.shape)
print('test_X.shape', test_X.shape)
print('test_Y.shape', test_Y.shape)
print(train_Y[0,:], validation_Y[1,:], test_Y[2,:])

print(np.sum(train_Y,axis=0))
print(np.sum(validation_Y,axis=0))
print(np.sum(test_Y,axis=0))

model = build_AlexNet(data_shape, len(set_of_labels))
driver = Driver()
driver.report(model, 
              train_X, train_Y, 
              test_X, test_Y, 
              set_of_labels, 
              validation_X=validation_X, validation_Y=validation_Y)
