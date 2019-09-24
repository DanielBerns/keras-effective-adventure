import numpy as np

from Classifiers import create_custom_image_dataset, Classifier, build_AlexNet

dataset = create_custom_image_dataset((224, 224, 3), ["/home/dberns/Datasets/dl4cv/animals/"])

train_X, train_y, validation_X, validation_y, test_X, test_y, data_shape, set_of_labels = dataset.get()

images = dataset.images
onehots = dataset.onehots
set_of_labels = dataset.set_of_labels

print('images', len(images))
print('onehots', len(onehots))
print('set_of_labels', str(set_of_labels))

print('train_X.shape', train_X.shape)
print('train_y.shape', train_y.shape)
print('validation_X.shape', validation_X.shape)
print('validation_y.shape', validation_y.shape)
print('test_X.shape', test_X.shape)
print('test_y.shape', test_y.shape)
print('train_y[0]', train_y[0,:], 'validation_y[1]', validation_y[1,:], 'test_y[2]', test_y[2,:])

print(np.sum(train_y,axis=0))
print(np.sum(validation_y,axis=0))
print(np.sum(test_y,axis=0))

model = build_AlexNet(data_shape, len(set_of_labels))
classifier = Classifier()
classifier.train_epochs = 100
classifier.output = 'output/mnist/AlexNet-alpha'
classifier.build(model, 
                 train_X, train_y, test_X, test_y, set_of_labels, 
                 validation_X=validation_X, validation_y=validation_y)
