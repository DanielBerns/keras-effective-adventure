from keras.optimizers import SGD

from Core import Coach
from NN.conv import LeNet

from pathlib import Path

def get_model(args):
    # initialize the optimizer and model
    model = None
    print("[INFO] compiling model...")
    model = LeNet.build(32, 32, 3, 3) # build(width, height, depth, classes, nlf='relu')
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.005), metrics=["accuracy"])
    return model
    
coach = Coach()
model_file = str(Path(coach.args['model'], 'lenet.h5').expanduser())
model = get_model(coach.args)
coach.train(model)
model.save(model_file)
