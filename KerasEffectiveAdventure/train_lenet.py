from pathlib import Path

from keras.optimizers import SGD

from Core import Trainer
from NN.conv import LeNet


def get_model(args):
    # initialize the model and the model_filename
    model = None
    print("[INFO] compiling model...")
    model = LeNet.build(32, 32, 3, 3) # build(width, height, depth, classes, nlf='relu')
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.005), metrics=["accuracy"])
    model_filename = str(Path(args['model'], 'lenet.h5').expanduser())
    return model, model_filename
    

trainer = Trainer()
model, model_filename = get_model(trainer.args)
trainer.execute(model)
model.save(model_filename)
