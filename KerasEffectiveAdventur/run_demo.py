from keras.optimizers import SGD

from Demo import Demo
from NN.conv import ShallowNet

def get_model(args):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.005), metrics=["accuracy"])
    return model
    
demo_nn = Demo()

model = get_model(demo_nn.args)
demo_nn.train(model)
