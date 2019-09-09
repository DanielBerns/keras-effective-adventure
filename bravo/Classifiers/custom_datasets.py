def get_custom_dataset(expected_shape, base_paths):
    source = ImageWithLabelSource(base_paths)
    context = ImageContext(expected_shape=expected_shape)
    processors = [ResizeImageProcessor(expected_shape), ImageToArrayProcessor()]
    loader = Loader(source, context, processors)
    loader.execute()
    trainX, trainY, testX, testY = context.get_dataset()
    return trainX, trainY, testX, testY
