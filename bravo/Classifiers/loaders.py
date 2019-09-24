class Loader:
    def __init__(self, source, processors, dataset):
        # source is an object with a read method returning an image and a label
        # dataset is a data container for image operations
        # processors is a list of objects with an execute method operating on
        #    the dataset data
        self._source = source
        self._dataset = dataset
        self._processors = processors
        # Connect the processors with the dataset
        for p in self.processors:
            p.dataset = dataset

    @property
    def source(self):
        return self._source

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def processors(self):
        return self._processors
    
    def execute(self):
        self.dataset.start()
        for (image, label) in self.source.read():
            self.dataset.current_image = image
            self.dataset.current_label = label
            try:
                for this_processor in self.processors:
                    this_processor.execute()
                self.dataset.add_sample()
            except Exception as message:
                print(message)
        self.dataset.stop()
