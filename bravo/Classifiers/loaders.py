class Loader:
    def __init__(self, source, context, processors):
        # source is an object with a read method returning an image and a label
        # context is a data container for image operations
        # processors is a list of objects with an execute method operating on
        #    the context data
        self._source = source
        self._context = context
        # Connect the processors with the context
        self._processors = processors
        for p in self.processors:
            p.context = context

    @property
    def source(self):
        return self._source

    @property
    def context(self):
        return self._context
    
    @property
    def processors(self):
        return self._processors
    
    def execute(self):
        self.context.start()
        for (image, label) in self.source.read():
            self.context.image = image
            self.context.label = label
            try:
                for this_processor in self.processors:
                    this_processor.execute()
                self.context.add_sample()
            except Exception as message:
                print(message)
        self.context.stop()
