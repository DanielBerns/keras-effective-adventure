# import the necessary packages
import os
from .Processors import ContextException

class Loader:
    def __init__(self, source, context, processors):
        # source is the object with a read method
        # processors is a list of objects with an execute method
        # Define the dataset source
        self._source = source
        self._context = context
        # define the dataset processors
        self._processors = processors
        for p in processors:
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
        count = -1
        for (image, label) in self.source.read():
            count += 1
            self.context.count = count
            self.context.image = image
            self.context.label = label
            try:
                for this_processor in self.processors:
                    this_processor.execute()
                self.context.add_sample_and_label()
            except ContextException as message:
                print(message)
        self.context.stop()
