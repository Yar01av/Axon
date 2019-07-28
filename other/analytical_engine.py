import matplotlib.pyplot as plt
import numpy as np
import uuid
import datetime


class AggregPlotter:
    """
    PLots the values of aggregated batches
    """

    def __init__(self):
        self.values = []
        self.curr_batch = []

    def add_to_curr_batch(self, value):
        self.curr_batch.append(value)

    def finish_curr_batch(self):
        """
        Finish editing the current batch and save it
        """

        self.values.append(list(self.curr_batch))  # Save the contents of the batch
        self.curr_batch = []  # Prepare for the next batch

    def plot(self, aggregator=np.sum):
        """
        Apply the aggregator and plot

        :param aggregator: How to aggregate. Type: List(int) -> List(int) || int
        """

        x_values = range(len(self.values))  # Create x-axis
        data = list(map(aggregator, self.values))  # Aggregate and make compatible with matplotlib

        plt.plot(x_values, data)
        plt.show()


class Logger():
    """
    Logs the data into the console with additional information (TODO: and optionally saves)
    """

    def __init__(self):
        self.count = 0  # number of data points logged
        self.memory = []

    def remember(self, value):
        """
        Store the value in the loggers memory

        :param value: value to store
        :return:
        """
        self.memory.append(value)

    def forget(self):
        """
        Empty the memory

        :return:
        """
        self.memory = []

    def log(self, message, interval=1):
        """
        Flexibly logs the values

        :param message: Message to print before the value
        :param interval: printing resolution
        :return:
        """

        # On the first run, name the log
        if self.count == 0:
            print(f"Log ID: {uuid.uuid4()}")

        if self.count % interval == 0:
            print(f"{message} \t at {datetime.datetime.now()}")

        self.count += 1

