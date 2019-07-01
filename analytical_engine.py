import matplotlib.pyplot as plt
import numpy as np


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

        :param aggregator: function to apply to each batch such that the output is one number
        """

        x_values = range(len(self.values))  # Create x-axis
        data = list(map(aggregator, self.values))

        plt.plot(x_values, data)
        plt.show()
