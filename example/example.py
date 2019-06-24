import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def example_plot(
    data, x=None, y=None, title="Plot", figsize=(2, 2), kind="line"
):
    data.plot(x=x, y=y, title=title, figsize=figsize, kind=kind)
