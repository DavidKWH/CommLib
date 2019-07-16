# placeholder for plotting routines
import numpy as np
import matplotlib.pyplot as plt

def scatterc(cx, **kwargs):
    ''' plot complex data '''
    plt.scatter(cx.real, cx.imag, **kwargs)

