
import matplotlib.pyplot as plt

def plot_path(x):
    for t in x:
        plt.plot(t[0], t[1], "o")


