import matplotlib as mpl
import matplotlib.cm as cm
from cycler import cycler


def plots_setup():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['lines.linewidth'] = 0.6
    mpl.rcParams['font.family'] = 'CMU Serif'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['axes.labelsize'] = 10

    colors = [cm.Dark2(i) for i in range(20)]
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)


def plot_border(ax):
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
