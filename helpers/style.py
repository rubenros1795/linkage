
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def load_style(font='CG Times'):
    plt.rcParams["font.family"] = font

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 22

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    mpl.rcParams.update({"axes.grid" : True, "grid.color": "lightgrey"})
    colors = '#3d5a80 #ee6c4d #17bebb #732c2c #a0b2a6 #61988e #e9eb87 #008148'.split(' ')
    sns.set_palette(sns.color_palette(colors))

    return sns.color_palette(colors)