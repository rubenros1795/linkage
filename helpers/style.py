
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def load_style():

    plt.rcParams['font.family'] = 'Aptos'
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 22

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    # mpl.rcParams.update({"axes.grid" : True, "grid.color": "lightgrey"})
    plt.rcParams['axes.unicode_minus'] = False

    sns.set_palette(sns.color_palette('BrBG'))