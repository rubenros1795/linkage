
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def load_style(font='Source Sans Pro', SMALL_SIZE=15,MEDIUM_SIZE=20,BIGGER_SIZE=30):

    plt.rcParams['font.family'] = font
    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 30

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    # mpl.rcParams.update({"axes.grid" : True, "grid.color": "lightgrey"})
    plt.rcParams['axes.unicode_minus'] = False
