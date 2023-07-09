import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22
BIGGEST_SIZE = 24

plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title

data_ = {
    "bitter": 738,
    "fruity": 640,
    "sweet": 635,
    "green": 540,
    "floral": 309,
    "woody": 243,
    "herbal": 221,
    "waxy": 211,
    "fatty": 191,
    "odorless": 162,
    "spicy": 159,
    "fresh": 158,
    "nutty": 143,
    "citrus": 138,
    "earthy": 127,
    #"rose": 120,
    #"balsam": 117,
    #"sulfurous": 115,
    #"roasted": 115,
    #"oily": 112,
    #"tropical": 108,
    #"meaty": 103,
    #"apple": 100
}

data = {
    "bitter": 738,
    "fruity": 676,
    "green": 540,
    "floral": 309,
    "woody": 254
}

ind = np.arange(len(data))

fig, ax = plt.subplots()
g = ax.bar(list(data.keys()), list(data.values()), color="red")
ax.bar_label(g, padding=10)
palette = sns.color_palette("husl", len(data))

plt.title("Frequencia de los tags")
plt.xlabel("Tag")
plt.ylabel("Apariciones")
ax.set_ylim(0, 800)
ax = plt.bar(ind, list(data.values()), color=palette)
# ax.bar_label(ax.containers[0])
plt.xticks(ind, list(data.keys()))
plt.show()
