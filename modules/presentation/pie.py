import pandas as pd
import matplotlib.pyplot as plt
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
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title

df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

# Prepare Data
df = df_raw.groupby('class').size().reset_index(name='counts')

# Draw Plot
fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=80)

data = [13782, 8172, 3641]
categories = ['Solo sweet-like', 'Solo sweet', 'Otro conjunto de tags']
explode = [0, 0, 0.1]


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}% ({:d})".format(pct, absolute)


wedges, texts, autotexts = ax.pie(data,
                                  autopct=lambda pct: func(pct, data),
                                  textprops={'color': "w"},
                                  colors=plt.cm.Dark2.colors,
                                  startangle=140,
                                  explode=explode
                                  )

# Decoration
ax.legend(wedges, categories, title="Tags", loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))
plt.setp(autotexts, size=BIGGER_SIZE, weight=700)
ax.set_title("Tags presentes en las moleculas")
plt.show()
