#%%

import os

import pandas as pd
import seaborn as sns
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt

from statannotations.Annotator import Annotator

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

EXCLUSION = [
    'AD261_2023-09-07_18-33-01'
]

PLOT_ALL = False
W_SIG = False

### 360px = 30cm; 240px = 20cm (px/12=cm)

palette1 = sns.color_palette(['#ffffff', '#272927'])
# palette1 = sns.color_palette(['#272927', '#8fcde6', '#38b01e'])
palette2 = sns.color_palette(['#272927', '#008000'])


result_path = '../GridWalking-wyc-2023-10-30/new_analyzed/result.csv'

# reading and excluding outliers
df = pd.read_csv(result_path)
df = df[~(df['video_name'].isin(EXCLUSION))]

df['ratio'] = df['prop_center'] / df['prop_peripheral']
df['cm_per_sec'] = df['px_per_second'] / 12
df['95th_cm_per_sec'] = df['95th_px_per_second'] / 12
df['filtered_mm_per_sec'] = df['filtered_px_per_second'] / 12 * 10

df.to_csv('../GridWalking-wyc-2023-10-30/new_analyzed/to_cm.csv')


hue = 'type'
col = 'loc'
x = 'month'
y = 'filtered_mm_per_sec'

# y = "cm_per_sec"
# y = 'ratio'

# plot of all
# hue_order = pd.unique(df[hue])
hue_order = ['WT', 'AD']
col_order = pd.unique(df[col])
order = ['3-4M', '5-8M', '9-12M']

pairs = [[(o, ho) for ho in hue_order] for o in order]

if PLOT_ALL:

    plot = sns.catplot(
        x=x, y=y, hue=hue, col=col, data=df, kind="bar",
        order=order, hue_order=hue_order, col_order=col_order,
        palette=palette2, height=4, aspect=.7, alpha=0.5,)

    # pairs = [[(o, ho) for ho in hue_order] for o in order]
    if W_SIG:
        for ax in plot.axes[0]:
            sub_df = pd.DataFrame([row for _, row in df.iterrows() if row[col] in ax.get_title()])
            annotator = Annotator(ax=ax, data=sub_df, pairs=pairs,
                                x=x, hue=hue, y=y, 
                                order=order, hue_order=hue_order)
            annotator.configure(test="t-test_ind", text_format="star", loc="inside")
            annotator.apply_and_annotate()

    # fig = plot.get_figure()
    plot.savefig(f"out/all_result__{y}.tiff") 

#%%
##################################################################
####### saving mid spine plot
sub_df = df[df['loc']=='mid_spine']
mid_plot = sns.catplot(
    x=x, y=y, hue=hue, data=sub_df, kind='bar',
    order=order, hue_order=hue_order, legend=False,
    palette=palette2, alpha=0.5,
    height=8, aspect=1., edgecolor='white', linewidth=2,
    ci=95, n_boot=1000, errwidth=2, capsize=.1,)

# mid_plot.fig.subplots_adjust(wspace=1)

ax = mid_plot.ax
ax.grid(axis='y', color='grey', linestyle='-', alpha=0.1, linewidth=1)

sns.swarmplot(x=x, y=y, data=sub_df, hue=hue, order=order, 
              hue_order=hue_order, 
              size=12, 
              palette=palette2, alpha=0.9, ax=ax, dodge=True,
              edgecolor='white', linewidth=2)

handles, labels = ax.get_legend_handles_labels()


ax.legend(
    handles=handles[:len(hue_order)], labels=labels[:len(hue_order)],
    loc='center left', bbox_to_anchor=(1, 0.5), fontsize=25)

ax.set_xlabel('Age', weight='bold',fontsize=35)
# ax.set_ylabel('percentage moving less than 1 px/fr', fontsize=30)
ax.set_ylabel('Travel speed (mm/s)', weight='bold', fontsize=35)
# ax.set_ylabel('Center:Peripheral Ratio', fontsize=30)
ax.set_xticklabels(['3-4M', '6-7M', '10-11M'], weight='bold',
                   rotation=40, horizontalalignment='right')

ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

for i, bar in enumerate(ax.patches):  # iterate over the bars
    # Calculate the center x position of each bar
    x_value = bar.get_x() + bar.get_width() / 2
    ax.axvline(x=x_value, color='grey', linestyle='-', alpha=0.1, linewidth=1)  # add the gridline

axis_width = 2
for _, spine in ax.spines.items():
    spine.set_linewidth(axis_width)
for axis in [ax.xaxis, ax.yaxis]:
    for tick in axis.get_major_ticks():
        tick.tick1line.set_linewidth(axis_width)
        tick.tick2line.set_linewidth(axis_width)


if W_SIG:
    sig_pairs = pairs + [((first, ho), (second, ho)) for first, second in combinations(order, 2) for ho in hue_order]

    annotator = Annotator(ax=mid_plot.ax, data=df, pairs=sig_pairs,
                        x=x, hue=hue, y=y, 
                        order=order, hue_order=hue_order)
    annotator.configure(test="t-test_ind", loc="inside",
                        text_offset=0.5, comparisons_correction="BH",
                        hide_non_significant=True, 
                        text_format='star',
                        line_offset=0.01, line_offset_to_group=0.5)
    annotator.apply_and_annotate()

    mid_plot.savefig(f"out/mid_spine__{y}_wsig.tiff") 
else:
    mid_plot.savefig(f"out/mid_spine__{y}_nosig.tiff") 


######## plotting for ratio
y = 'ratio'

sub_df = df[df['loc']=='mid_spine']
mid_plot = sns.catplot(
    x=x, y=y, hue=hue, data=sub_df, kind='bar',
    order=order, hue_order=hue_order, legend=False,
    palette=palette2, alpha=0.5,
    height=8, aspect=1., edgecolor='white', linewidth=2,
    ci=95, n_boot=1000, errwidth=2, capsize=.1,)

# mid_plot.fig.subplots_adjust(wspace=1)

ax = mid_plot.ax
ax.grid(axis='y', color='grey', linestyle='-', alpha=0.1, linewidth=1)

sns.swarmplot(x=x, y=y, data=sub_df, hue=hue, order=order, 
              hue_order=hue_order, 
              size=12, 
              palette=palette2, alpha=0.9, ax=ax, dodge=True,
              edgecolor='white', linewidth=2)

handles, labels = ax.get_legend_handles_labels()


ax.legend(
    handles=handles[:len(hue_order)], labels=labels[:len(hue_order)],
    loc='center left', bbox_to_anchor=(1, 0.5), fontsize=25)

ax.set_xlabel('Age', weight='bold',fontsize=35)
ax.set_ylabel('Center:Peripheral Ratio', weight='bold',fontsize=35)
ax.set_xticklabels(['3-4M', '6-7M', '10-11M'],weight='bold',
                   rotation=40, horizontalalignment='right')

ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)

for i, bar in enumerate(ax.patches):  # iterate over the bars
    # Calculate the center x position of each bar
    x_value = bar.get_x() + bar.get_width() / 2
    ax.axvline(x=x_value, color='grey', linestyle='-', alpha=0.1, linewidth=1)  # add the gridline

axis_width = 2
for _, spine in ax.spines.items():
    spine.set_linewidth(axis_width)
for axis in [ax.xaxis, ax.yaxis]:
    for tick in axis.get_major_ticks():
        tick.tick1line.set_linewidth(axis_width)
        tick.tick2line.set_linewidth(axis_width)

ax.axhline(y=1., color='black', linestyle='--', linewidth=1)

if W_SIG:
    sig_pairs = pairs + [((first, ho), (second, ho)) for first, second in combinations(order, 2) for ho in hue_order]

    annotator = Annotator(ax=mid_plot.ax, data=df, pairs=sig_pairs,
                        x=x, hue=hue, y=y, 
                        order=order, hue_order=hue_order)
    annotator.configure(test="t-test_ind", loc="inside",
                        text_offset=0.5, comparisons_correction="BH",
                        hide_non_significant=True, 
                        text_format='star',
                        line_offset=0.01, line_offset_to_group=0.5)
    annotator.apply_and_annotate()

    mid_plot.savefig(f"out/mid_spine__{y}_wsig.tiff") 
else:
    mid_plot.savefig(f"out/mid_spine__{y}_nosig.tiff") 


# wide_df = sub_df.pivot_table(index=['video_name'], columns=['type', 'month'], values='filtered_mm_per_sec')
# wide_df.to_csv('../GridWalking-wyc-2023-10-30/new_analyzed/to_cm_wide.csv')






# #%%
# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# sub_df = df[df['loc'] == 'mid_spine']

# sub_df['month'] = pd.Categorical(sub_df['month'], ordered=True, categories = ['3-4M','5-8M','9-12M'])
# sub_df['type'] = pd.Categorical(sub_df['type'], ordered=False, categories = ['AD','WT'])

# md = smf.mixedlm(f"{y} ~ type * month", sub_df, groups=sub_df["month"])
# mdf = md.fit()
# print(mdf.summary())


#%%


# ######### comparing left and right shoulder
# shoulder_df = df[(df['loc']=='L_shoulder') | (df['loc']=='R_shoulder')]

# hue = 'loc'
# col = 'type'
# x = 'month'
# y = "px_per_second"

# hue_order = pd.unique(shoulder_df[hue])
# col_order = pd.unique(shoulder_df[col])
# order = ['3-4M', '5-8M', '9-12M']

# plot = sns.catplot(
#     x=x, y=y, hue=hue, col=col, data=shoulder_df, kind="bar",
#     order=order, hue_order=hue_order, col_order=col_order,
#     height=4, aspect=.7)

# pairs = [[(o, ho) for ho in hue_order] for o in order]
# for ax in plot.axes[0]:
#     sub_df = pd.DataFrame([row for _, row in shoulder_df.iterrows() if row[col] in ax.get_title()])
#     annotator = Annotator(ax=ax, data=shoulder_df, pairs=pairs,
#                         x=x, hue=hue, y=y, 
#                         order=order, hue_order=hue_order)
#     annotator.configure(test="t-test_paired", text_format="star", loc="inside")
#     annotator.apply_and_annotate()

# plot.savefig("out/LR_shoulder.tiff") 

# ####### comparing left and right thigh

# thigh_df = df[(df['loc']=='L_thigh') | (df['loc']=='R_thigh')]

# hue_order = pd.unique(thigh_df[hue])
# col_order = pd.unique(thigh_df[col])
# order = ['3-4M', '5-8M', '9-12M']

# plot = sns.catplot(
#     x=x, y=y, hue=hue, col=col, data=thigh_df, kind="bar",
#     order=order, hue_order=hue_order, col_order=col_order,
#     height=4, aspect=.7)

# pairs = [[(o, ho) for ho in hue_order] for o in order]
# # pairs = pairs + [(("SPP", "TPP"))]
# # st.write(pairs)
# for ax in plot.axes[0]:
#     sub_df = pd.DataFrame([row for _, row in thigh_df.iterrows() if row[col] in ax.get_title()])
#     annotator = Annotator(ax=ax, data=thigh_df, pairs=pairs,
#                         x=x, hue=hue, y=y, 
#                         order=order, hue_order=hue_order)
#     annotator.configure(test="t-test_paired", text_format="star", loc="inside")
#     annotator.apply_and_annotate()

# plot.savefig("out/LR_thigh.tiff") 
