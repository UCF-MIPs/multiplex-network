import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

### Participant Coefficient Plot for 4 Layers ###
def part_coef_plot_out_4layers(inf, df, name, results_dir):
    infl_type = str(inf + '_*')
    pc_sc = sns.jointplot(data=df,\
                            x=infl_type,\
                            y='pc_out_4',
                            xlim=(0,df[infl_type].max() + 0.5),
                            ylim=(0,1)
                            )
    pc_sc.plot(sns.scatterplot, sns.histplot)
    pc_sc.set_axis_labels(f'Total TE of {name} {inf} Sources', 'Participant Coefficient')
    pc_sc.fig.suptitle(f'Participation Coefficient of {name} {inf} Sources')
    plt.savefig(f'{results_dir}/{name}_{inf}_pc_out.png')

### Participant Coefficient Plot for 2 Layers ###
def part_coef_plot_out_2layers(inf, df, name, results_dir):
    infl_type = str(inf + '_*')
    pc_sc = sns.jointplot(data=df,\
                            x=infl_type,\
                            y='pc_out_2',
                            xlim=(0,df[infl_type].max() + 0.5),
                            ylim=(0,1)
                            )
    pc_sc.plot(sns.scatterplot, sns.histplot)
    if inf == 'T':
        pc_sc.set_axis_labels(f'Total TE of {name} Trustworthy Sources', 'Participant Coefficient')
        pc_sc.fig.suptitle(f'Participation Coefficient of {name} Trustworthy Sources')
    elif inf == 'U':
        pc_sc.set_axis_labels(f'Total TE of {name} Untrustworthy Sources', 'Participant Coefficient')
        pc_sc.fig.suptitle(f'Participation Coefficient of {name} Untrustworthy Sources')
    plt.savefig(f'{results_dir}/{name}_{inf}_pc_out.png')
