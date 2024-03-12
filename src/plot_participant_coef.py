import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

### Participant Coefficient Plot for 4 Layers ###
def part_coef_plot_out_4layers(inf, df, name, dic, results_dir, scatter_color, hist_color):
    infl_type = str(inf + '_*')

    pc_sc = sns.jointplot(data=df,\
                            x=infl_type,\
                            y='pc_out_4',
                            xlim=(0,1.1*df[infl_type].max()),
                            ylim=(0,1)
                            )
    #pc_sc.plot(sns.scatterplot, sns.histplot)
    # Change scatter plot color
    pc_sc = pc_sc.plot_joint(sns.scatterplot, color=scatter_color)
    # Change histogram color
    pc_sc = pc_sc.plot_marginals(sns.histplot, color=hist_color)
    pc_sc.set_axis_labels(f'Total TE of {name} {inf} Sources', 'Participant Coefficient')
    pc_sc.ax_joint.axhline(y=0.66, color='g', linestyle='--', label='Participant Coefficient when influence is fully distributed over two layers')
    pc_sc.ax_joint.axhline(y=0.89, color='r', linestyle='--', label='Participant Coefficient when influence is fully distributed over three layer')
    pc_sc.ax_joint.legend(fontsize='xx-small', loc='lower center', bbox_to_anchor=(0.4, 0), fancybox=True, shadow=True)
    # pc_sc.set_axis_labels(' ', ' ')
    # pc_sc.figure.suptitle(f'{dic[name]} {inf} Sources', size = 20)
    plt.savefig(f'{results_dir}/{name}_{inf}_pc_out.png')

### Participant Coefficient Plot for 2 Layers ###
def part_coef_plot_out_2layers(inf, df, name, dic, results_dir):
    infl_type = str(inf + '_*')
    pc_sc = sns.jointplot(data=df,\
                            x=infl_type,\
                            y='pc_out_2',
                            xlim=(0,1.1*df[infl_type].max()),
                            ylim=(0,1)
                            )
    pc_sc.plot(sns.scatterplot, sns.histplot)
    if inf == 'T':
        pc_sc.set_axis_labels(f'Total TE of {name} Trustworthy Sources', 'Participant Coefficient')
        #pc_sc.set_axis_labels(' ', ' ')
        pc_sc.figure.suptitle(f'{dic[name]} Trustworthy Sources', size = 20)
    elif inf == 'U':
        pc_sc.set_axis_labels(f'Total TE of {name} Untrustworthy Sources', 'Participant Coefficient')
        #pc_sc.set_axis_labels('', '')
        pc_sc.figure.suptitle(f'{dic[name]} Untrustworthy Sources', size = 20)
    plt.savefig(f'{results_dir}/{name}_{inf}_pc_out.png')
