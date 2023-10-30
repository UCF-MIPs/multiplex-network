import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

def part_coef_plot_out(inf, df, name, results_dir):
    infl_type = str(inf + '_*')
    #pc_sc = sns.scatterplot(data=df,\
    #                        x=infl_type,\
    #                        y='pc_out')
    pc_sc = sns.jointplot(data=df,\
                            x=infl_type,\
                            y='pc_out',
                            xlim=(0,25),
                            ylim=(0,1)
                            )
    pc_sc.plot(sns.scatterplot, sns.histplot)
    #pc_sc.set(xlabel=f'Total TE of {name} {inf} Sources in the Multiplex Network',\
    #          ylabel=f'Participant Coefficient of {name} {inf} Sources') 
    pc_sc.set_axis_labels(f'Total TE of {name} {inf} Sources', 'Participant Coefficient')
    #pc_sc.set_xlim(0,25)
    #pc_sc.set_ylim(0,1)
    #pc_sc.set_title(f'Participation Coefficient of {name} {inf} Sources')
    #fig = pc_sc.get_figure()
    pc_sc.fig.suptitle(f'Participation Coefficient of {name} {inf} Sources')
    #fig.savefig(f'{results_dir}/{name}_{inf}_pc_out.png')
    plt.savefig(f'{results_dir}/{name}_{inf}_pc_out.png')
    
def part_coef_plot_in(inf, df, name, results_dir):
    infl_type = str('*_' + inf)
    pc_sc = sns.jointplot(data=df,\
                            x=infl_type,\
                            y='pc_in',
                            xlim=(0,25),
                            ylim=(0,1)
                            )
    pc_sc.plot(sns.scatterplot, sns.histplot)
    
    #pc_sc.set(xlabel=f'Total TE of {name} {inf} Targets in the Multiplex Network',\
    #          ylabel=f'Participant Coefficient of {name} {inf} Targets')
    pc_sc.set_axis_labels(f'Total TE of {name} {inf} Targets', 'Participant Coefficient')
    #pc_sc.set_xlim(0,25)
    #pc_sc.set_ylim(0,1)
    #pc_sc.set_title(f'Participation Coefficient of {name} {inf} Targets')
    pc_sc.fig.suptitle(f'Participation Coefficient of {name} {inf} Targets')
    #fig = pc_sc.get_figure()
    plt.savefig(f'{results_dir}/{name}_{inf}_pc_in.png')


