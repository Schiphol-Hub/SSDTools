#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 07:39:22 2019

@author: edgordijn
"""

import pandas as pd
import numpy as np
from os.path import splitext
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import lines
from matplotlib import ticker
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from cycler import cycler

# -----------------------------------------------------------------------------
# uit de doc29lib
# -----------------------------------------------------------------------------
def read_file (filename, delimiter='\t', **kwargs):
    '''Importeer xls-, xlsx- of tekstbestand in een dataframe,
       als filename geen sting is dan is het waarschijnlijk al een dataframe'''    
    if isinstance(filename, str):
        _, ext = splitext(filename)
        if ext in ['.xls', '.xlsx']:
            return pd.read_excel(filename, **kwargs)
        else:
            print(delimiter)
            return pd.read_csv(filename, delimiter=delimiter, **kwargs)
    else:
         return filename   


# -----------------------------------------------------------------------------
# Pas de algemene plot style aan
# -----------------------------------------------------------------------------
def plot_style(style='MER2019', plottype='lijnplot'):
    ''' Algemene opmaak van een plot'''
    
    global xParams  # extra parameters t.o.v. rcParams
    xParams = dict()

    #Python_default = plt.rcParams.copy()
    
    # https://matplotlib.org/users/customizing.html
    # [(k,v) for k,v in plt.rcParams.items() if 'color' in k]
    
    if ':' in style:
        style, reeks = style.split(':')
    else:
        reeks = 1

    if style == 'MER2020':      
        # fonts
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif':'Frutiger for Schiphol Book', 'size':6})   

        # grid
        plt.rc('axes', axisbelow=True, grid=True)
        plt.rc('grid', color='#9491AA', linewidth=0.2, linestyle='solid')
        
        # spines en background
        plt.rc('axes', edgecolor='#9491AA', linewidth=0.2, facecolor='#EAE9EE')
#        xParams['hidespines'] = ['left', 'right']
        xParams['hidespines'] = []
        
        # labels
        plt.rc('axes', labelcolor='black', labelsize=10, labelpad=4)

        # label lines        
        xParams['labellineprop'] = {'linewidth':1, 'color':'black', 'marker':'None'}
        xParams['labellinemin'] = 1
        
        # tick marks en labels
        plt.rc('xtick', labelsize=6, color='black')
        plt.rc('ytick', labelsize=6, color='black')       
        
        # ticks
        plt.rc('xtick.major', size=0, width=0.5, pad=4)
        plt.rc('ytick.major', size=0, width=0.5, pad=4)
        plt.rc('xtick.minor', size=0, width=0.5, pad=4)
        plt.rc('ytick.minor', size=0, width=0.5, pad=4)
        
        # legend
        plt.rc('legend', markerscale=0.8, fontsize=6, frameon=False, borderaxespad=0)
        plt.rc('text', color='Black')
        xParams['legend'] = dict(loc='lower right', bbox_to_anchor=(1, 1))
    
        # lines en marker
        plt.rc('lines', linewidth=1,
                        markersize=4, 
                        marker='o',
                        markerfacecolor='None',
                        markeredgecolor='#141251',
                        markeredgewidth=0.5)
    
        # patches, o.a. voor een barplot
        plt.rc('patch', force_edgecolor=True,
                        linewidth=0.5,
                        edgecolor = 'white')
        # heatmap
        xParams['cmap'] = colors.LinearSegmentedColormap.from_list('', ['#14125133', '#141251', 'black'])
        
        if reeks == '1':
            plt.rc('axes', prop_cycle=cycler(color=             # MER en hieronder de 
                                        ['#141251', '#25D7F4', '#027E9B', '#94B0EA', '#AA3191', '#FF8FB2', 
                                                                # de standaardkleuren
                                         '#d62728', '#9467bd', '#8c564b', '#e377c2',
                                         '#7f7f7f', '#bcbd22', '#17becf']))
        elif reeks == '1a':
            plt.rc('axes', prop_cycle=cycler(color=             # MER en hieronder de 
                                        ['#141251', '#027E9B', '#94B0EA', '#25D7F4', '#AA3191', '#FF8FB2', 
                                                                # de standaardkleuren
                                         '#d62728', '#9467bd', '#8c564b', '#e377c2',
                                         '#7f7f7f', '#bcbd22', '#17becf']))
        elif reeks == '2':
            plt.rc('axes', prop_cycle=cycler(color=             # MER en hieronder de 
                                        ['#141251', '#D285D6', '#6552A8', '#94B0EA', '#1B60DB', '#FF8FB2',
                                                                # de standaardkleuren
                                         '#d62728', '#9467bd', '#8c564b', '#e377c2',
                                         '#7f7f7f', '#bcbd22', '#17becf']))
        elif reeks == '2a':
            plt.rc('axes', prop_cycle=cycler(color=             # MER en hieronder de 
                                        ['#141251', '#6552A8', '#94B0EA', '#D285D6', '#1B60DB', '#FF8FB2',
                                                                # de standaardkleuren
                                         '#d62728', '#9467bd', '#8c564b', '#e377c2',
                                         '#7f7f7f', '#bcbd22', '#17becf']))
        elif reeks == '3':
            plt.rc('axes', prop_cycle=cycler(color=             # MER en hieronder de 
                                        ['#141251', '#94B0EA', '#1B60DB', '#25D7F4', '#027E9B', '#FF8FB2',
                                                                # de standaardkleuren
                                         '#d62728', '#9467bd', '#8c564b', '#e377c2',
                                         '#7f7f7f', '#bcbd22', '#17becf']))
        elif reeks == '3a':
            plt.rc('axes', prop_cycle=cycler(color=             # MER en hieronder de 
                                        ['#141251', '#1B60DB', '#94B0EA', '#25D7F4', '#027E9B', '#FF8FB2',
                                                                # de standaardkleuren
                                         '#d62728', '#9467bd', '#8c564b', '#e377c2',
                                         '#7f7f7f', '#bcbd22', '#17becf']))
        elif reeks == '4':
            plt.rc('axes', prop_cycle=cycler(color=             # MER en hieronder de 
                                        ['#141251', '#FF8FB2', '#AA3191', '#D285D6', '#6552A8', '#94B0EA',
                                                                # de standaardkleuren
                                         '#d62728', '#9467bd', '#8c564b', '#e377c2',
                                         '#7f7f7f', '#bcbd22', '#17becf']))

    else:          
        # fonts
        # Let op Myriad gaat niet goed bij aanmaken pdf-figuren
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif':'Myriad Pro', 'size':6})
           
        # grid
        plt.rc('axes', axisbelow=True, grid=True)
        plt.rc('grid', color='white', linewidth=0.5, linestyle='solid')
        
        # spines  en background
        plt.rc('axes', edgecolor='#757575', linewidth=0.5, facecolor='#e3e1d3')                 
        xParams['hidespines'] = []
        
        # labels
        plt.rc('axes', labelcolor='#757575', labelsize=10, labelpad=4)

        # label lines        
        xParams['labellineprop'] = {'linewidth':0.5, 'color':'#757575', 'marker':'None'}
        xParams['labellinemin'] = 1
        
        # tick marks en labels
        plt.rc('xtick', labelsize=6, color='#757575')
        plt.rc('ytick', labelsize=6, color='#757575')       
        
        # ticks
        plt.rc('xtick.major', size=0, width=0.5, pad=4)
        plt.rc('ytick.major', size=0, width=0.5, pad=4)
        plt.rc('xtick.minor', size=0, width=0.5, pad=4)
        plt.rc('ytick.minor', size=0, width=0.5, pad=4)
        
        # legend
        plt.rc('legend', markerscale=0.8, fontsize=6, frameon=False, borderaxespad=0)
        plt.rc('text', color='#757575')
        xParams['legend'] = dict(loc='lower right', bbox_to_anchor=(1, 1))
    
        # lines en marker
        plt.rc('lines', linewidth=1,
                        markersize=4, 
                        marker='o',
                        markerfacecolor='none',
                        markeredgecolor='#666666',
                        markeredgewidth=0.5)
    
        # patches, o.a. voor een barplot
        plt.rc('patch', force_edgecolor=True,
                        linewidth=0.5,
                        edgecolor = '#4d4d4d')
        
        # heatmap
        xParams['cmap'] = 'YlOrBr'
        
        # specifiek voor een lijnplot
        if plottype == 'lijnplot':
            # colors
            plt.rc('axes', prop_cycle=cycler(color=
                                        ['#e4af00', '#4a8ab7',  # MER en hieronder de 
                                                                # de standaardkleuren
                                         '#ff7f0e', '#2ca02c', '#d62728',
                                         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                         '#bcbd22', '#17becf']))
        elif plottype == 'bar':
            # colors
            plt.rc('axes', prop_cycle=cycler(color=             # MER en hieronder de 
                                        ['#da9100', '#e4af00', '#f0d373', '#fcf7e6',  
                                                                # de standaardkleuren
                                         '#ff7f0e', '#2ca02c', '#d62728',
                                         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                         '#bcbd22', '#17becf']))
    
# -----------------------------------------------------------------------------
# as-labels
# -----------------------------------------------------------------------------
def set_xlabels (labels, ax, gap=0.01, mid=None, y=None):
    '''Twee labels met lijntjes
     
       gap bepaalt hoe lang de lijntjes zijn (transAxes)
       mid (optioneel) als het midden niet exact in het midden ligt (transData)
       y (optioneel) list met ylijn en ylabel
    ''' 


    # negeer automatisch label
    ax.set_xlabel('')
    
    if y is not None:
        ylijn, ylabel = y
    else:
        # bepaal bbox van de as-labels
        fig = plt.gcf()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer)
    
        # gebruik zelfde padding als voor de tick-labels en as-labels
        bbox.y0 -= plt.rcParams['xtick.major.pad']        # lijntjes  
        bbox.y1 = bbox.y0 - plt.rcParams['axes.labelpad'] # labels
        
        # transform naar axis-coordinaten
        bbox = bbox.transformed(ax.transAxes.inverted())
        ylijn = bbox.y0
        ylabel = bbox.y1
    
    # bepaal het midden van de as
    if mid is not None:
        xlim = ax.get_xlim()
        xmid = (mid-xlim[0]) / (xlim[1]-xlim[0])
    else:
        xmid = 0.5

    # slechts één label?
    if isinstance(labels, str): 
        labels = [labels]
        xmid *= 2 # trucje voor de loop
    
    # labels centreren relatief t.o.v. het mid-punt
    for i, label in enumerate(labels):
        xlabel = (i + xmid) / 2
        ax.text(xlabel, ylabel,
                label,
                ha='center',
                va='top',
                rotation='horizontal',
                color=plt.rcParams['axes.labelcolor'],
                fontsize=plt.rcParams['axes.labelsize'],
                transform=ax.transAxes)
        # teken lijntjes
        if len(labels) >= xParams['labellinemin']:
            x1 = i * xmid
            x2 = (1 - i) * xmid + i
            line = lines.Line2D([x1+gap, x2-gap], [ylijn, ylijn],
                                clip_on=False,
                                transform=ax.transAxes,
                                **xParams['labellineprop'])
            ax.add_line(line)
    
    return ylijn, ylabel


def set_ylabels (labels, ax, gap=0.02, mid=None, x=None):
    '''Twee labels met lijntjes
     
       gap bepaalt hoe lang de lijntjes zijn (transAxes)
       mid (optioneel) als het midden niet exact in het midden ligt (transData)
       x (optioneel) list met xlijn en xlabel
    ''' 

    # negeer automatisch label
    ax.set_ylabel('')
    
    if x is not None:
        xlijn, xlabel = x
    else:
        # bepaal bbox van de as-labels
        fig = plt.gcf()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer)
    
        # gebruik zelfde padding als voor de tick-labels en as-labels
        bbox.x0 -= plt.rcParams['xtick.major.pad']        # lijntjes  
        bbox.x1 = bbox.x0 - plt.rcParams['axes.labelpad'] # labels
        
        # transform naar axis-coordinaten
        bbox = bbox.transformed(ax.transAxes.inverted())
        xlijn = bbox.x0
        xlabel = bbox.x1
    
    # bepaal het midden van de as
    if mid is not None:
        ylim = ax.get_ylim()
        ymid = (mid-ylim[0]) / (ylim[1]-ylim[0])
    else:
        ymid = 0.5

    # slechts één label?
    if isinstance(labels, str): 
        labels = [labels]
        ymid *= 2 # trucje voor de loop
    
    # labels centreren relatief t.o.v. het mid-punt
    for i, label in enumerate(labels):
        ylabel = (i + ymid) / 2
        ax.text(xlabel, ylabel,
                label,
                ha='right',
                va='center',
                rotation='vertical',
                color=plt.rcParams['axes.labelcolor'],
                fontsize=plt.rcParams['axes.labelsize'],
                transform=ax.transAxes)
        # teken lijntjes
        if len(labels) >= xParams['labellinemin']:
            y1 = i * ymid
            y2 = (1 - i) * ymid + i
            line = lines.Line2D([xlijn, xlijn], [y1+gap, y2-gap],
                                clip_on=False,
                                transform=ax.transAxes,
                                **xParams['labellineprop'])
            ax.add_line(line)

    return xlijn, xlabel

# -----------------------------------------------------------------------------
# Baansimulaties
# -----------------------------------------------------------------------------
def plot_baansimulaties(inpFile,
                        inpFileDict = {},
                        x='dagvolume',
                        y='D4', 
                        fname='',
                        xlabel='verkeersbewegingen per dag',
                        ylabel='gebruik vierde baan',
                        xlim=[900, 1600],
                        ylim=[0,200],
                        xstep=None,
                        ystep=20,
                        bin=2,
                        histogram=True, # subplot met histogram
                        histxlabel='histogram',
                        histxlim=[0, 25],
                        histxstep=5,
                        histbin=10,
                        style='MER2019',
                        dpi=600):
    '''Plot gebruik van tweede en vierde baan'''

    # algemene plotstyle voor de mer
    plot_style(style)

    # inlezen data
    df = read_file(inpFile, **inpFileDict)
        
    # init plot
    if histogram:
        fig = plt.figure()
#        fig.subplots_adjust(wspace=0.1)
        fig.subplots_adjust(wspace=0.2)
        gs = GridSpec(1, 5, figure=fig)
        ax1 = fig.add_subplot(gs[0, :-1])
    else:
        fig, ax1 = plt.subplots()
    
    fig.set_size_inches(21/2.54, 7/2.54)

    # margins
    plt.subplots_adjust(bottom=0.2)
            
    # linker plot (2D histogram)
    xmin, xmax = xlim
    xbins = np.arange(xmin, xmax+bin, bin)
    ymin, ymax = ylim
    ybins = np.arange(ymin, ymax+bin, bin)
    
    p, _, _ = np.histogram2d(df[x], df[y], bins=[xbins, ybins])
    p = np.ma.masked_equal(p, 0) # maskeer nul-waarden
    p = p.T                      # x- en y-as zijn verwisseld
    
    ax1.imshow(p, 
               interpolation='nearest', 
               origin='low',
               extent=[xmin, xmax, ymin, ymax],
               cmap= xParams['cmap'], #'Blues',  #'YlOrBr',
               norm=colors.LogNorm(vmin=1, vmax=p.max()), # 0,2 om witte datapunten te voorkomen
               aspect='auto',
               zorder=4)
    
    # X-as
    yloc = set_xlabels(xlabel, ax=ax1)
    if xstep is not None:    
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
    
    # Y-as
    set_ylabels(ylabel, ax=ax1)
    ax1.set_ylim(0, ymax)
    if ystep is not None:    
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(ystep))
    
    # histogram
    if histogram:
        ax2 = fig.add_subplot(gs[0, -1], sharey=ax1)
        ybins = np.arange(ymin, ymax+histbin, histbin)
        df[y].hist(bins=ybins, 
                   weights=np.ones_like(df[y]) * 100. / len(df),
                   #color='#e4af00',
                   #linewidth=0.25,
                   #edgecolor='#4d4d4d',
                   rwidth=0.7,
                   orientation='horizontal', 
                   ax=ax2)
        
        # X-as
        set_xlabels(histxlabel, ax=ax2, y=yloc)
        ax2.set_xlim(histxlim)
#        ax2.tick_params(labelbottom=False, labeltop=True)
        
        if histxstep is not None:    
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(histxstep))
        ax2.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        
        # Y-as
        ax2.tick_params(labelleft=False)
                  
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    fig.savefig(fname, dpi=dpi)


#------------------------------------------------------------------------------
# Plot kansverdeling
#------------------------------------------------------------------------------
def plot_kansverdeling(inpFile,
                      inpFileDict = {},
                      y='p80',
                      bins=None,
                      simulaties=100000,
                      fname='',
                      xlabel='aantal dagen',
                      xlim=None,
                      ylabel='kans',
                      ylim=None,
                      ystep=None,
                      style='MER2019',
                      dpi=600):
    '''Maak een kansverdeling op basis van de kansvector'''

    # algemene plotstyle voor de mer
    plot_style(style, plottype='bar')

    # inlezen data
    print('inlezen data')
    df = read_file(inpFile, **inpFileDict)
    p = df[y].values
    print('en gaan')
    
    # n simulaties
    n = np.zeros(simulaties)
    for i in range(simulaties):
        normaldist = np.random.rand(p.size)
        n[i] = np.sum((normaldist-p) < 0)

    # init plot
    fig, ax = plt.subplots()
    fig.set_size_inches(21/2.54, 7/2.54)

    # margins
    plt.subplots_adjust(bottom=0.2)
    
    # plot  
    plt.hist(n,
             bins=bins,
             weights=np.ones_like(n) * 100. / simulaties,
             rwidth=0.8)

    # X-as
    if xlim is not None:
        ax.set_xlim(bins[0], bins[-1])
    else:
        ax.set_xlim(bins[0], bins[-1])
    set_xlabels(xlabel, ax=ax)

    # Y-as
    ax.set_ylim(ylim)
    if ystep is not None:    
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    set_ylabels(ylabel, ax=ax)
    
    # hide spines
    for side in xParams['hidespines']:
        ax.spines[side].set_color('none')
    
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax
    
    
    
# -----------------------------------------------------------------------------
# Concentraties (Luchtkwaliteit)
# -----------------------------------------------------------------------------
def plot_concentraties(inpFile,
                       inpFileDict = {},
                       y = ['GCN', 'Wegverkeer', 'Grondgebonden', 'Vliegtuigbijdrage'],
                       labels = ['achtergrond', 'wegverkeer', 'grondgebonden', 'vliegtuigbijdrage'],
                       stof='PM10',
                       ylabel=r'PM$_{10}$ in $\mu$g/m$^3$',
                       xlabels=['situatie 2015', 'situatie 2020'],
                       xticklabels=None,
                       ylim=[0,25],
                       ncol=2,
                       style='MER2019',
                       fname='',
                       dpi=600):
    '''Plot concentraties'''

    # algemene plotstyle voor de mer
    plot_style(style, plottype='bar')

    # lees data in een dataframe
    df = read_file(inpFile, sheet_name=stof, **inpFileDict)

    # rename columns
    if labels is not None:
        df = df.rename(columns=dict(zip(y, labels)))
        y = labels
        
    # plot  
    ax = df.plot.bar(y=y,
                     stacked=True,
                     figsize=(21/2.54, 7/2.54), # figsize is in inches        
                     width=0.2,
                     ylim=ylim)

    # margins
    plt.subplots_adjust(bottom=0.2)
        
    # gridlines
    ax.xaxis.grid(which='major', color='None')
        
    # assen
    ax.axes.tick_params(axis='both', which='both', labelrotation=0)    

    # X-as
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels(df['zichtjaar'].map(str) + '\n' + df['stelsel'])
    set_xlabels(xlabels, ax=ax)    
    
    # Y-as
    set_ylabels(ylabel, ax=ax) 
    
    # hide spines
    for side in xParams['hidespines']:
        ax.spines[side].set_color('none')

    # legend
    if ncol is None: ncol = len(y)
    leg = ax.legend(ncol=ncol,
                    handletextpad=-0.5,
                    **xParams['legend'])
    
    for patch in leg.get_patches():  # Maak de patches vierkant
        patch.set_height(5)
        patch.set_width(5)
        patch.set_y(-1)              # Vertikaal uitlijnen      
        
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax
    

# -----------------------------------------------------------------------------
# Groepsrisico (externe veiligheid)
# -----------------------------------------------------------------------------
def plot_groepsrisico(inpFile,
                      inpFileDict = {},
                      x='Group',
                      y=['HS_450', 'NS_500'],
                      labels=['referentiesituatie', 'voorgenomen activiteit'],
                      xlabel='groepsgrootte',
                      ylabel='groepsrisico',
                      ylim=[0,25],
                      ncol=None,
                      style='MER2019',
                      fname='',
                      dpi=600):
    '''Plot groepsrisico'''


    # algemene plotstyle voor de mer
    plot_style(style)

    # converteer naar list
    if isinstance(y, str): y =[y]
    if isinstance(labels, str): labels =[labels]
    
    # lees data in een dataframe
    cols = [x] + y
    df = read_file(inpFile, **inpFileDict)[cols]

    # rename columns
    if labels is not None:
        df = df.rename(columns=dict(zip(y, labels)))
        y = labels

    # plot  
    # TODO colors opslaan in MERplot
    # colors = ['#3a96b2', '#da9100'] 
    ax = df.plot(x=x,
                 y=y,
                 figsize=(21/2.54, 7/2.54), # figsize is in inches
                 logx=True,
                 logy=True,
                 xlim=[10,1000],
                 ylim=[10**-8, 10**-3],
                 marker='None') #,
#                 clip_on=False,
                 #color=colors)
    
    # margins
    plt.subplots_adjust(bottom=0.2)
    
    # X-as
    set_xlabels(xlabel, ax=ax)
    ax.xaxis.set_tick_params(which='minor', labelsize=4, pad=5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    
    # toon ook minor gridlines
    ax.xaxis.grid(which='minor')
    
    # Y-as
    set_ylabels(ylabel, ax=ax)
    
    # hide spines
    for side in xParams['hidespines']:
        ax.spines[side].set_color('none')
    
    # legend
    if ncol is None: ncol = len(y)
    ax.legend(ncol=ncol, **xParams['legend'])
             
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax


# -----------------------------------------------------------------------------
# History-file met ontwikkeling van verkeersvolume en GWC
# -----------------------------------------------------------------------------
def plot_verkeer(inpFile,
                 inpFileDict = {'sheet_name': 'realisatie'},
                 x='jaar',
                 y='verkeer', 
                 labels=None,
                 xlabel=None,
                 ylabel=None,
                 fname='',
                 xstep=1,
                 ystep=None,
                 clip_on=False,
                 ncol=None,
                 style='MER2019',
                 dpi=600,
                 **kwargs):
    '''Plot ontwikkeling van verkeersvolume'''

    def NumberFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:,.0f}'.format(x).replace(',', '.')

    # algemene plotstyle voor de mer
    plot_style(style)
 
    # converteer naar list
    if isinstance(y, str): y =[y]
    if isinstance(labels, str): labels =[labels]
    
    # lees data in een dataframe
    cols = [x] + y
    df = read_file(inpFile, **inpFileDict)[cols]

    # rename columns
    if labels is not None:
        df = df.rename(columns=dict(zip(y, labels)))
        y = labels
        
    # plot
    ax = df.plot(x=x,
                 y=y,
                 figsize=(21/2.54, 7/2.54), # figsize is in inches
                 clip_on=clip_on,
                 **kwargs)
            
    # margins
    plt.subplots_adjust(bottom=0.2)
    
    # X-as
    if xlabel is not None:
        set_xlabels(xlabel, ax=ax)
    else:
        ax.set_xlabel('') # verberg as-label
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    if xstep is not None:    
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
    
    # Y-as
    if xlabel is not None:
        set_ylabels(ylabel, ax=ax)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(NumberFormatter))
    if ystep is not None:    
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep))

    # legend
    if ncol is None: ncol = len(y)
    ax.legend(ncol=ncol, **xParams['legend'])
    
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax


# -----------------------------------------------------------------------------
# Verkeersverdeling
# -----------------------------------------------------------------------------
def plot_verkeersverdeling(trafficFile,
                           trafficFileDict = {'usecols': ['d_lt', 'd_schedule', 'd_date', 'total']},
                           bracketFile = None,
                           bracketFileDict = {},
                           capacityFile  = None,
                           capacityFileDict = {},
                           capFactor = 1,
                           percentiel = None,
                           taxitime = {'L':0, 'T':0}, 
                           fname=None,
                           ylim=[-30, 30],
                           reftraffic=1,
                           style='MER2019',
                           dpi=600):
    '''Plot verkeersverdeling'''

    def AbsFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:1.0f}'.format(abs(x))
    
    def BlokuurFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:d}:00'.format(x//3)

    # algemene plotstyle voor de mer
    plot_style(style)

    # init plot
    fig, ax = plt.subplots()
    fig.set_size_inches(21/2.54, 9/2.54)

    # SLOND-colors
    colorTab = {'S': '#cdbbce',
                'L': '#cdbbce',
                'O': '#8fbbd6',
                'N': '#d1e6bd',
                'D': '#f1b7b1',
                'DS': '#f1b7b1',
                'DL': '#f1b7b1'}
                
    # plot bracketlist/periodstable (optioneel)
    if bracketFile is not None:
        bracketList = read_file(bracketFile, **bracketFileDict)
    
        if capacityFile is not None:
            capacity = read_file(capacityFile, **capacityFileDict)
    
            # merge capacity met bracketlist
            bracketList['period'] = bracketList['period'].str.split(',', 1).str[0]
            bracketList = bracketList.merge(capacity, on='period', how='left')
    
        # arrivals onder de as
        bracketList['Lcap'] *= -1 * capFactor
        bracketList['Tcap'] *= 1 * capFactor
        
        # SLOND-color      
        if 'period' in bracketList:
            color = bracketList['period'].map(colorTab)
        else:
            color = '#cdbbce'
            
        for cap in ['Lcap', 'Tcap']:
            bracketList.plot.bar(y=cap,
                                 legend=None,
#                                 width=0.92, # barwidth
                                 width=1, # barwidth
                                 color=color,
                                 edgecolor='none',
                                 zorder=-1,
                                 ax=ax)    
        
    # read traffic/sirFile                          
    print('Read\n', trafficFile)
    df = read_file(trafficFile, **trafficFileDict)
    
    # Add keys
    if not 'total' in df.columns:
        df['total'] = 1    
    
    # baantijd
    tijd = pd.to_datetime(df['d_schedule'])
    
    arr = df['d_lt'].isin(['A', 'L'])
    dep = ~arr # df['d_lt'].isin(['D', 'T'])
    
    tijd[arr] -= pd.Timedelta(taxitime['L'], unit='minute')
    tijd[dep] += pd.Timedelta(taxitime['T'], unit='minute')
    
    hour = tijd.dt.hour
    minute = tijd.dt.minute
    
    # tijdblok
    df['tijdsblok'] = pd.Categorical(1 + hour*3 + minute//20, categories=range(1, 73))
    
    # verdeling over het etmaal - eerst per dag, daarna mean of percentiel
    PerDag = df.groupby(['d_lt', 'tijdsblok', 'd_date']).sum().fillna(0)
    
    if percentiel is not None:
        PerTijdsblok = PerDag.groupby(['d_lt', 'tijdsblok']).quantile(percentiel).reset_index()
    else:
        PerTijdsblok = PerDag.groupby(['d_lt', 'tijdsblok']).mean().reset_index()
    
    # debug: export naar Excel
    # PerTijdsblok.to_excel('test.xlsx')
    
    # dep boven en arr onder
    arr = PerTijdsblok['d_lt'].isin(['A', 'L'])
    dep = ~arr
    PerTijdsblok.loc[arr, 'total'] *= -1
    
    # plot departures / arrivals
#    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:1]
    for dest, c in zip([arr, dep], ['#fdbb4b', '#4a8ab7']):
        PerTijdsblok[dest].plot.bar(x='tijdsblok',
                                    y='total',
                                    legend=None,
                                    width=0.5,
                                    facecolor=c,
                                    edgecolor='#757575',
                                    lw=0.25,
                                    ax=ax)
    
    # gridlines
#    ax.xaxis.grid(which='minor')
#    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5,72,1)))
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(2.5,72,3)))

    # X-as
    ax.xaxis.set_tick_params(labelrotation=0)
    ax.set_xlabel('')             # verberg as-label 
    ax.xaxis.set_ticklabels([])   # verberg major tick labels

    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(1,71,3)))  
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(BlokuurFormatter))
    
    # maak een tweede x-as voor de lijntjes
    # https://stackoverflow.com/questions/45404686/how-to-add-third-level-of-ticks-in-python-matplotlib 
#    ax2 = ax.twiny()
#    ax2.set_xlim(ax.get_xlim())
#    ax2.spines["bottom"].set_position(('outward', 3))
#    # alleen tickmarks, dus verberg de assen zelf 
#    for side in ax2.spines:
#        ax2.spines[side].set_color('none')
#        
#    ax2.xaxis.set_major_locator(ticker.FixedLocator(np.arange(-0.5,72.5,3)))
#    ax2.xaxis.set_ticks_position("bottom")
#    ax2.xaxis.set_tick_params(which='major', length=7, grid_color='none') 
#    ax2.xaxis.set_ticklabels([])
     
    # Y-as
    ax.set_ylim(*ylim)        
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(AbsFormatter))

    # labels met lijntjes
    set_xlabels('tijd', ax=ax)
    set_ylabels(['landingen', 'starts'], mid=0, ax=ax)
    
    # legend - capaciteit
    if bracketFile is not None:       
        w = 1/72
        x = 0.72
        ys = [1.0492, 1.0353, 1.0492, 1.0561, 1.0457]
        heights = [4*w, 4*w, 3*w, 2*w, 3.5*w]
        for s, y, h in zip(colorTab, ys, heights):
            if 'period' in bracketList:
                c = colorTab[s]
                ax.text(x+w/2, 1.07,
                    s,
                    fontsize=4,
                    ha='center',
                    va='center',
                    transform=ax.transAxes)
            else:
                c = '#cdbbce'
                
            rect = Rectangle(xy=(x, y), 
                             width=w,
                             height=h,
                             facecolor=c,
                             edgecolor='white',
                             linewidth=0.5,
                             clip_on=False,
                             transform=ax.transAxes)
            ax.add_patch(rect)
            x += w
        
        ax.text(x+w/2, 1.07,
                'baancapaciteit',
                fontsize=plt.rcParams['legend.fontsize'],
                ha='left',
                va='center',
                transform=ax.transAxes)
    
    # legend - verkeer
    w = .5/72
    y = 1.07
    h1, h2 = [6*w, 2*w, 3*w], [-2*w, -4*w, -7*w]
    colors = ['#4a8ab7', '#fdbb4b']
    for c, heights in zip(colors, [h1, h2]):
        x = 0.92
        for h in heights:
            rect = Rectangle(xy=(x, y), 
                             width=w,
                             height=h,
                             facecolor=c,
                             edgecolor='#757575',
                             linewidth=0.25,
                             clip_on=False,
                             transform=ax.transAxes)
            ax.add_patch(rect)    
            x += w
    
    ax.text(x+w/2, 1.07,
            'verkeer',
            fontsize=plt.rcParams['legend.fontsize'],
            ha='left',
            va='center',
            transform=ax.transAxes)
        
    # save figure
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax
    
# -----------------------------------------------------------------------------        
# Plot vlootmix
# -----------------------------------------------------------------------------    
def plot_vlootmix(inpFile,
                 inpFileDict = {},
                 x='d_mtow',
                 y='total',
                 xlabel = 'maximum startgewicht (ton)',
                 ylabel = 'aandeel in het jaarvolume',
                 labels=None,
                 fname='',
                 bins=None,
                 nbins=30,
                 widths=None,
                 ylim=[0,50],
                 ncol=None,
                 style='MER2019',
                 dpi=600,
                 **kwargs):
    '''Plot vlootmix'''

    # converteer naar list
    if isinstance(inpFile, str): inpFile =[inpFile]
    if isinstance(labels, str): labels =[labels]

    # algemene plotstyle voor de mer
    plot_style(style, plottype='bar')
        
    # init plot
    fig, ax = plt.subplots()
    fig.set_size_inches(21/2.54, 7/2.54)
    zorder = len(inpFile)

    # auto bins, nbins logaritmisch tussen 10 en 600 
    if bins is None:
        # afronden naar int strikt genomen niet noodzakelijk
        bins = np.rint(np.logspace(np.log10(10),np.log10(600), nbins)).astype(int)

    # auto widths
    if widths is None:
        widths = [0.7**(len(inpFile)-i-1) for i in range(len(inpFile))]
    
    # ter info: gemiddeld MTOW
    print('situatie         jaarvolume  avg.MTOW')
    print('-------------------------------------')
    
    for inp, label, width in zip(inpFile, labels, widths):
    
        df = read_file(inp)      
        print('{:20s} {:6.0f}  {:8.0f}'.format(label, df[y].sum(), (df[x]*df[y]).sum()/df[y].sum()))

        df[x].hist(label=label,
                   bins=bins,
                   weights=df[y]*100/df[y].sum(), # percentage van het totaal
                   rwidth=width,
                   zorder=zorder,
                   ax=ax,
                   **kwargs)
        zorder += -1 # reverse plot order
        
        # add bin en save als xls
        df['bin'] = pd.cut(df[x], bins)
        out = splitext(inp)[0] + f'_{nbins}bins.xls'
        df.to_excel(out)
        
    print('-------------------------------------\n')
        
    # margins
    plt.subplots_adjust(bottom=0.2)
    
    # X-as
    ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    
    set_xlabels(xlabel, ax=ax)
    
    # toon ook minor gridlines
    ax.xaxis.grid(which='minor')
        
    # Y-as
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    
    set_ylabels(ylabel, ax=ax)
    
    # legend
    if ncol is None: ncol = len(inpFile)
    leg = ax.legend(ncol=ncol, 
                    handletextpad=-0.5,
                    **xParams['legend'])
    
    for patch in leg.get_patches():  # Maak de patches vierkant
        patch.set_height(5)
        patch.set_width(5)
        patch.set_y(-1)              # Vertikaal uitlijnen      
             
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax


# -----------------------------------------------------------------------------
# Geluidbelasting in handhavingspunten
# -----------------------------------------------------------------------------
def hs_to_db(hs, t):
    '''Converteer hindersommen naar dB'''
    return 10*np.log10(hs) - 10*np.log10(365*t*60*60)
    
def plot_hhp(eppmy,
             eppmyDict = {'delimiter':' ', 'header':None, 'index_col':0},
             mtg=None,
             mtgDict = {},
             deltaplot=False,
             xlabel=None,
             ylabel=None,
             xlim=None,
             ylim1=[45,65],
             ylim2=[45,65],
             xstep=1,
             ystep1=5,
             ystep2=5,
             fname='',
             clip_on=False,
             style='MER2019',
             alpha=0.2,
             dpi=600,
             **kwargs):
    '''Plot geluidbelasting in handhavingspunten. 
       bij een deltaplot wordt de eerste eppmy geplot en
       wordt de tweede gebruikt voor het berekenen van het verschil.'''

    def dbaFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:1.0f} dB(A)'.format(x)

    # converteer naar list
    if isinstance(eppmy, str): eppmy =[eppmy]
    ylim = [ylim1, ylim2]
    ystep = [ystep1, ystep2]

    # algemene plotstyle voor de mer
    plot_style(style)
    
    # init plot
    fig = plt.figure()
    fig.set_size_inches(21/2.54, 14/2.54)

    # sub plots
    gs = GridSpec(ncols=1, nrows=2,
                  hspace=0.1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    

    # lees MTG in een dataframe
    gw = read_file(mtg, **mtgDict)
    gw.index += 1 # index vanaf 1 ipv 0

    df = []
    for i, ax in enumerate([ax1, ax2]):     
        # lees data in een dataframe
        df.append(read_file(eppmy[i], **eppmyDict))
    
        # hindersom naar dB
        nhhp = len(df[i].index)
        if nhhp == 35:
            t = 24
        else:
            t = 8 
        df[i] = df[i].apply(hs_to_db, t=t)
                
        # verschilplot
        if i==1 and deltaplot:
            df[i] = df[0] - df[i]
        
        # 0-as
        ax.axhline(marker='None', lw=plt.rcParams['axes.linewidth'])
        
        # plot
        df[i].plot(style='o',
                  alpha=alpha,
                  legend=False,
                  clip_on=clip_on,
                  ax=ax,
                  **kwargs)
    
        if i==0 or not deltaplot:
            gw.plot(style='_',
                    markeredgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
                    markersize=8,
                    markeredgewidth=1.5,
                    legend=False,
                    clip_on=clip_on,
                    ax=ax,
                    **kwargs)
    
        # X-as
        ax.set_xlim(xlim)
        if xstep is not None:    
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
        if i==1 and xlabel is not None:
            set_xlabels(xlabel, ax=ax)
        else:
            ax.set_xlabel('') # verberg as-label
        
        # Y-as
        ax.set_ylim(ylim[i])
        if ystep is not None:    
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep[i]))
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(dbaFormatter))

        if ylabel is not None:
            set_ylabels(ylabel[i], ax=ax)

    # legend
    ax0 = fig.add_axes([0.72, 0.89, 0.1, 0.1]) 
    
    # geen assen
    ax0.axis('off')
    
    # genormaliseerde asses
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    # reeks 1: dots
    np.random.seed(0)
    y = np.random.normal(0.2,0.1,15)
    x = [0] * len(y)    
    ax0.plot(x, y, 'o',
             markersize=plt.rcParams['lines.markersize']*0.8,
             alpha=alpha,
             clip_on=False)
    ax0.text(0.09, 0.2, 'geluidbelasting',
             transform=ax0.transAxes,
             horizontalalignment='left')

    # reeks 2: dash
    ax0.plot(1, 0.25, '_',
             markeredgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
             markersize=8*0.8,
             markeredgewidth=1.5*0.8,
             clip_on=False)
    ax0.text(1.12, 0.2, 'grenswaarde',
             transform=ax0.transAxes,
             horizontalalignment='left')
    
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax1, ax2


# -----------------------------------------------------------------------------
# Baangebruik, kopie van doc29lib aangepast voor MER2020-opmaak
# -----------------------------------------------------------------------------
def plot_baangebruik(trf_files,
                     labels,
                     den=['D', 'E', 'N'],
                     fname=None,
                     n=7,
                     runways=None,
                     ylabel='vliegtuigbewegingen',
                     ylim=[0,110000],
                     dy=10000,
                     reftraffic=1,
                     style='MER2020',
                     dpi=300):
    '''Plot het baangebruik'''

    def NumberFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:,.0f}'.format(x).replace(',', '.')
    def GetVal(var, i):
        'Scalar of een list'
        if isinstance(var, list):
            i = min(i, len(var)-1) # hergebruik van de laatste waarde 
            return var[i]
        else:
            return var

    # kopie van plotformat
    MarkerWidth =  [0.3, 0.3, 0.08]  # voor 1, 2 en >2 traffics
    MarkerHeight = [0.1, 0.1, 0.08]
    BarWidth = [0.1, 0.08, 0.04]
    BarGap = [0, 0, 0.05]

    # algemene plotstyle voor de mer
    plot_style(style)
    
    # converteer naar list
    if isinstance(trf_files, str): trf_files =[trf_files]
            
    # X-positie van de bars
    x = np.arange(n)

    ntrf = len(trf_files)
    i = ntrf - 1
    w = (GetVal(BarWidth, i) * n/7) # normaliseer voor de aslengte
    g = GetVal(BarGap, i)           # of /ntrf?
    
    dx = [(w+g)*(i - 0.5*(ntrf-1)) for i in range(ntrf)]
    
    # markers en staafjes
    marker_height = (GetVal(MarkerHeight, i) * (ylim[-1] - ylim[0]) / 10) 
    mw = (GetVal(MarkerWidth, i) * n/7)
    dxm = list(dx)
    
    # clip marker
    if ntrf == 2:
        mw = (mw + w)/2
        dxm[0] = dx[0] - (mw-w)/2
        dxm[1] = dx[1] + (mw-w)/2
    elif ntrf > 2:
        mw = [min(mw, w+g)]
    
    # twee aansluitende subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(21/2.54, 10/2.54)
    
    # margins
    fig.subplots_adjust(bottom=0.18)    
    fig.subplots_adjust(wspace=0.02)
    
    # legenda
    ax0 = fig.add_axes([0.79, 0.89, 0.05, 0.1]) 
    
    # geen assen
    ax0.axis('off')
    
    # genormaliseerde asses
    ax0.set_xlim(-0.5*n/7, 0.5*n/7)
    ax0.set_ylim(0, 1)
    
    # staafjes
    if ntrf == 2:
        #TODO: 1 of >2 staafjes
        # gemiddelde
        for i, yi, bottom, xt, yt, alignment in [(0, 0.4, 0.1, 0.2, 0.3, 'right'),
                                                 (1, 0.5, 0.3, 0.8, 0.3, 'left')]:
            if i == reftraffic:
                c = 1
            else:
                c = 0
            ax0.bar(dx[i], height=0.6, bottom=bottom,
                    width=w,
                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][c])
            ax0.bar(dxm[i], height=0.05, bottom=yi,
                    width=mw,
                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][c])
            ax0.text(xt, yt, labels[i],
                     transform=ax0.transAxes,
                     horizontalalignment=alignment)
    
    
    # verwerken traffics
    for i, trf_file in enumerate(trf_files):
        
        # lees csv
        trf = pd.read_csv(trf_file, delimiter='\t')
        trf = trf.loc[trf['d_den'].isin(den)]
        
        # aggregeer etmaalperiode en bereken stats
        trf = trf.groupby(['d_lt', 'd_runway', 'd_myear'])['total'].sum().reset_index()
        trf_stats = trf.groupby(['d_lt', 'd_runway'])['total'].agg(['min','max','mean']).reset_index()
        
        # sorteer
        if 'key' not in trf_stats.columns:
            trf_stats['key'] = trf_stats['d_lt'] + trf_stats['d_runway']

        if runways is not None:
            # tweede traffic in dezelfde volgorde
            keys = [k + r for k in runways for r in runways[k]]    # keys: combinatie van lt en runway
            sorterIndex = dict(zip(keys, range(len(keys))))        # plak een volgnummer aan de keys     
            trf_stats['order'] = trf_stats['key'].map(sorterIndex) # soteerindex toevoegen
            trf_stats = trf_stats.sort_values(by=['order'])        # sorteer dataframe
        else:
            trf_stats = trf_stats.sort_values(by=['d_lt', 'mean'], ascending=False)
            runways = {'L': trf_stats['d_runway'].loc[trf_stats['d_lt'] == 'L'],
                       'T': trf_stats['d_runway'].loc[trf_stats['d_lt'] == 'T']}
        
        # maak de plot
        for lt, ax in zip(['T', 'L'], [ax1, ax2]):
            
            # selecteer L of T
            trf2 = trf_stats.loc[trf_stats['d_lt'] == lt]
            trf2 = trf2.head(n) # gaat alleen goed als er ook echt n-runways zijn
            
            # staafjes
            bar_height = trf2['max'] - trf2['min']
            if i == reftraffic:
                c = 1
                # ref = 'ref'
            else:
                c = 0
                # ref = ''
                
            ax.bar(x+dx[i], 
                   height=bar_height.values,
                   bottom=trf2['min'].values,
                   width=w,
                   color=plt.rcParams['axes.prop_cycle'].by_key()['color'][c])
            
            # gemiddelde
            ax.bar(x+dxm[i],
                    height=marker_height,
                    bottom=trf2['mean'].values-marker_height/2,
                    width=mw,
                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][c])

    for xlabel, spine, ax in zip(['starts', 'landingen'],
                                 ['right', 'left'],
                                 [ax1, ax2]):          
        # geen scheidingslijntje tussen subplots
        # ax.spines[spine].set_color('none')
                 
        # geen vertikale gridlines
        ax.grid(which='major', axis='x', b=False)            
                                
        # X-as
        ax.set_xticks(x)
        ax.set_xticklabels(trf2['d_runway'])
        set_xlabels(xlabel, gap=0.01, ax=ax)
        
        # Y-as
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=dy))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(NumberFormatter))
        if ax==ax1:
            set_ylabels(ylabel, ax=ax)
            
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()