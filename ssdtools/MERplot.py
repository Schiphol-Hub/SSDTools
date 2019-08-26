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
    
    # fonts
    #plt.rcParams['font.family'] = 'sans-serif'
    #plt.rcParams['font.sans-serif'] = 'Myriad Pro'
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif':'Myriad Pro', 'size':6})
    
    # grid
    plt.rc('axes', axisbelow=True, grid=True)
    plt.rc('grid', color='white', linewidth=0.5, linestyle='solid')
    
    # spines  en background
    plt.rc('axes', edgecolor='#757575', linewidth=0.5, facecolor='#e3e1d3')                 
    
    # labels
    plt.rc('axes', labelcolor='#757575', labelsize=10, labelpad=4)
    
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
    xParams = {'legend': dict(loc='lower right', bbox_to_anchor=(1, 1))}

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
        x1 = i * xmid
        x2 = (1 - i) * xmid + i
        line = lines.Line2D([x1+gap, x2-gap], [ylijn, ylijn],
                            lw=plt.rcParams['axes.linewidth'],
                            color=plt.rcParams['axes.edgecolor'],
                            marker='None',
                            clip_on=False,
                            transform=ax.transAxes)
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
        y1 = i * ymid
        y2 = (1 - i) * ymid + i
        line = lines.Line2D([xlijn, xlijn], [y1+gap, y2-gap],
                            lw=plt.rcParams['axes.linewidth'],
                            color=plt.rcParams['axes.edgecolor'],
                            marker='None',
                            clip_on=False,
                            transform=ax.transAxes)
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
        fig.subplots_adjust(wspace=0.1)
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
               cmap='YlOrBr',
               norm=colors.LogNorm(vmin=0.2, vmax=p.max()), # 0,2 om witte datapunten te voorkomen
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
                   color='#e4af00',
                   linewidth=0.25,
                   edgecolor='#4d4d4d',
                   rwidth=0.7,
                   orientation='horizontal', 
                   ax=ax2)
        
        # X-as
        set_xlabels(histxlabel, ax=ax2, y=yloc)
        ax2.set_xlim(histxlim)
        ax2.tick_params(labelbottom=False, labeltop=True)
        
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
    
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    fig.savefig(fname, dpi=dpi)
    
    
    
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
    
    # legend
    if ncol is None: ncol = len(y)
    leg = ax.legend(ncol=ncol,
                    handletextpad=-0.5,
                    **xParams['legend'])
    
    for patch in leg.get_patches():  # Maak de patches vierkant
        patch.set_height(5)
        patch.set_width(5)
        patch.set_y(-1)              # Vertikaal uitlijnen      
        
    # save fig
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    fig.savefig(fname, dpi=dpi)
    

# -----------------------------------------------------------------------------
# Groepsrisico (externe veiligheid)
# -----------------------------------------------------------------------------
def plot_groepsrisico(inpFile,
                      inpFileDict = {},
                      x='Group',
                      y=['HS_450', 'NS_500'],
                      labels=['referentiesituatie', 'voorgenomen activiteit'],
                      xlabel='Groepsgrootte',
                      ylabel='Groepsrisico',
                      ylim=[0,25],
                      ncol=None,
                      style='MER2019',
                      fname='',
                      dpi=600):
    '''Plot groepsrisico'''


    # algemene plotstyle voor de mer
    plot_style('MER2019')

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
    colors = ['#3a96b2', '#da9100'] 
    ax = df.plot(x=x,
                 y=y,
                 figsize=(21/2.54, 7/2.54), # figsize is in inches
                 logx=True,
                 logy=True,
                 xlim=[10,1000],
                 ylim=[10**-8, 10**-3],
                 marker='None',
#                 clip_on=False,
                 color=colors)
    
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
    
    # legend
    if ncol is None: ncol = len(y)
    ax.legend(ncol=ncol, **xParams['legend'])
             
    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    fig.savefig(fname, dpi=dpi)


# -----------------------------------------------------------------------------
# History-file met ontwikkeling van verkeersvolume en GWC
# -----------------------------------------------------------------------------
def plot_verkeer(inpFile,
                 inpFileDict = {'sheet_name': 'realisatie'},
                 x='jaar',
                 y='verkeer', 
                 labels=None,
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
            
    # X-as
    ax.set_xlabel('') # verberg as-label
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    if xstep is not None:    
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
    
    # Y-as
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
        plt.show()
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
                                 width=0.92, # barwidth
                                 color=color,
                                 edgecolor='none',
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
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5,72,1)))

    # X-as
    ax.xaxis.set_tick_params(labelrotation=0)
    ax.set_xlabel('')             # verberg as-label 
    ax.xaxis.set_ticklabels([])   # verberg major tick labels

    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(1,71,3)))  
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(BlokuurFormatter))
    
    # maak een tweede x-as voor de lijntjes
    # https://stackoverflow.com/questions/45404686/how-to-add-third-level-of-ticks-in-python-matplotlib 
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.spines["bottom"].set_position(('outward', 3))
    # alleen tickmarks, dus verberg de assen zelf 
    for side in ax2.spines:
        ax2.spines[side].set_color('none')
        
    ax2.xaxis.set_major_locator(ticker.FixedLocator(np.arange(-0.5,72.5,3)))
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_tick_params(which='major', length=7, grid_color='none') 
    ax2.xaxis.set_ticklabels([])
     
    # Y-as
    ax.set_ylim(*ylim)        
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(AbsFormatter))

    # labels met lijntjes
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
    fig.savefig(fname, dpi=dpi)
