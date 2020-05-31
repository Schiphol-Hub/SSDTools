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
from matplotlib import ticker
from matplotlib import colors
from matplotlib.gridspec import GridSpec

from ssdtools import branding ###TODO Is dit nodig
from ssdtools.traffic import read_file
from ssdtools.figures import get_cycler_color, update_legend_position
from ssdtools.figures import plot_bar


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
                        dpi=600):
    '''Plot gebruik van tweede en vierde baan'''


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
               cmap= branding.xParams['cmap_mono'],
               norm=colors.LogNorm(vmin=1, vmax=p.max()), # 0,2 om witte datapunten te voorkomen
               aspect='auto',
               zorder=4)

    # X-as
    yloc = branding.set_xlabels(xlabel, ax=ax1)
    if xstep is not None:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(xstep))

    # Y-as
    branding.set_ylabels(ylabel, ax=ax1)
    ax1.set_ylim(0, ymax)
    if ystep is not None:
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(ystep))

    # histogram
    if histogram:
        ax2 = fig.add_subplot(gs[0, -1], sharey=ax1)
        ybins = np.arange(ymin, ymax+histbin, histbin)
        df[y].hist(bins=ybins,
                   weights=np.ones_like(df[y]) * 100. / len(df),
                   # color=get_cycler_color(3),
                   color='#1B60DB',
                   rwidth=0.7,
                   orientation='horizontal',
                   ax=ax2)

        # X-as
        branding.set_xlabels(histxlabel, ax=ax2, y=yloc)
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
                      dpi=600):
    '''Maak een kansverdeling op basis van de kansvector'''


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
    branding.set_xlabels(xlabel, ax=ax)

    # Y-as
    ax.set_ylim(ylim)
    if ystep is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    branding.set_ylabels(ylabel, ax=ax)

    # geen verticale  gridlines
    ax.xaxis.grid(which='major', color='None')

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
                       figsize=(8.27, 2.76),
                       fname='',
                       dpi=600,
                       **kwargs):
    '''Plot concentraties'''


    # lees data in een dataframe
    df = read_file(inpFile, sheet_name=stof, **inpFileDict)

    # rename columns
    if labels is not None:
        df = df.rename(columns=dict(zip(y, labels)))
    
    # Stel index in op combinatie van stelsel en zichtjaar 
    df['scenario'] = df['stelsel'] + '\n' + df['zichtjaar'].map(str)
    df = df.set_index(keys='scenario')
    
    # plot
    return plot_bar(df,
                    y=labels,
                    stacked=True,
                    xlabel=xlabels,
                    ylabel=ylabel,
                    ylim=ylim,
                    ncol=ncol,
                    figsize=figsize,
                    dpi=dpi,
                    fname=fname,
                    **kwargs)


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
                      fname='',
                      dpi=600):
    '''Plot groepsrisico'''


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
                 logx=True,
                 logy=True,
                 xlim=[10,1000],
                 ylim=[10**-8, 10**-3],
                 marker='None')

    # margins
    plt.subplots_adjust(bottom=0.2)

    # X-as
    branding.set_xlabels(xlabel, ax=ax)
    ax.xaxis.set_tick_params(which='minor', labelsize=4, pad=5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

    # toon ook minor gridlines
    ax.xaxis.grid(which='minor')

    # Y-as
    branding.set_ylabels(ylabel, ax=ax)

    # legend
    if ncol is None: ncol = len(y)
    ax.legend(ncol=ncol, **branding.xParams['legend'])

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
                           dpi=600):
    '''Plot verkeersverdeling'''

    def AbsFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:1.0f}'.format(abs(x))

    def BlokuurFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:d}:00'.format(x//3)


    # init plot
    fig, ax = plt.subplots()
    fig.set_size_inches(21/2.54, 9/2.54)

    # SLOND-colors
    # colorTab = {'S': '#cdbbce',
    #             'L': '#cdbbce',
    #             'O': '#8fbbd6',
    #             'N': '#d1e6bd',
    #             'D': '#f1b7b1',
    #             'DS': '#f1b7b1',
    #             'DL': '#f1b7b1'}
    colorTab = {'S': '#a393bd',
                'L': '#a393bd',
                'O': '#91b5de',
                'N': '#c0c1cf',
                'D': '#f8a3a8',
                'DS': '#f8a3a8',
                'DL': '#f8a3a8'}
# ik
    colorTab = {'S': '#ffb9cf',
                'L': '#ffb9cf',
                'O': '#91b5de',
                'N': '#a093cb',
                'D': '#db83c9',
                'DS': '#db83c9',
                'DL': '#db83c9'}
# Jan blz 2, fig 1 - andere kleurvolgorde
    colorTab = {'S': '#91DCE8',
                'L': '#91DCE8',
                'O': '#E39A96',
                'N': '#F6CECC',
                'D': '#63A5BA',
                'DS': '#63A5BA',
                'DL': '#63A5BA'}
    alpha=1.0# Jan blz 2, fig 1 - nacht gewijzigd
    colorTab = {'S': '#91DCE8',
                'L': '#91DCE8',
                'O': '#E39A96',
                'N': '#D9D7E2',
                'D': '#63A5BA',
                'DS': '#63A5BA',
                'DL': '#63A5BA'}
    alpha=1
# # Jan blz 3, fig 1
#     colorTab = {'S': '#AD8CB9',
#                 'L': '#AD8CB9',
#                 'O': '#E39A96',
#                 'N': '#F6CECC',
#                 'D': '#8D076B',
#                 'DS': '#8D076B',
#                 'DL': '#8D076B'}
#     alpha=0.6   
    
    
    
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
                                 alpha=alpha,
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
    for dest, c in [[arr, 0], [dep, 0]]:  #, ['#fdbb4b', '#4a8ab7']:
        PerTijdsblok[dest].plot.bar(x='tijdsblok',
                                    y='total',
                                    legend=None,
                                    width=0.5,
                                    facecolor=get_cycler_color(c),
                                    edgecolor='none',
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

    # 0-as
    ax.axhline(marker='None', lw=plt.rcParams['axes.linewidth'])

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
    branding.set_xlabels('tijd', ax=ax)
    branding.set_ylabels(['landingen', 'starts'], mid=0, ax=ax)

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
                             alpha=alpha,
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
#    colors = ['#4a8ab7', '#fdbb4b']
    colors = [get_cycler_color(0)] * 2 # tijdelijke oplossing
    for c, heights in zip(colors, [h1, h2]):
        x = 0.92
        for h in heights:
            rect = Rectangle(xy=(x, y),
                             width=w,
                             height=h,
                             facecolor=c,
                             edgecolor='none',
#                             edgecolor='#757575',
#                             linewidth=0.25,
                             clip_on=False,
                             transform=ax.transAxes)
            ax.add_patch(rect)
            x += w*1.2

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
                 dpi=600,
                 **kwargs):
    '''Plot vlootmix'''

    # converteer naar list
    if isinstance(inpFile, str): inpFile =[inpFile]
    if isinstance(labels, str): labels =[labels]


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

    branding.set_xlabels(xlabel, ax=ax)

    # toon ook minor gridlines
    ax.xaxis.grid(which='minor')

    # Y-as
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

    branding.set_ylabels(ylabel, ax=ax)

    # legend
    if ncol is None: ncol = len(inpFile)
    leg = ax.legend(ncol=ncol,
                    handletextpad=-0.5,
                    **branding.xParams['legend'])

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
             mtgcolor = 4,
             deltaplot=False,
             xlabel='handhavingspunten',
             ylabel=None,
             xlim=None,
             ylim1=[45,65],
             ylim2=[45,65],
             xstep=1,
             ystep1=5,
             ystep2=5,
             fname='',
             clip_on=False,
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

        # drop nul-waarden
        df[i] = df[i].replace(0,np.nan).dropna(axis=0, how='all')

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

        # plot
        df[i].plot(style='o',
                  markeredgecolor=get_cycler_color(2),
                  alpha=alpha,
                  legend=False,
                  clip_on=clip_on,
                  ax=ax,
                  **kwargs)

        # grenswaarden (MTG)
        if i==0 or not deltaplot:
            gw.plot(style='_',
                    markeredgecolor=get_cycler_color(mtgcolor),
                    markersize=8,
                    markeredgewidth=1.5,
                    legend=False,
                    clip_on=clip_on,
                    ax=ax,
                    **kwargs)

        # X-as
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            z = ax.get_xlim()
            ax.set_xlim(z[0]-0.99, z[1]+0.99)
        if xstep is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
        if i==1 and xlabel is not None:
            branding.set_xlabels(xlabel, ax=ax)
        else:
            ax.set_xlabel('') # verberg as-label
        if i==0 or deltaplot:
            # 0-as
            ax.axhline(marker='None', lw=plt.rcParams['axes.linewidth'])

        # Y-as
        ax.set_ylim(ylim[i])
        if ystep is not None:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep[i]))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(dbaFormatter))

        if ylabel is not None:
            branding.set_ylabels(ylabel[i], ax=ax)

    # legenda
    ax0 = fig.add_axes([0.0, 0.89, 0.1, 0.1])

    # geen assen
    ax0.axis('off')

    # genormaliseerde asses
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    # reeks 1: dots
    np.random.seed(0)
    y = np.random.normal(0.11,0.1,15)
    x = [0] * len(y)
    ax0.plot(x, y, 'o',
             markeredgecolor=get_cycler_color(2),
             markersize=plt.rcParams['lines.markersize']*0.8,
             alpha=alpha,
             clip_on=False)
    ax0.text(0.09, 0.1, 'geluidbelasting',
             transform=ax0.transAxes,
             horizontalalignment='left')

    # reeks 2: dash
    ax0.plot(1, 0.15, '_',
             markeredgecolor=get_cycler_color(mtgcolor),
             markersize=8*0.8,
             markeredgewidth=1.5*0.8,
             clip_on=False)
    ax0.text(1.12, 0.1, 'grenswaarde',
             transform=ax0.transAxes,
             horizontalalignment='left')

    # Check uitlijning van de legend
    update_legend_position(ax0, target=0.9-0.01)

    # save figure
    fig = plt.gcf()  # alternatief fig = ax.get_figure()
    if fname:
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        return fig, ax1, ax2
