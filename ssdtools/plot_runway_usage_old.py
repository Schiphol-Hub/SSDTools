#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:55:58 2020

Oude versie van de functie
"""

def plot_runway_usage(traffic, labels, den=('D', 'E', 'N'), n=7, runways=None, ylim=(0, 110000), dy=10000, reftraffic=1,
                      style='MER'):
    """
    Plot the runway usage.

    :param list|tuple runways: the runways to include.
    :param list(TrafficAggregate)|TrafficAggregate traffic: the traffic to plot the runway usage for.
    :param list(str) labels: the list of labels to use for the provided traffics.
    :param list|tuple den: the periods of the day to include (D: day, E: evening, N: night).
    """

    def NumberFormatter(x, pos):
        'The two args are the value and tick position'
        return '{:,.0f}'.format(x).replace(',', '.')

    def GetVal(var, i):
        'Scalar of een list'
        if isinstance(var, list):
            i = min(i, len(var) - 1)  # hergebruik van de laatste waarde
            return var[i]
        else:
            return var

    # Check if multiple traffics are provided
    if not isinstance(traffic, list):
        traffic = [traffic]

    matplotlib.rcParams['font.size'] = 12 ##############################


    # Get the X-positions of the bars
    x = np.arange(n)

    ntrf = len(traffic)
    i = ntrf - 1
    w = (GetVal(branding.baangebruik[style]['barwidth'], i)  # of /ntrf
         * n / 7)  # normaliseer voor de aslengte
    g = GetVal(branding.baangebruik[style]['bargap'], i)  # of /ntrf?

    dx = [(w + g) * (i - 0.5 * (ntrf - 1)) for i in range(ntrf)]

    # markers and bars
    marker_height = (GetVal(branding.baangebruik[style]['markerheight'], i)
                     * (ylim[-1] - ylim[0]) / 10)
    mw = (GetVal(branding.baangebruik[style]['markerwidth'], i)
          * n / 7)
    dxm = list(dx)

    # clip marker
    if ntrf == 2:
        mw = (mw + w) / 2
        dxm[0] = dx[0] - (mw - w) / 2
        dxm[1] = dx[1] + (mw - w) / 2
    elif ntrf > 2:
        mw = [min(mw, w + g)]

    # Two subplots without gutter between the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(21 / 2.54, 10 / 2.54)

    # margins
    fig.subplots_adjust(bottom=0.18)
    fig.subplots_adjust(wspace=0)
   
    # Set the colors for each column. These are the current house-colours 
    colors_HS = {
            'a': '#141251',
            'b': '#1B60DB',
            'c': '#9491AA',
            'd': '#027E9B'}

    # Legend
    ax0 = fig.add_axes([0.8, 0.9, 0.05, 0.1])

    # No axis
    ax0.axis('off')

    # Normalize the axes
    ax0.set_xlim(-0.5 * n / 7, 0.5 * n / 7)
    ax0.set_ylim(0, 1)

    # Bars
    if ntrf == 2:
        # TODO: 1 of >2 staafjes
        # gemiddelde
        for i, yi, bottom, xt, yt, alignment in [(0, 0.4, 0.1, 0.2, 0.4, 'right'),
                                                 (1, 0.5, 0.3, 0.8, 0.4, 'left')]:
            if i == reftraffic:
                ref = 'ref'
            else:
                ref = ''
            ax0.bar(dx[i], height=0.6, bottom=bottom,
                    width=w,color=colors_HS['b'], #
                    **branding.baangebruik[style][ref + 'bar'],##########
                    zorder=4)
            ax0.bar(dxm[i], height=0.05, bottom=yi,
                    width=mw,color=colors_HS['a'], #
                    **branding.baangebruik[style][ref + 'marker'],##############
                    zorder=6)
            ax0.text(xt, yt, labels[i],
                     transform=ax0.transAxes,
                     horizontalalignment=alignment,
                     #**branding.baangebruik[style]['legendtext']
                     )

    # Process the provided traffics
    for i, trf_file in enumerate(traffic):

        # Get the runway statistics
        trf_stats = trf_file.get_runway_usage_statistics('|'.join(den)).reset_index()

        # sorteer
        if 'key' not in trf_stats.columns:
            trf_stats['key'] = trf_stats['d_lt'] + trf_stats['d_runway']

        if runways is not None:
            # tweede traffic in dezelfde volgorde
            keys = [k + r for k in runways for r in runways[k]]  # keys: combinatie van lt en runway
            sorterIndex = dict(zip(keys, range(len(keys))))  # plak een volgnummer aan de keys
            trf_stats['order'] = trf_stats['key'].map(sorterIndex)  # soteerindex toevoegen
            trf_stats = trf_stats.sort_values(by=['order'])  # sorteer dataframe
        else:
            trf_stats = trf_stats.sort_values(by=['d_lt', 'mean'], ascending=False)
            runways = {'L': trf_stats['d_runway'].loc[trf_stats['d_lt'] == 'L'],
                       'T': trf_stats['d_runway'].loc[trf_stats['d_lt'] == 'T']}

        # maak de plot
        for lt, xlabel, fc, spine, ax in zip(['T', 'L'],
                                             ['starts', 'landingen'],
                                             branding.baangebruik[style]['facecolor'],
                                             ['right', 'left'],
                                             [ax1, ax2]):

            # selecteer L of T
            trf2 = trf_stats.loc[trf_stats['d_lt'] == lt]
            trf2 = trf2.head(n)  # gaat alleen goed als er ook echt n-runways zijn

            # staafjes
            bar_height = trf2['max'] - trf2['min']
            if i == reftraffic:
                ref = 'ref'
            else:
                ref = ''

            ax.bar(x + dx[i], height=bar_height, bottom=trf2['min'],
                   width=w,color=colors_HS['c'],
                   zorder=4)

            # gemiddelde
            ax.bar(x + dxm[i], height=marker_height, bottom=trf2['mean'] - marker_height / 2,
                   width=mw,color=colors_HS['d'],
                   zorder=4)

            # border
            plt.setp(ax.spines.values(), **branding.baangebruik[style]['spines'])

            # Removing all borders except for lower line in the plots
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Adding gridlines
            ax.grid(which='major', axis='y', linewidth=0.5, color='grey')
    

            # Tweaking line between subplots
            frame = lines.Line2D([0, 1], [0, 0],
                     transform=ax.transAxes,
                     **branding.baangebruik[style]['spines'],
                     zorder=10)
            frame.set_clip_on(False)
            ax.add_line(frame)

            # geen tickmarks
            plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='none')

            # label size and color
            ax.tick_params(axis='both'
                           #, **branding.baangebruik[style]['axislabel']
                           )

            # X-as
            ax.set_xticks(x)
            ax.set_xticklabels(trf2['d_runway'])
            ax.text(0.5, -0.18, xlabel,
                    transform=ax.transAxes,
                    **branding.baangebruik[style]['grouptext']
                    )

            # X-as lijntjes
            ax.set_xlim(-0.5, n - 0.5)
            line = lines.Line2D([0.02, 0.98], [-.11, -.11],
                                transform=ax.transAxes,
                                **branding.baangebruik[style]['grouplines'])
            line.set_clip_on(False)
            ax.add_line(line)

            # Y-as
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(ticker.FormatStrFormatter(base=dy))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(NumberFormatter))
            
            # Adding a vertical line in the middle of the plots
            frame2 = lines.Line2D([1, 1], [0, 1],
                                 transform=ax1.transAxes,
                                 **branding.baangebruik[style]['spines'],
                                 zorder=10)
            frame2.set_clip_on(False)
            ax.add_line(frame2)

    return fig, (ax1, ax2)
