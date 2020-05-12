import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib import colors
from cycler import cycler

global xParams  # extra parameters t.o.v. rcParams
xParams = dict()

##############################################################################
###TODO contourplot en verschilplot kunnen weg

# contourplot = {'MER': {'colors': ['#8c564b', '#9467bd', '#ff7f0e', '#1f77b4', '#17becf'],
#                        'contour': {'linewidths': [0.35, 0.5]},
#                        'annotate': {'bbox': {'boxstyle': 'round4,pad=0.3',
#                                              'facecolor': (1.0, 1.0, 1.0, 0.4),
#                                              'edgecolor': (0, 0.45, 0.45, 1.0),
#                                              'linewidth': 0.5},
#                                     'arrowprops': {'arrowstyle': '-',
#                                                    'color': (0, 0.45, 0.45, 1.0),
#                                                    'linewidth': 0.5},
#                                     'color': 'black',
#                                     'size': 6},
#                        'annotatemarker': {'color': (0, 0.45, 0.45, 1.0),
#                                           'marker': 'o',
#                                           's': 3},
#                        'legend': {'fontsize': 6,
#                                   'fancybox': True},
#                        'legendframe': {'facecolor': (1.0, 1.0, 1.0, 0.4),
#                                        'edgecolor': (0, 0.45, 0.45, 1.0)},
#                        'legendtitle': {'fontsize': 8}}}

# verschilplot = {'MER': {'cmap': plt.cm.BrBG_r,
#                         'colorbar': {'format': '%.1f'},
#                         'colorbarlabel': {'labelsize': 6,
#                                           'labelcolor': (0, 0.45, 0.45, 1.0)},
#                         'colorbaroutline': {'edgecolor': (0, 0.45, 0.45, 1.0),
#                                             'linewidth': 0.5, }}}
##############################################################################

default = {
    'kleuren': {
        'schipholblauw': '#141251',
        'wit': '#000000',
        'ochtendroze': '#AA3191',
        'middagblauw': '#1B60DB',
        'schemergroen': '#027E9B',
        'avondpaars': '#6552A8',
        'ochtendlichtroze': '#FF8FB2',
        'middaglichtblauw': '#94B0EA',
        'schemerblauw': '#25D7F4',
        'avondlila': '#D285D6',
        'wolkengrijs_1': '#9491AA',
        'wolkengrijs_2': '#BFBDCC',
        'wolkengrijs_3': '#EAE9EE',
        'wolkengrijs_4': '#F2F1F4'
    },
    'fontname': {
        'grafiek': 'Arial',
        'tabel': 'Arial',
        'tekst': 'Frutiger for Schiphol'
    },
    'size': {
        'font': 13,
        'line': 2.5,
        'marker': 8
    }
}
        
###TODO Ed/Vincent: onderstaande functies netjes in de code-style van SSDtools zetten
# -----------------------------------------------------------------------------
# Pas de algemene plot style aan
# -----------------------------------------------------------------------------
def plot_style(style='MER2020'):
    ''' Algemene opmaak van een plot'''

    global xParams  # extra parameters t.o.v. rcParams
    xParams = dict()

    # Start met defaults
    plt.rcParams.update(plt.rcParamsDefault)
    
    # https://matplotlib.org/users/customizing.html
    # [(k,v) for k,v in plt.rcParams.items() if 'color' in k]
    
    if ':' in style:
        style, reeks = style.split(':')
    else:
        reeks = 1

    if style == 'MER2020':
        # fonts
        plt.rc('font', **{'family': 'sans-serif',
                          'sans-serif':'Frutiger for Schiphol Book',
                          'size':6})

        # grid
        # wolkengrijs 1: #9491AA
        # wolkengrijs 2: #BFBDCC
        # wolkengrijs 3: #EAE9EE
        plt.rc('axes',
               axisbelow=True,
               grid=True)
        plt.rc('grid',
               color='#BFBDCC',
               linewidth=0.3,
               linestyle='solid')

        # spines en background
        plt.rc('axes',
               edgecolor='#BFBDCC',
               linewidth=0.2,
               facecolor='white')   # achtergrond wit

        # labels
        plt.rc('axes',
               labelcolor='black',
               labelsize=10,
               labelpad=4)

        # label lines
        xParams['labellineprop'] = {'linewidth':1,
                                    'color':'black',
                                    'marker':'None'}
        xParams['labellinemin'] = 1

        # tick marks en labels
        plt.rc('xtick', labelsize=6, color='black')
        plt.rc('ytick', labelsize=6, color='black')

        # ticks
        plt.rc('xtick.major', size=0, width=0.5, pad=4)
        plt.rc('ytick.major', size=0, width=0.5, pad=4)
        plt.rc('xtick.minor', size=0, width=0.5, pad=4)
        plt.rc('ytick.minor', size=0, width=0.5, pad=4)

        # x- en Y-as
        plt.rc('axes', xmargin=0, ymargin=0)
        
        # legend
        plt.rc('legend',
                markerscale=0.8,
               fontsize=6,
               frameon=False,
               borderaxespad=0)
        plt.rc('text', color='Black')
        xParams['legend'] = dict(loc='lower right', bbox_to_anchor=(1, 1))
        
        # margins
        plt.rc('figure.subplot', bottom=0.2)       

        # lines en marker
        plt.rc('lines', linewidth=1,
                        markersize=4,
                        marker='o',
                        # markerfacecolor='None', # transparant
                        markerfacecolor='white',
                        # markeredgecolor='#141251', # zonder instellen volgen ze de cycler
                        markeredgewidth=0.5)


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
        # errorbars
        # door de kleuren niet in te stellen worden deze gelijk aan de lijnkleur
        # van de laatst geplotte lijn    
        xParams['errorbar'] = {#'color': '#141251',           # gemiddelde, lijnkleur
                               #'marker': 'None',             # gemiddelde, marker  
                               #'markeredgecolor': '#027E9B', # gemiddelede, markerkleur
                               'capsize': 3,                 # length of the error bar caps in points
                               'capthick': 2,                # thickness of the error bar cap
                               #'ecolor': '#9491AA',          # color the errorbar lines
                               'elinewidth': 1.5,            # linewidth of the errorbar lines

                               #'fillstyle': 'none',          # ???          
                               }
        
        xParams['prediction_fill'] = {#'color': '#027E9B',    # lijnkleur
                                      'color': get_cycler_color(1),
                                      'alpha': 0.2,          # doorzichtigheid
                                                   }        

        # barplot
        xParams['barplot'] = {'width': 0.6}
        plt.rc('patch', force_edgecolor=True,
                        linewidth=0.3,
                        edgecolor = 'white')
        
        # heatmap
        ###TODO: Afhankelijk maken van reeks, zie hieronder
#        xParams['cmap'] = colors.LinearSegmentedColormap.from_list('', ['#14125133', '#141251', 'black'])
        xParams['cmap'] = colors.LinearSegmentedColormap.from_list('', ['#94B0EA33', '#94B0EA', '#141251'])

    else:
        print('Warning: style not defined:', style)
        
        
# -----------------------------------------------------------------------------
# Get colors
# -----------------------------------------------------------------------------
def get_cycler_color(index=0):
    ''' Cycler color o.b.v. de index
    '''
    return plt.rcParams['axes.prop_cycle'].by_key()['color'][index]

# -----------------------------------------------------------------------------
# legenda
# -----------------------------------------------------------------------------
def update_legend_position(ax, target, fig=None):
    '''Update legend position, target, o.b.v. bbox.x1
    '''
    if fig is None:
        fig = plt.gcf()
    transf = fig.transFigure.inverted()
    renderer = fig.canvas.get_renderer()

    bbox = ax.get_tightbbox(renderer).transformed(transf)
    diff = target-bbox.x1

    l, b, w, h = ax.get_position().bounds
    # corrigeer x-positie
    ax.set_position([l+diff, b, w, h])

def get_text_bbox(handle, ax=None, fig=None):
    '''Bounding box of text
    '''
    if fig is None: fig = plt.gcf()
    if ax is None: ax = plt.gca()

    fig.canvas.draw()
    return handle.get_window_extent().inverse_transformed(ax.transData)

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
        lw = xParams['labellineprop']['linewidth']
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
        ###TODO hoe is die lw gedifinieerd? de x2 is een benadering die werkt
        lw = xParams['labellineprop']['linewidth']
        bbox.x0 -= plt.rcParams['ytick.major.pad'] + lw*2 # lijntjes
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