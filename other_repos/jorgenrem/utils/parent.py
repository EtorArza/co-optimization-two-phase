

"""
Command line utility to parse and output parent network from DEAP history
"""
from matplotlib.widgets import CheckButtons
from modular_er.ea.map_elites import Map, size_behavior
from termcolor import colored, cprint
import argparse
import matplotlib.pyplot as plt
import networkx
import numpy as np
import os.path
import pickle
import sys
import zipfile

NUM_GENS = 100


def _generational_layout(network):
    import pygraphviz
    graph = networkx.nx_agraph.to_agraph(network)
    # Iterate all generations present in the network and place nodes with the
    # same generation at the same 'rank' to force 'dot' to place them at them
    # same height
    gen_ranks = {}
    for node in network.nodes(data=True):
        gen = node[1]['generation']
        if gen not in gen_ranks:
            gen_ranks[gen] = []
        gen_ranks[gen].append(node[0])
    for rank in gen_ranks.values():
        graph.add_subgraph(rank, rank='same')
    # Create layout
    graph.layout(prog='dot')
    # Create position dictionary used by 'networkx'
    node_pos = {}
    for n in graph:
        node = pygraphviz.Node(graph, n)
        xs = node.attr['pos'].split(',')
        node_pos[int(n)] = tuple(map(float, xs))
    return node_pos


def _plot_tree(network, ax, cbar=None):
    """Helper function to plot genealogy of a single individual"""
    network = network.reverse()
    # pos = graphviz_layout(network, prog='dot')
    pos = _generational_layout(network)
    color_tag = 'fitness'
    colors = [i[1][color_tag] if color_tag in i[1] else -1
              for i in network.nodes(data=True)]
    morph_edges = []
    ctrl_edges = []
    for edge in network.edges:
        a = network.nodes[edge[0]]
        b = network.nodes[edge[1]]
        if a['movable'] != b['movable'] or a['non_movable'] != b['non_movable']:
            morph_edges.append(edge)
        else:
            ctrl_edges.append(edge)
    if cbar:
        labels = {i[0]: i[1]['generation'] if 'generation' in i[1] else ''
                  for i in network.nodes(data=True)}
        networkx.draw_networkx(network, pos, ax=ax, node_color=colors,
                               edgelist=[], node_size=100,
                               vmin=cbar.vmin, vmax=cbar.vmax,
                               labels=labels,
                               font_size=8, font_color='white')
    else:
        networkx.draw_networkx(network, pos, ax=ax, node_color=colors,
                               edgelist=[], with_labels=False, node_size=100)
    networkx.draw_networkx_edges(network, pos, ax=ax, edgelist=morph_edges,
                                 edge_color='green')
    networkx.draw_networkx_edges(network, pos, ax=ax, edgelist=ctrl_edges,
                                 edge_color='black')


def _plot_map(m, ax):
    fitness = m.fitness()
    fitness = np.ma.masked_where(fitness == 0.0, fitness)
    imag = ax.imshow(fitness.T, origin='lower')
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    return imag


def _plot_network(network, ax):
    # Create network and positions
    network = network.reverse()
    pos = {}
    filt = {}
    for node in network.nodes(data=True):
        # Subtract 1 from non movable so it conforms with the fitness map above
        pos[node[0]] = (node[1]['non_movable'] - 1, node[1]['movable'])
        if (pos[node[0]] not in filt
                or node[1]['fitness'] > filt[pos[node[0]]][0]):
            filt[pos[node[0]]] = (node[1]['fitness'], node[0])
    nodes = [n[1] for n in filt.values()]
    color = [network.nodes[i]['generation']
             if 'generation' in network.nodes[i] else -1
             for i in nodes]
    morph_edges = []
    for edge in network.edges:
        a = network.nodes[edge[0]]
        b = network.nodes[edge[1]]
        if (a['movable'] != b['movable']
                or a['non_movable'] != b['non_movable']):
            morph_edges.append(edge)
    networkx.draw_networkx(network, pos, ax=ax, edgelist=[],
                           with_labels=False, node_size=100,
                           nodelist=nodes,
                           node_color=color, vmin=0, vmax=NUM_GENS)
    networkx.draw_networkx_edges(network, pos, ax=ax, edgelist=morph_edges,
                                 edge_color='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="History extractor")
    parser.add_argument('file', help="ZIP archive to process")
    args = parser.parse_args()
    with zipfile.ZipFile(args.file, 'r') as archive:
        seeds = set(filter(lambda p: p and 'checkpoint' not in p,
                           map(os.path.dirname, archive.namelist())))
        seed = ""
        if len(seeds) > 1:
            cprint("Found {:d} seeds, please select:".format(len(seeds)), 'blue')
            for seed in sorted(seeds):
                print("{}".format(seed))
            seed = input("Select ID: ").strip()
        history = [f for f in archive.namelist() if 'history.pickle' in f and
                   seed in f]
        population = [f for f in archive.namelist() if 'population.pickle' in f
                      and seed in f]
        if not history:
            sys.exit(colored("No valid history in archive '{!s}'"
                             .format(args.file),
                             'yellow'))
        history = history[-1]
        history = pickle.loads(archive.read(history))
        # This is safe since population will always be present if history
        # is present
        population = pickle.loads(archive.read(population[0]))
        # Create MAP to plot and pick from
        m = Map((20, 20))
        for indiv in population:
            behave = size_behavior(indiv, 20.)
            m.insert(behave, indiv)
        # Create plotting surfaces
        fig, (map_ax, tree_ax) = plt.subplots(1, 2)
        # Create check buttons for different modes
        ax_gui = plt.axes([0.25, 0.01, 0.15, 0.1])
        gui_check = CheckButtons(ax_gui, ['Plot on map', 'Save figures'],
                                 actives=[0, 0])
        # Plot initial map and colorbar for all
        map_imag = _plot_map(m, map_ax)
        cbar = fig.colorbar(map_imag, ax=tree_ax)

        def on_pick(event):
            if event.inaxes != map_ax:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            indiv = m._storage[x][y]
            if not indiv:
                return
            fig.suptitle('Loading individual')
            fig.canvas.draw()
            gens = history.genealogy(indiv)
            net = history.network(gens, prune=True)
            tree_ax.cla()
            map_ax.cla()
            _plot_map(m, map_ax)
            checks = gui_check.get_status()
            _plot_tree(net, tree_ax, cbar)
            if checks[0]:
                _plot_network(net, map_ax)
            fig.suptitle('')
            fig.canvas.draw_idle()
            import pdb
            pdb.set_trace()
            if checks[1]:
                extent = tree_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig("tree.pdf", dpi=600, bbox_inches=extent,
                            transparent=True)
                extent = map_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig("map.pdf", dpi=600, bbox_inches=extent,
                            transparent=True)
        fig.canvas.mpl_connect('button_press_event', on_pick)
        plt.show()
