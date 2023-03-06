import numpy as np
import networkx as nx
import itertools
import matplotlib

# quick-and-dirty way for approximating this
def approx_hitting_time(P, i, j, T = 500):
    Q = P.copy()
    Q[j, :] = 0
    Q[j, j] = 1
    return np.sum(1-np.array([np.linalg.matrix_power(Q, k)[i, j] for k in range(T)]))

def draw(g, gene_names, thresh = None, node_list = None, edge_list = None, pos = None, cmap_dict = {0 : "MyGrey", 1 : "MyGrey"}, layout_args = "", draw = True, 
         kwargs_nodes = {"cmap" : matplotlib.colormaps["viridis"], "alpha" : 0.5}, kwargs_edges = {"edge_vmin" : -0.1, "edge_vmax" : 1.1}):
    # get weights from g
    edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())
    weights = np.array(list(weights))
    weights /= np.max(weights)
    # node and edgelist
    nl = list(g.nodes())
    el = list(g.edges())
    # subset edges
    if thresh is not None:
        edge_idx = np.where(weights > np.quantile(weights, thresh))[0]
        edgelist = [el[i] for i in edge_idx]
    elif edge_list is not None:
        edgelist = [tuple(i) for i in edge_list]
    # subset nodes
    if node_list is not None:
        nodelist = node_list
        edgelist = [e for e in edgelist if ((e[0] in nodelist) and (e[1] in nodelist))]
    else:
        node_idx = list(set(itertools.chain(*edgelist)))
        nodelist = [nl[i] for i in node_idx]
    # take subgraph
    g_sub = g.subgraph(nodelist)
    if draw == False:
        return g_sub 
    # now plot
    edges,weights = zip(*nx.get_edge_attributes(g_sub,'weight').items())
    _,ref = zip(*nx.get_edge_attributes(g_sub,'ref').items())
    weights = np.array(list(weights))
    weights /= np.max(weights)
    if pos is None:
        pos = nx.nx_agraph.graphviz_layout(g_sub, prog = 'fdp', args = layout_args)
    for (k, v) in cmap_dict.items():
        edgelist_color = [e for e in edgelist if g_sub.edges[e]['ref'] == k]
        edge_idx_color = np.where([e in edgelist_color for e in g_sub.edges()])[0]
        arrows = nx.draw_networkx_edges(g_sub, pos, 
                                        edgelist = edgelist_color, edge_color = weights[edge_idx_color], alpha = weights[edge_idx_color], 
                                        width = 2.5*weights[edge_idx_color], edge_cmap = matplotlib.colormaps[v], node_size = 600, **kwargs_edges)
        try:
            for a, w in zip(arrows, weights[edge_idx_color]):
                # from https://stackoverflow.com/questions/67251763/how-to-obtain-non-rounded-arrows-on-fat-lines
                a.set_mutation_scale(20 + w)
                a.set_joinstyle('miter')
                a.set_capstyle('butt')
        except:
            pass
    nodes, centrality = zip(*nx.get_node_attributes(g_sub,'centrality').items())
    centrality /= np.max(centrality)
    nx.draw_networkx_nodes(g, pos, nodelist = nodes, node_color = centrality, node_size = 600, **kwargs_nodes)
    nx.draw_networkx_labels(g, pos, labels = {x : gene_names[x] for (c, x) in zip(centrality, nodes)}, font_size = 14);
    return g_sub


def get_union_graph(g_all, gene_names, thresh = 0.99, layout_args = ""):
    nodes = []
    A_all = [np.array(nx.adjacency_matrix(g).todense()) for g in g_all]
    for g in g_all:
        g_sub = draw(g, gene_names, thresh = thresh, draw = False)
        nodes += g_sub.nodes
    nodes = list(set(nodes))
    A_agg = np.dstack(A_all).mean(-1)
    for i in range(A_agg.shape[0]):
        if i not in nodes:
            A_agg[i, :] = 0;
            A_agg[:, i] = 0;
    g_agg = nx.DiGraph(A_agg)
    pos = nx.nx_agraph.graphviz_layout(g_agg, prog = "fdp", args = layout_args)
    return g_agg, nodes, pos