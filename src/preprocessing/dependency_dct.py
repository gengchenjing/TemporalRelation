
from __future__ import print_function, unicode_literals
from nltk.parse.dependencygraph import DependencyGraph
import networkx as nx
from collections import defaultdict
from itertools import chain
from pprint import pformat
import subprocess
import warnings

from six import string_types

from nltk.tree import Tree
from nltk.compat import python_2_unicode_compatible

import sys   
sys.setrecursionlimit(100000)


#################################################################
# DependencyGraph Class
#################################################################


@python_2_unicode_compatible
class DependencyGraph(object):
    """
    A container for the nodes and labelled edges of a dependency structure.
    """

    def __init__(
        self,
        tree_str=None,
        cell_extractor=None,
        zero_based=False,
        cell_separator=None,
        top_relation_label='root',
    ):
        """Dependency graph.

        We place a dummy `TOP` node with the index 0, since the root node is
        often assigned 0 as its head. This also means that the indexing of the
        nodes corresponds directly to the Malt-TAB format, which starts at 1.

        If zero-based is True, then Malt-TAB-like input with node numbers
        starting at 0 and the root node assigned -1 (as produced by, e.g.,
        zpar).

        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        :param str top_relation_label: the label by which the top relation is
        identified, for examlple, `root`, `null` or `TOP`.

        """
        self.nodes = defaultdict(
            lambda: {
                'address': None,
                'word': None,
                'lemma': None,
                'ctag': None,
                'tag': None,
                'feats': None,
                'head': None,
                'deps': defaultdict(list),
                'rel': None,
            }
        )

        self.nodes[0].update({'ctag': 'TOP', 'tag': 'TOP', 'address': 0})

        self.root = None

        if tree_str:
            self._parse(
                tree_str,
                cell_extractor=cell_extractor,
                zero_based=zero_based,
                cell_separator=cell_separator,
                top_relation_label=top_relation_label,
            )


    def connect_graph(self):
        """
        Fully connects all non-root nodes.  All nodes are set to be dependents
        of the root node.
        """
        for node1 in self.nodes.values():
            for node2 in self.nodes.values():
                if node1['address'] != node2['address'] and node2['rel'] != 'TOP':
                    relation = node2['rel']
                    node1['deps'].setdefault(relation, [])
                    node1['deps'][relation].append(node2['address'])

                    # node1['deps'].append(node2['address'])

    def get_by_address(self, node_address):
        """Return the node with the given address."""
        return self.nodes[node_address]


    def contains_address(self, node_address):
        """
        Returns true if the graph contains a node with the given node
        address, false otherwise.
        """
        return node_address in self.nodes


    def to_dot(self):
        """Return a dot representation suitable for using with Graphviz.

        >>> dg = DependencyGraph(
        ...     'John N 2\\n'
        ...     'loves V 0\\n'
        ...     'Mary N 2'
        ... )
        >>> print(dg.to_dot())
        digraph G{
        edge [dir=forward]
        node [shape=plaintext]
        <BLANKLINE>
        0 [label="0 (None)"]
        0 -> 2 [label="root"]
        1 [label="1 (John)"]
        2 [label="2 (loves)"]
        2 -> 1 [label=""]
        2 -> 3 [label=""]
        3 [label="3 (Mary)"]
        }

        """
        # Start the digraph specification
        s = 'digraph G{\n'
        s += 'edge [dir=forward]\n'
        s += 'node [shape=plaintext]\n'

        # Draw the remaining nodes
        for node in sorted(self.nodes.values(), key=lambda v: v['address']):
            s += '\n%s [label="%s (%s)"]' % (
                node['address'],
                node['address'],
                node['word'],
            )
            for rel, deps in node['deps'].items():
                for dep in deps:
                    if rel is not None:
                        s += '\n%s -> %s [label="%s"]' % (node['address'], dep, rel)
                    else:
                        s += '\n%s -> %s ' % (node['address'], dep)
        s += "\n}"

        return s


    def __str__(self):
        return pformat(self.nodes)

    def __repr__(self):
        return "<DependencyGraph with {0} nodes>".format(len(self.nodes))

    @staticmethod
    def load(
        filename, zero_based=False, cell_separator=None, top_relation_label='root'
    ):
        """
        :param filename: a name of a file in Malt-TAB format
        :param zero_based: nodes in the input file are numbered starting from 0
        rather than 1 (as produced by, e.g., zpar)
        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.
        :param str top_relation_label: the label by which the top relation is
        identified, for examlple, `root`, `null` or `TOP`.

        :return: a list of DependencyGraphs

        """
        with open(filename) as infile:
            return [
                DependencyGraph(
                    tree_str,
                    zero_based=zero_based,
                    cell_separator=cell_separator,
                    top_relation_label=top_relation_label,
                )
                for tree_str in infile.read().split('\n\n')
            ]


    def left_children(self, node_index):
        """
        Returns the number of left children under the node specified
        by the given address.
        """
        children = chain.from_iterable(self.nodes[node_index]['deps'].values())
        index = self.nodes[node_index]['address']
        return sum(1 for c in children if c < index)


    def right_children(self, node_index):
        """
        Returns the number of right children under the node specified
        by the given address.
        """
        children = chain.from_iterable(self.nodes[node_index]['deps'].values())
        index = self.nodes[node_index]['address']
        return sum(1 for c in children if c > index)


    def add_node(self, node):
        if not self.contains_address(node['address']):
            self.nodes[node['address']].update(node)


    def _parse(
        self,
        input_,
        cell_extractor=None,
        zero_based=False,
        cell_separator=None,
        top_relation_label='root',
    ):
        """Parse a sentence.

        :param extractor: a function that given a tuple of cells returns a
        7-tuple, where the values are ``word, lemma, ctag, tag, feats, head,
        rel``.

        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        :param str top_relation_label: the label by which the top relation is
        identified, for examlple, `root`, `null` or `TOP`.

        """

        def extract_3_cells(cells, index):
            word, tag, head = cells
            return index, word, word, tag, tag, '', head, ''

        def extract_4_cells(cells, index):
            word, tag, head, rel = cells
            return index, word, word, tag, tag, '', head, rel

        def extract_7_cells(cells, index):
            line_index, word, lemma, tag, _, head, rel = cells
            try:
                index = int(line_index)
            except ValueError:
                # index can't be parsed as an integer, use default
                pass
            return index, word, lemma, tag, tag, '', head, rel

        def extract_10_cells(cells, index):
            line_index, word, lemma, ctag, tag, feats, head, rel, _, _ = cells
            try:
                index = int(line_index)
            except ValueError:
                # index can't be parsed as an integer, use default
                pass
            return index, word, lemma, ctag, tag, feats, head, rel

        extractors = {
            3: extract_3_cells,
            4: extract_4_cells,
            7: extract_7_cells,
            10: extract_10_cells,
        }

        if isinstance(input_, string_types):
            input_ = (line for line in input_.split('\n'))

        lines = (l.rstrip() for l in input_)
        lines = (l for l in lines if l)

        cell_number = None
        for index, line in enumerate(lines, start=1):
            cells = line.split(cell_separator)
            if cell_number is None:
                cell_number = len(cells)
            else:
                assert cell_number == len(cells)

            if cell_extractor is None:
                try:
                    cell_extractor = extractors[cell_number]
                except KeyError:
                    raise ValueError(
                        'Number of tab-delimited fields ({0}) not supported by '
                        'CoNLL(10) or Malt-Tab(4) format'.format(cell_number)
                    )

            try:
                index, word, lemma, ctag, tag, feats, head, rel = cell_extractor(
                    cells, index
                )
            except (TypeError, ValueError):
                # cell_extractor doesn't take 2 arguments or doesn't return 8
                # values; assume the cell_extractor is an older external
                # extractor and doesn't accept or return an index.
                word, lemma, ctag, tag, feats, head, rel = cell_extractor(cells)

            if head == '_':
                continue

            head = int(head)
            if zero_based:
                head += 1

            self.nodes[index].update(
                {
                    'address': index,
                    'word': word,
                    'lemma': lemma,
                    'ctag': ctag,
                    'tag': tag,
                    'feats': feats,
                    'head': head,
                    'rel': rel,
                }
            )

            # Make sure that the fake root node has labeled dependencies.
            if (cell_number == 3) and (head == 0):
                rel = top_relation_label
            self.nodes[head]['deps'][rel].append(index)

        if self.nodes[0]['deps'][top_relation_label]:
            root_address = self.nodes[0]['deps'][top_relation_label][0]
            self.root = self.nodes[root_address]
            self.top_relation_label = top_relation_label
        else:
            warnings.warn(
                "The graph doesn't contain a node " "that depends on the root element."
            )

    def _word(self, node, filter=True):
        w = node['word']
        if filter:
            if w != ',':
                return w
        return w

    def _tree(self, i):
        """ Turn dependency graphs into NLTK trees.

        :param int i: index of a node
        :return: either a word (if the indexed node is a leaf) or a ``Tree``.
        """
        node = self.get_by_address(i)
        word = node['word']
        deps = sorted(chain.from_iterable(node['deps'].values()))

        if deps:
            return Tree(word, [self._tree(dep) for dep in deps])
        else:
            return word

    def tree(self):
        """
        Starting with the ``root`` node, build a dependency tree using the NLTK
        ``Tree`` constructor. Dependency labels are omitted.
        """
        node = self.root

        word = node['word']
        deps = sorted(chain.from_iterable(node['deps'].values()))
        return Tree(word, [self._tree(dep) for dep in deps])


    def triples(self, node=None):
        """
        Extract dependency triples of the form:
        ((head word, head tag), rel, (dep word, dep tag))
        """

        if not node:
            node = self.root

        head = (node['word'], node['ctag'])
        for i in sorted(chain.from_iterable(node['deps'].values())):
            dep = self.get_by_address(i)
            yield (head, dep['rel'], (dep['word'], dep['ctag']))
            for triple in self.triples(node=dep):
                yield triple


    def _hd(self, i):
        try:
            return self.nodes[i]['head']
        except IndexError:
            return None

    def _rel(self, i):
        try:
            return self.nodes[i]['rel']
        except IndexError:
            return None

    # what's the return type?  Boolean or list?
    def contains_cycle(self):
        """Check whether there are cycles.

        >>> dg = DependencyGraph(treebank_data)
        >>> dg.contains_cycle()
        False

        >>> cyclic_dg = DependencyGraph()
        >>> top = {'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0}
        >>> child1 = {'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1}
        >>> child2 = {'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2}
        >>> child3 = {'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3}
        >>> child4 = {'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4}
        >>> cyclic_dg.nodes = {
        ...     0: top,
        ...     1: child1,
        ...     2: child2,
        ...     3: child3,
        ...     4: child4,
        ... }
        >>> cyclic_dg.root = top

        >>> cyclic_dg.contains_cycle()
        [3, 1, 2, 4]

        """
        distances = {}

        for node in self.nodes.values():
            for dep in node['deps']:
                key = tuple([node['address'], dep])
                distances[key] = 1

        for _ in self.nodes:
            new_entries = {}

            for pair1 in distances:
                for pair2 in distances:
                    if pair1[1] == pair2[0]:
                        key = tuple([pair1[0], pair2[1]])
                        new_entries[key] = distances[pair1] + distances[pair2]

            for pair in new_entries:
                distances[pair] = new_entries[pair]
                if pair[0] == pair[1]:
                    path = self.get_cycle_path(self.get_by_address(pair[0]), pair[0])
                    return path

        return False  # return []?


    def get_cycle_path(self, curr_node, goal_node_index):
        for dep in curr_node['deps']:
            if dep == goal_node_index:
                return [curr_node['address']]
        for dep in curr_node['deps']:
            path = self.get_cycle_path(self.get_by_address(dep), goal_node_index)
            if len(path) > 0:
                path.insert(0, curr_node['address'])
                return path
        return []


    def to_conll(self, style):
        """
        The dependency graph in CoNLL format.

        :param style: the style to use for the format (3, 4, 10 columns)
        :type style: int
        :rtype: str
        """

        if style == 3:
            template = '{word}\t{tag}\t{head}\n'
        elif style == 4:
            template = '{word}\t{tag}\t{head}\t{rel}\n'
        elif style == 10:
            template = (
                '{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n'
            )
        else:
            raise ValueError(
                'Number of tab-delimited fields ({0}) not supported by '
                'CoNLL(10) or Malt-Tab(4) format'.format(style)
            )

        return ''.join(
            template.format(i=i, **node)
            for i, node in sorted(self.nodes.items())
            if node['tag'] != 'TOP'
        )





class DependencyGraphError(Exception):
    """Dependency graph exception."""

def demo():
    #malt_demo()
    wf = open('dct_result.txt','w+')
    for i in range(1, 55):
        c, lst = confi(i)
        conll_file_demo(wf, c, lst, i)
    
def conll_file_demo(wf, c, lst, i):
    print('Mass conll_read demo...')
    graphs = [DependencyGraph(entry) for entry in lst if entry]
    ars = []
    sent_ids = []
    events = []
    reltypes = []
    test_path = "./test/test{:0>2d}.txt".format(i)
    with open(test_path,'r+')as f1:
        lines = f1.readlines()
        for line in lines:
            cols = line.split("\t")
            ars.append((cols[1],cols[4]))
            sent_ids.append(cols[0])
            events.append(cols[3])
            reltypes.append(cols[6])
    print(len(graphs),len(ars))
    for l, graph, ar, sent_id_tmp, event, reltype in zip(lst, graphs, ars, sent_ids, events, reltypes):
        reltype = reltype.strip()
        depgraph = DependencyGraph(l.strip())
        depgraph.tree = graph.tree() 
        dep_nx = nxGraphWroot(depgraph)
        dep_nx = dep_nx.to_undirected()         
        #print(sent_id_tmp, ar)
        shortest_path = nx.shortest_path(dep_nx, source = int(ar[0]),target = int(ar[1]))
        shortest_path_word = []
        for i in shortest_path:
            shortest_path_word.append(c[sent_id_tmp][i])
        write_line = sent_id_tmp + '\t' + "reltype=" + reltype + '\t' + str(shortest_path_word) + '\n'
        wf.write(write_line)
        
    
def confi(i):
    conll_str = ' '
    conll_path = "./dct_final/{:0>3d}.conll".format(i)
    with open(conll_path,'r')as f:
        for lines in f.readlines():
            if lines.startswith('#'):
                continue
            else:
                conll_str += lines

    lst = conll_str.split('\n\n')

    f = open(conll_path,'r')
    lines = f.readlines()
    c1 = {}
    c2 = {}
    sent_id = ''
    for i in range(len(lines)):
        if lines[i].startswith('# sent_id'):
            c1[sent_id] = c2
            c2 = {}
            sent_id = lines[i].split('=')[1][1:-1]
            continue
        elif lines[i].startswith('#'):
            continue
        else:
            tmp = lines[i].split('\t')
            if tmp[0] == '\n':
                continue
            a = int(tmp[0])
            b = tmp[1]
            c2[a] = b
    c1[sent_id] = c2
    # ff = open('e2e.txt','r')
    # lines = ff.readlines()
    # c3 = {}
    # c4 = {}
    # name = ''
    # i_name = ''
    # for i in range(len(lines)):
    #     tmp = lines[i].split('\t')
    #     if tmp[0] != name:
    #         c3[i_name] = c4
    #         c4 = {}
    #         name = tmp[0]
    #         i_name = 'A'+tmp[0][2:5]+'n'+tmp[0][7:]
    #         c4[tmp[3]] = tmp[8][1:-1]
    #     c4[tmp[3]] = tmp[8][1:-1]
    # c3[i_name] = c4
    # return ((c1, c3),lst)

    return (c1,lst)
    
def nxGraphWroot(depgraph):
        
        """Convert the data in a ``nodelist`` into a networkx labeled directed graph.
        Include the ROOT node
        """
        import networkx
        
        nx_nodelist = list(range(0, len(depgraph.nodes))) ##
        nx_edgelist = [
            (n, depgraph._hd(n), depgraph._rel(n))
            for n in nx_nodelist
        ]
        depgraph.nx_labels = {}
        for n in nx_nodelist:
            depgraph.nx_labels[n] = depgraph.nodes[n]['word']
            
        g = networkx.MultiDiGraph()
        g.add_nodes_from(nx_nodelist)
        g.add_edges_from(nx_edgelist)
            
        return g

if __name__ == '__main__':
    demo()


    


