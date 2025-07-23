# import math
# from typing import List, Tuple


# class Node:
#     """
#     Nodo del grafo di corrispondenza che rappresenta un'associazione
#     tra un punto di input e un punto target.
#     """
#     def __init__(self, index: int, input_id: int, target_id: int):
#         self.index = index
#         self.input_id = input_id
#         self.target_id = target_id
#         self.adjacents = []

#     def degree(self) -> int:
#         return len(self.adjacents)


# class Constraint:
#     """
#     Vincolo di distanza tra due punti appartenenti allo stesso set.
#     """
#     def __init__(self, i: int, j: int, dist: float):
#         self.i = i
#         self.j = j
#         self.dist = dist

#     def __lt__(self, other):
#         return self.dist < other.dist


# class CCDAMatcher:
#     """
#     Implementazione del Combined Constraint Data Association (CCDA).
#     """
#     def __init__(self, dist_tol: float = 0.10, dist_min: float = 0.0):
#         self.dist_tol = dist_tol
#         self.dist_min = dist_min

#     def set_dist_tol(self, dist_tol: float):
#         """
#         Imposta la tolleranza sulle differenze di distanza per essere compatibili.
#         """
#         self.dist_tol = dist_tol

#     def set_dist_min(self, dist_min: float):
#         """
#         Imposta la distanza minima tra due keypoint per essere un vincolo.
#         """
#         self.dist_min = dist_min

#     def match(self, v1: List, v2: List, match: List[Tuple[int, int]]) -> int:
#         """
#         Esegue il matching tra due set di punti.
#         """
#         nodes = []
#         constraints1 = []
#         constraints2 = []
#         clique_max = []

#         num2 = len(v2)

#         match.clear()

#         # Crea vincoli relativi all'interno di ciascun gruppo di keypoint
#         self.make_relative_constraints(v1, constraints1)
#         self.make_relative_constraints(v2, constraints2)

#         # Crea il grafo di corrispondenza basato sui vincoli
#         self.make_node_set(v1, v2, nodes)

#         # Crea gli archi
#         for constr_curr in constraints1:
#             constr_tmp = Constraint(0, 0, constr_curr.dist - self.dist_tol)
#             it = self.upper_bound(constraints2, constr_tmp)
#             # scorriamo tutti i vincoli di v2 che sono simili a quello corrente v1
#             while it < len(constraints2) and constraints2[it].dist < constr_curr.dist + self.dist_tol:
#                 target_constr = constraints2[it]
#                 if abs(target_constr.dist - constr_curr.dist) < self.dist_tol and \
#                         (target_constr.dist + constr_curr.dist) > self.dist_min:
                            
#                     # aggiungo un arco tra i nodi corrispondenti al primo punto
#                     # del vincolo corrente e il primo punto del vincolo target
#                     # e tra il secondo punto del vincolo corrente e il secondo
#                     # punto del vincolo target
                    
#                     isrc = target_constr.i + constr_curr.i * num2 
#                     idst = target_constr.j + constr_curr.j * num2
#                     nodes[isrc].adjacents.append(idst)
#                     nodes[idst].adjacents.append(isrc)
#                     # aggiungo un arco tra i nodi corrispondenti al secondo punto
#                     # del vincolo corrente e il primo punto del vincolo target
#                     # e tra il primo punto del vincolo corrente e il secondo
#                     # punto del vincolo target
                    
#                     isrc = target_constr.i + constr_curr.j * num2
#                     idst = target_constr.j + constr_curr.i * num2
#                     nodes[isrc].adjacents.append(idst)
#                     nodes[idst].adjacents.append(isrc)
#                 it += 1

#         # Trova la clique massima
#         self.find_clique_dyn(nodes, clique_max)

#         # Costruisce il risultato del matching
#         match.extend([(nodes[id].input_id, nodes[id].target_id) for id in clique_max])

#         return len(match)

#     def make_node_set(self, points1: List, points2: List, nodes: List[Node]):
#         """
#         Crea i nodi del grafo di corrispondenza. Ogni nodo rappresenta
#         un'associazione tra un punto di input e un punto target. 
#         """
#         nodes.clear()
#         for i in range(len(points1)):
#             for j in range(len(points2)):
#                 index = i * len(points2) + j
#                 nodes.append(Node(index, i, j))

#     def make_relative_constraints(self, points: List, constraints: List[Constraint]):
#         """
#         Crea i vincoli relativi tra i punti di un set.
#         """
#         constraints.clear()
#         for i in range(len(points)):
#             for j in range(i + 1, len(points)):
#                 dist = points[i].distance(points[j])
#                 if dist > self.dist_min:
#                     constraints.append(Constraint(i, j, dist))
#         constraints.sort()

#     def find_clique_dyn(self, nodes: List[Node], clique_max: List[int]):
#         """
#         Trova la clique massima nel grafo di corrispondenza.
#         """
#         nsize = len(nodes)
#         if nsize == 0:
#             return

#         # Crea la matrice di adiacenza
#         conn = [[False] * nsize for _ in range(nsize)]
#         for node in nodes:
#             for adj in node.adjacents:
#                 conn[node.index][adj] = True

#         # Trova la clique massima (algoritmo semplificato)
#         clique_max.clear()
#         visited = [False] * nsize
#         for i in range(nsize):
#             if not visited[i]:
#                 clique = [i]
#                 visited[i] = True
#                 for j in range(i + 1, nsize):
#                     if all(conn[k][j] for k in clique):
#                         clique.append(j)
#                         visited[j] = True
#                 if len(clique) > len(clique_max):
#                     clique_max[:] = clique

#     def upper_bound(self, constraints: List[Constraint], target: Constraint) -> int:
#         """
#         Trova il primo elemento maggiore di `target` in una lista ordinata.
#         """
#         low, high = 0, len(constraints)
#         while low < high:
#             mid = (low + high) // 2
#             if constraints[mid].dist > target.dist:
#                 high = mid
#             else:
#                 low = mid + 1
#         return low

import bisect
import networkx as nx
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np


class Constraint:
    """
    Constraint representing the distance between two points in the same set.
    """
    def __init__(self, i: int, j: int, dist: float):
        self.i = i
        self.j = j
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist


class CCDAMatcher:
    """
    Combined Constraint Data Association (CCDA) implementation using NetworkX.
    """
    def __init__(self, dist_tol: float = 0.50, dist_min: float = 0.0):
        self.dist_tol = dist_tol
        self.dist_min = dist_min

    def set_dist_tol(self, dist_tol: float):
        """
        Set tolerance for distance differences to be considered compatible.
        """
        self.dist_tol = dist_tol

    def set_dist_min(self, dist_min: float):
        """
        Set minimum distance between two keypoints to be considered a constraint.
        """
        self.dist_min = dist_min

    def match(self, v1: List, v2: List, match: List[Tuple[int, int]]) -> int:
        """
        Perform matching between two sets of points.
        """
        constraints1 = []
        constraints2 = []
        num2 = len(v2)
        
        match.clear()

        # Create relative constraints within each keypoint group
        self.make_relative_constraints(v1, constraints1)
        self.make_relative_constraints(v2, constraints2)

        # Create compatibility graph
        G = nx.Graph()
        
        # Create nodes
        for i in range(len(v1)):
            for j in range(len(v2)):
                # Each node represents a potential match between point i in v1 and point j in v2
                node_id = (i, j)
                G.add_node(node_id, input_id=i, target_id=j)

        # Create edges between compatible nodes
        for constr_curr in constraints1:
            # Find constraints in set 2 that are within tolerance range
            lower_bound = self.find_lower_bound(constraints2, constr_curr.dist - self.dist_tol)
            upper_bound = self.find_upper_bound(constraints2, constr_curr.dist + self.dist_tol)
            
            for idx in range(lower_bound, upper_bound):
                target_constr = constraints2[idx]
                
                # Check if constraints are compatible
                # if abs(target_constr.dist - constr_curr.dist) < self.dist_tol and \
                #    (target_constr.dist + constr_curr.dist) > self.dist_min:
                    
                # Create edges for direct match
                src_node = (constr_curr.i, target_constr.i)
                dst_node = (constr_curr.j, target_constr.j)
                if src_node in G and dst_node in G:
                    G.add_edge(src_node, dst_node)
                
                # Create edges for alternate match (handles orientation differences)
                src_node = (constr_curr.j, target_constr.i)
                dst_node = (constr_curr.i, target_constr.j)
                if src_node in G and dst_node in G:
                    G.add_edge(src_node, dst_node)
        # visualize graph
        # nx.draw(G, pos=nx.spring_layout(G, k=1.5),with_labels=True)
        # plt.savefig('compatibility_graph.png', dpi=300)
        # plt.show()
        
        # Find maximum clique
        clique = self.find_max_clique(G, v1, v2)
        
        # Build matching result
        for node in clique:
            # Verifica che il nodo sia una tupla, altrimenti usa i dati degli attributi
            if isinstance(node, tuple) and len(node) == 2:
                input_id, target_id = node
            else:
                # Se il nodo è un ID e non una tupla, ottieni gli attributi
                input_id = G.nodes[node]['input_id']
                target_id = G.nodes[node]['target_id']
                
            match.append((input_id, target_id))

        return len(match)

    def make_relative_constraints(self, points: List, constraints: List[Constraint]):
        """
        Create relative constraints between points in a set.
        """
        constraints.clear()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1 = points[i].position
                p2 = points[j].position
                arr1 = np.array([p1.x, p1.y, p1.z])
                arr2 = np.array([p2.x, p2.y, p2.z])
                dist = np.linalg.norm(arr1 - arr2)
                if dist > self.dist_min:
                    constraints.append(Constraint(i, j, dist))
        constraints.sort()

    def find_lower_bound(self, constraints: List[Constraint], value: float) -> int:
        """
        Find the index of first element >= value in sorted list.
        """
        return bisect.bisect_left([c.dist for c in constraints], value)

    def find_upper_bound(self, constraints: List[Constraint], value: float) -> int:
        """
        Find the index of first element > value in sorted list.
        """
        return bisect.bisect_right([c.dist for c in constraints], value)

    def find_max_clique(self, G: nx.Graph, v1: List, v2: List) -> List[Any]:
        """
        Find the maximum clique in the compatibility graph.
        
        Uses NetworkX's approximate maximum clique finder since the exact 
        algorithm may be too slow for large graphs.
        """
        try:
            # Per grafi piccoli, usa l'algoritmo esatto
            if G.number_of_nodes() <= 100:
                # Trova tutte le clique massimali e prendi quella di dimensione maggiore
                all_cliques = list(nx.find_cliques(G))
                if not all_cliques:
                    return []
                max_len = max(len(c) for c in all_cliques)
                max_cliques = [c for c in all_cliques if len(c) == max_len]

                def clique_distance_sum(clique):
                    s = 0.0
                    for node in clique:
                        i, j = node
                        p1 = v1[i].position
                        p2 = v2[j].position
                        arr1 = np.array([p1.x, p1.y, p1.z])
                        arr2 = np.array([p2.x, p2.y, p2.z])
                        s += np.linalg.norm(arr1 - arr2)
                    return s

                best = min(max_cliques, key=clique_distance_sum)
                return best
        
            # Per grafi più grandi, usa l'algoritmo approssimato
            return nx.approximation.max_clique(G)
        except:
            # Fallback to our custom greedy algorithm
            clique = []
            remaining = list(G.nodes())
            
            while remaining:
                # Add node with highest degree to clique
                node = max(remaining, key=lambda n: G.degree(n))
                clique.append(node)
                
                # Remove nodes not connected to all clique nodes
                new_remaining = []
                for n in remaining:
                    if n != node and all(G.has_edge(n, c) for c in clique):
                        new_remaining.append(n)
                
                if not new_remaining:
                    break
                    
                remaining = new_remaining
                
            return clique