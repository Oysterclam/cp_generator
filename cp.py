import random
from scipy.spatial import Delaunay
import numpy as np
import math
from scipy.optimize import minimize
from itertools import product
import svgwrite


class Vertex():
    """A vertex is a point in the crease pattern.
    """

    def __init__(self, x, y):
        self.d = 0 # degree zero by default
        self.adj = set() # adjacent folds
        self.adjv = set() # adjacent vertices
        self.x = x
        self.y = y

class Fold():
    """A fold is a line segment between two vertices.
       It is either a mountain or a valley (0 or 1)
       Or unknown (-1)
    """
    def __init__(self, v1, v2, type):
        self.v1 = v1
        self.v2 = v2
        self.type = type

class CreasePattern():
    """A crease pattern is a graph with vertices and folds between them
    """

    def __init__(self):
        self.side = 1 #default side length
        self.vertices = []
        self.folds = set()

    def normalize(self):
        # scale the crease pattern so that the vertices are in the range [0, 1]
        # do this by dividing by the side length

        for v in self.vertices:
            v.x /= self.side
            v.y /= self.side

        self.side = 1

    def scale(self, n):
        # scale the crease pattern by n
        for v in self.vertices:
            v.x *= n
            v.y *= n
        self.side *= n

    def add_vertex(self, x, y):
        # if the vertex is already in the crease pattern, don't add it
        for v in self.vertices:
            if v.x == x and v.y == y:
                return
        self.vertices.append(Vertex(x, y))

    def add_foldf(self, f):
        # if the fold is already in the crease pattern, don't add it
        for f2 in self.folds:
            if (f2.v1 == f.v1 and f2.v2 == f.v2) or (f2.v1 == f.v2 and f2.v2 == f.v1):
                return
        # if the vertices are not in the crease pattern, add them
        if f.v1 not in self.vertices:
            self.vertices.append(f.v1)
        if f.v2 not in self.vertices:
            self.vertices.append(f.v2)
        self.folds.add(f)
        f.v1.d += 1
        f.v1.adj.add(f)
        f.v1.adjv.add(f.v2)
        f.v2.d += 1
        f.v2.adj.add(f)
        f.v2.adjv.add(f.v1)

    def add_fold(self, v1, v2, type=-1):
        # if the fold is already in the crease pattern, don't add it
        for f in self.folds:
            if (f.v1 == v1 and f.v2 == v2) or (f.v1 == v2 and f.v2 == v1):
                return
        # if the vertices are not in the crease pattern, add them
        if v1 not in self.vertices:
            self.vertices.append(v1)
        if v2 not in self.vertices:
            self.vertices.append(v2)
        f = Fold(v1, v2, type)
        self.folds.add(f)
        v1.d += 1
        v1.adj.add(f)
        v1.adjv.add(v2)
        v2.d += 1
        v2.adj.add(f)
        v2.adjv.add(v1)

    def add_random_vertex(self):
        self.add_vertex(random.randint(0, self.side), random.randint(0, self.side))

    def add_random_vertex_on_edge(self):
        # add a vertex on the edge of the square
        x = random.randint(0, self.side)
        y = random.randint(0, self.side)
        if x == 0 or x == self.side:
            y = random.randint(0, self.side)
        else:
            y = 0
        self.vertices.add(Vertex(x, y))

    def add_square_vertices(self):
        # add the vertices of the square
        self.add_vertex(0, 0)
        self.add_vertex(self.side, 0)
        self.add_vertex(self.side, self.side)
        self.add_vertex(0, self.side)

    def push_to_edge(self, x):
        # if a vertex is within x of an edge, push it to the edge
        for v in self.vertices:
            if v.x < x:
                v.x = 0
            if v.x > self.side - x:
                v.x = self.side
            if v.y < x:
                v.y = 0
            if v.y > self.side - x:
                v.y = self.side
        # if any folds are on the edge, remove them
        set_copy = self.folds.copy()
        for f in set_copy:
            if self.on_edge(f.v1) and self.on_edge(f.v2):
                self.remove_fold(f)

    def triangulate(self):
        # perform Delaunay triangulation
        vertices = list(self.vertices)
        points = []
        for v in vertices:
            points.append([v.x, v.y])
        points = np.array(points)
        tri = Delaunay(points)
        # add folds with default to nothing
        for t in tri.simplices:
            self.add_fold(vertices[t[0]], vertices[t[1]], -1)
            self.add_fold(vertices[t[1]], vertices[t[2]], -1)
            self.add_fold(vertices[t[2]], vertices[t[0]], -1)

    def remove_fold(self, f):
        # remove a fold from the crease pattern
        self.folds.remove(f)
        f.v1.d -= 1
        f.v2.d -= 1
        f.v1.adj.remove(f)
        f.v2.adj.remove(f)
        # if the vertices are now degree zero, remove them
        if f.v1.d == 0:
            self.vertices.remove(f.v1)
        if f.v2.d == 0:
            self.vertices.remove(f.v2)

    def remove_vertex(self, v):
        # remove a vertex from the crease pattern
        # first, remove the adjacent folds
        adj = v.adj.copy()
        for f in adj:
            self.remove_fold(f)
        # then, remove the vertex
        self.vertices.remove(v)

    def remove_edge_folds(self):
        # remove folds that are on the edge of the square
        set_copy = self.folds.copy()
        for f in set_copy:
            if f.v1.x == 0 and f.v2.x == 0:
                self.remove_fold(f)
            if f.v1.x == self.side and f.v2.x == self.side:
                self.remove_fold(f)
            if f.v1.y == 0 and f.v2.y == 0:
                self.remove_fold(f)
            if f.v1.y == self.side and f.v2.y == self.side:
                self.remove_fold(f)

    def on_edge(self, v):
        # check if a vertex is on the edge of the square
        if v.x == 0 or v.x == self.side or v.y == 0 or v.y == self.side:
            return True
        return False

    def on_edge_fold(self, f):
        # check if a fold lies on the edge of the square
        return self.on_edge(f.v1) and self.on_edge(f.v2)

    def on_corner(self, v):
        # check if a vertex is on a corner of the square
        if (v.x == 0 or v.x == self.side) and (v.y == 0 or v.y == self.side):
            return True
        return False

    def evenize_vertices(self):
        # make the vertices have even degree by removing edges
        # There must be an even number of odd vertices

        # first, get a list of the vertices that have odd degree
        odd_vertices = []
        for v in self.vertices:
            if v.d % 2 == 1:
                odd_vertices.append(v)

        # loop until there are no odd vertices
        while len(odd_vertices) > 0:
            # pick the first odd vertex
            v = odd_vertices[0]
            # find the next closest odd vertex using BFS

            # first, initialize the queue
            queue = []
            queue.append(v)
            # initialize the set of visited vertices
            visited = set()
            visited.add(v)
            # initialize the dictionary of parents
            parents = {}
            parents[v] = None
            # search for the next closest odd vertex
            found = False
            while len(queue) > 0:
                # get the next vertex
                v = queue.pop(0)
                # check if it is odd
                if v.d % 2 == 1 and v != odd_vertices[0]:
                    found = True
                    break
                # add the adjacent vertices to the queue
                for f in v.adj:
                    if f.v1 == v:
                        v2 = f.v2
                    else:
                        v2 = f.v1
                    if v2 not in visited:
                        queue.append(v2)
                        visited.add(v2)
                        parents[v2] = v
            # get the path from the first odd vertex to the next closest odd vertex
            path = []
            if found:
                while v is not None:
                    path.append(v)
                    v = parents[v]
            # remove the edges along the path
            for i in range(len(path)-1):
                # make a copy of the set of adjacent folds
                # because we can't modify the set while iterating over it
                adj = path[i].adj.copy()
                for f in adj:
                    if f.v1 == path[i+1] or f.v2 == path[i+1]:
                        self.remove_fold(f)
            # update the list of odd vertices
            odd_vertices = []
            for v in self.vertices:
                if v.d % 2 == 1:
                    odd_vertices.append(v)

    def even_degree(self):
        # check if all vertices have even degree
        for v in self.none_edge_vertices():
            if v.d % 2 == 1:
                return False
        return True

    def clockwise_neighbors(self, v):
        # return list of vertices adjacent to v in clockwise order

        # first, get a list of the vertices adjacent to v
        adj = []
        for f in v.adj:
            if f.v1 == v:
                adj.append(f.v2)
            else:
                adj.append(f.v1)

        # next, sort them in clockwise order
        # use atan2 to get the angle of each vertex relative to v
        angles = []
        for v2 in adj:
            angles.append(math.atan2(v2.y - v.y, v2.x - v.x) + math.pi)
        # sort the vertices by angle
        adj = [x for _,x in sorted(zip(angles, adj), key=lambda x: x[0])]
        return adj

    def adjacent_angles(self, v):
        # return list of angles between adjacent vertices in clockwise order
        # if the angle would be negative, add 2pi
        angles = []
        adj = self.clockwise_neighbors(v)
        for i in range(len(adj)):
            v1 = adj[i]
            v2 = adj[(i+1)%len(adj)]
            angle = math.atan2(v2.y - v.y, v2.x - v.x) - math.atan2(v1.y - v.y, v1.x - v.x)
            if angle < 0:
                angle += 2*math.pi
            angles.append(angle)
        return angles

    def none_edge_vertices(self):
        # return list of vertices that are not on the edge of the square
        vertices = []
        for v in self.vertices:
            if not self.on_edge(v):
                vertices.append(v)
        return vertices

    def l2_regularization(self, x, alpha):
        # Calculate the L2 regularization term
        reg_term = 0.5 * alpha * np.sum(x**2)
        return reg_term

    def objective(self, on_edge):
        # the objective is to minimize the sum of the squares of the distances between the guess and the actual coordinates
        # we only care about the vertices that are not on the edge of the square
        # x is a guess

        # get the indices in self.vertices of the vertices that are not on the edge of the square
        on_edge_indices = []
        for i in range(len(on_edge)):
            if on_edge[i]:
                on_edge_indices.append(i)

        # make a list of coordinates for each vertex to match the format of x
        coords = []
        for v in self.vertices:
            coords.append(v.x)
            coords.append(v.y)

        W = 1
        def fun(X):
            sum = 0
            for i in range(len(self.vertices)):
                if i in on_edge_indices:
                    W = 1000000
                else:
                    W = 1
                sum += W * ((X[2*i] - coords[2*i])**2 + (X[2*i+1]- coords[2*i+1])**2)
            res = self.l2_regularization(sum, .1)
            return res
        return fun

    def make_constraints(self, parity, vi1, vi2, indices):
        def fun(X):
            v_coords = [X[vi1], X[vi2]]
            # get the coordinates of the adjacent vertices indexed from x
            adj_coords = []
            for i in indices:
                adj_coords.append(X[i])
            # so adj_coords is a list of the coordinates of the adjacent vertices
            # x, y, x, y, x, y, ...
            # then we get the coordinates of the vertex from X

            # get the angles between adjacent vertices in clockwise order
            angles = []
            for i in range(0, len(indices), 2):
                x1 = adj_coords[i]
                y1 = adj_coords[i+1]
                x2 = adj_coords[(i+2)%len(adj_coords)]
                y2 = adj_coords[(i+3)%len(adj_coords)]

                angle = math.atan2(y2 - v_coords[1], x2 - v_coords[0]) - math.atan2(y1 - v_coords[1], x1 - v_coords[0])
                if angle < 0:
                    angle += 2*math.pi
                angles.append(angle)
            # sum of even angles should be pi
            s = 0
            for i in range(len(angles)):
                if i % 2 == parity:
                    s += angles[i]
            return s - math.pi
        return fun

    def generate_constraints(self):
        # generate constraints for scipy.optimize.minimize
        # the sum of the even angles around the vertex should be pi
        # the sum of the odd angles around the vertex should be pi as well
        # this should take in a list of coordinates of vertices
        # then, it should generate constraints for each vertex using the coordinates and the adjacent vertices

        # get the vertices that are not on the edge of the square
        vertices = self.vertices

        # generate constraints
        constraints = []

        for i in range(len(vertices)):
            v = vertices[i]
            vi1 = 2*i
            vi2 = 2*i+1

            # get the indices of the adjacent vertices
            indices = []
            for v2 in self.clockwise_neighbors(v):
                indices.append(2*self.vertices.index(v2))
                indices.append(2*self.vertices.index(v2)+1)

            constraint2 = self.make_constraints(1, vi1, vi2, indices)

            if not self.on_edge(v):
                # add the constraints only if the vertex is not on the edge of the square
                constraints.append({'type': 'eq', 'fun': constraint2})
        return constraints


    def optimize(self):
        # initial guess is current coordinates of vertices
        # constraints are that the even angles around each vertex are pi
        # and the odd angles are pi as well

        # first, normalize the crease pattern
        self.normalize()
        # generate initial guess
        x0 = []
        # note which vertices are not on the edge of the square
        on_edge = []
        for v in self.vertices:
            x0.append(v.x)
            x0.append(v.y)
            if self.on_edge(v):
                on_edge.append(True)
            else:
                on_edge.append(False)

        # minimize
        myconstraints = self.generate_constraints()
        res = minimize(self.objective(on_edge), x0, constraints = myconstraints, method='SLSQP', options={'disp': True})
        # update the coordinates of the vertices
        for i in range(len(self.vertices)):
            self.vertices[i].x = res.x[2*i]
            self.vertices[i].y = res.x[2*i+1]

        # by default, push all vertices to the edge of the square with a 1% tolerance
        self.push_to_edge(0.001*self.side)

        return res

    def clear(self):
        # remove all vertices and folds
        self.vertices = []
        self.folds = set()

    def maekawa(self, v):
        # check if v satisfies Maekawa's theorem, that the number of mountain folds is equal to the number of valley folds +- 2
        # first, get the number of mountain and valley folds
        mn = 0
        vl = 0
        for f in v.adj:
            if f.type == 0:
                mn += 1
            elif f.type == 1:
                vl += 1
        if mn == vl + 2 or mn == vl - 2:
            return True
        return False

    def get_pairings(self, v):
        # Run algorithm to pair vertices around locally minimal angles
        # This is the algorithm from the paper The Complexity of Flat Origami by Bern and Hayes
        # angles[i] is the angle between adj[i] and adj[i+1]
        adj = self.clockwise_neighbors(v)
        adj_folds = []
        for v1 in adj:
            for f in v1.adj:
                if f.v1 == v or f.v2 == v:
                    adj_folds.append(f)
        angles = self.adjacent_angles(v)

        # start with empty pairings
        pairings = []
        # loop until all folds are paired
        i = 0
        while len(angles) > 2:
            for i in range(len(angles)):
                # check if angle is locally minimal (less than both adjacent angles)
                if angles[i] <= angles[(i+1)%len(angles)] and angles[i] <= angles[(i-1)%len(angles)]:
                    # then pair the folds
                    # add the flag 0 to indicate these folds have opposite mountain/valley assignment
                    pairings.append(([adj_folds[i], adj_folds[(i+1)%len(adj_folds)]], [2,0]))
                    # remove the wedge by subtracting angles[i] from angles[i+1], and removing angles[i]
                    # then, remove the adjacent folds
                    angles[(i-1)%len(angles)] = angles[(i-1)%len(angles)] + angles[(i+1)%len(angles)] - angles[i]
                    del angles[i]
                    del angles[(i+1)%len(angles)]
                    del adj_folds[i]
                    del adj_folds[i%len(adj_folds)]
                    break
        # the final two folds are paired with the flag 1 to indicate they have the same mountain/valley assignment
        if len(angles) == 2:
            pairings.append(([adj_folds[0], adj_folds[1]], [2,1]))
        return pairings

    def get_pairings_on_edge(self, v):
        if(self.on_corner(v)):
            # case when v is on a corner of the square
            # make different cases for each corner
            if v.x == 0 and v.y == 0:
                # top left corner
                f1 = Fold(v, Vertex(0, self.side), 0)
                f2 = Fold(v, Vertex(self.side, 0), 0)
            elif v.x == self.side and v.y == 0:
                # top right corner
                f1 = Fold(v, Vertex(0, 0), 0)
                f2 = Fold(v, Vertex(self.side, self.side), 0)
            elif v.x == self.side and v.y == self.side:
                # bottom right corner
                f1 = Fold(v, Vertex(self.side, 0), 0)
                f2 = Fold(v, Vertex(0, self.side), 0)
            elif v.x == 0 and v.y == self.side:
                # bottom left corner
                f1 = Fold(v, Vertex(0, 0), 0)
                f2 = Fold(v, Vertex(self.side, self.side), 0)
        elif self.on_edge(v):
            # case when v is on an edge of the square
            # make different cases for each edge
            if v.x == 0:
                # left edge
                f1 = Fold(v, Vertex(0, 0), 0)
                f2 = Fold(v, Vertex(0, self.side), 0)
            elif v.x == self.side:
                # right edge
                f1 = Fold(v, Vertex(self.side, 0), 0)
                f2 = Fold(v, Vertex(self.side, self.side), 0)
            elif v.y == 0:
                # top edge
                f1 = Fold(v, Vertex(0, 0), 0)
                f2 = Fold(v, Vertex(self.side, 0), 0)
            elif v.y == self.side:
                # bottom edge
                f1 = Fold(v, Vertex(0, self.side), 0)
                f2 = Fold(v, Vertex(self.side, self.side), 0)
        self.add_foldf(f1)
        self.add_foldf(f2)
        # case when v is not on the edge of the square
        adj = self.clockwise_neighbors(v)
        # rotate the list of adjacent vertices so that the first vertex is on the edge
        while not self.on_edge(adj[0]):
            adj.append(adj.pop(0))
        # get the adjacent folds
        adj_folds = []
        for v1 in adj:
            for f in v1.adj:
                if f.v1 == v or f.v2 == v:
                    adj_folds.append(f)
        # get the angle between successive adjacent folds
        angles = []
        for i in range(len(adj)):
            v1 = adj[i]
            v2 = adj[(i+1)%len(adj)]
            angle = math.atan2(v2.y - v.y, v2.x - v.x) - math.atan2(v1.y - v.y, v1.x - v.x)
            if angle < 0:
                angle += 2*math.pi
            angles.append(angle)
        # start with empty pairings
        pairings = []

        # loop until all vertices are paired
        # the difference between now and before is we don't take the mod at the edges
        while len(angles) > 2:
            for i in range(len(angles)):
                # check if angle is locally minimal (less than both adjacent angles)
                if angles[i] <= angles[min(i+1, len(angles)-1)] and angles[i] <= angles[max(0,i-1)]:
                    # then make a group with the vertices
                    # if one of the folds is on the edge, do not add it to the group
                    group = []
                    if not self.on_edge_fold(adj_folds[i]):
                        group.append(adj_folds[i])
                    if not self.on_edge_fold(adj_folds[min(i+1, len(angles)-1)]):
                        group.append(adj_folds[min(i+1, len(angles)-1)])
                    if len(group) == 1:
                        pairings.append((group, [2]))
                    if len(group) == 2:
                        pairings.append((group, [2,0]))
                    # remove the wedge by subtracting angles[i] from angles[i+1], and removing angles[i]
                    # then, remove the adjacent folds
                    angles[max(0,i-1)] = angles[max(0,i-1)] + angles[min(i+1, len(angles)-1)] - angles[i]
                    del angles[i]
                    del angles[min(i+1, len(angles)-1)]
                    del adj_folds[i]
                    del adj_folds[min(i,len(adj_folds)-1)]
                    break
        if len(angles) == 1:
            if not self.on_edge_fold(adj_folds[0]):
                # add the flag 2 to indicate this one fold has arbitrary assignment
                pairings.append(([adj_folds[0]], [2]))
        if len(angles) == 2:
            # check if both folds are on the edge
            if not self.on_edge_fold(adj_folds[0]) and not self.on_edge_fold(adj_folds[1]):
                # if not, add them to the pairings
                # add the flag 1 to indicate the second fold is the same mountain/valley assignment as the first
                pairings.append(([adj_folds[0], adj_folds[1]], [2, 1]))
        self.remove_fold(f1)
        self.remove_fold(f2)
        return pairings

    def assign_mv(self):
        # Run algorithm to assign mountain and valley folds
        # This is the algorithm from the paper The Complexity of Flat Origami by Bern and Hayes

        # first verify that the crease pattern has even degree
        if not self.even_degree():
            print("Crease pattern does not have even degree")
            return []

        groupings = []
        for v in self.vertices:
            if self.on_edge(v):
                groupings.append(self.get_pairings_on_edge(v))
            else:
                groupings.append(self.get_pairings(v))

        groupings = [item for sublist in groupings for item in sublist]
        # combine groups if we can form a chain by combining their folds
        # loop until no more groups can be combined
        combined = True
        while combined:
            combined = False
            for i in range(len(groupings)):
                for j in range(i+1, len(groupings)):
                    # check if the groups can be combined
                    # we may have to reverse the order of one of the groups
                    # so that the first fold of the first group is the last fold of the second group

                    # each grouping is a list of folds and a list of flags
                    # if flag[i] = 0, then the ith fold is mountain/valley opposite the previous fold
                    # if flag[i] = 1, then the ith fold is mountain/valley the same as the previous fold
                    # if flag[i] = 2, then the ith fold is arbitrary
                    # we combine groups and modify the flags accordingly
                    g1 = groupings[i][0]
                    g2 = groupings[j][0]
                    flag1 = groupings[i][1]
                    flag2 = groupings[j][1]
                    if g1[0] == g2[-1]:
                        # the end of g2 is the beginning of g1
                        # combine the groups and combine the flags
                        groupings[i] = (g2[:-1] + g1, flag2 + flag1[1:])
                        del groupings[j]
                        combined = True
                        break
                    elif g1[-1] == g2[0]:
                        # the end of g1 is the beginning of g2
                        # combine the groups
                        groupings[i] = (g1[:-1] + g2,  flag1 + flag2[1:])
                        del groupings[j]
                        combined = True
                        break
                    elif g1[0] == g2[0]:
                        # the beginning of g1 is the beginning of g2
                        # reverse the order of g2
                        g2.reverse()
                        flag2.reverse()
                        flag2 = [2] + flag2[:-1]
                        # combine the groups
                        groupings[i] = (g2[:-1] + g1,  flag2 + flag1[1:])
                        del groupings[j]
                        combined = True
                        break
                    elif g1[-1] == g2[-1]:
                        # reverse the order of g1
                        g1.reverse()
                        flag1.reverse()
                        flag1 = [2] + flag1[:-1]
                        # combine the groups
                        groupings[i] = (g2 + g1[1:],  flag2 + flag1[1:])
                        del groupings[j]
                        combined = True
                        break
        # print("check well formedness")
        # print([(self.vertices.index(f.v1), self.vertices.index(f.v2)) for f in self.folds])
        # print([([(self.vertices.index(f.v1), self.vertices.index(f.v2)) for f in g[0]], "parity", g[1]) for g in groupings])

        # I don't understand how there are duplicates in each group
        # But in anycase, remove them and their corresponding flag value
        for i in range(len(groupings)):
            g = groupings[i][0]
            flag = groupings[i][1]
            new_g = []
            new_flag = []
            for j in range(len(g)):
                if g[j] not in new_g:
                    new_g.append(g[j])
                    new_flag.append(flag[j])
            groupings[i] = (new_g, new_flag)

        # first, enumerate all lists of 1 or 0 of length len(groupings)
        choices  = list(product([0, 1], repeat=len(groupings)))

        # we can make a choice for each group by assigning the first fold arbitrarily
        # then, we can assign the rest of the folds based on the flags
        # enumerate all possible choices and check if every vertex not on an edge satisfies Maekawa's theorem
        # if so, return the choice
        # otherwise, return []

        # loop through the choices
        for k in range(len(choices)):
            choice = choices[k]
            for j in range(len(groupings)):
                g = groupings[j]
                g[0][0].type = choice[j]
                last = choice[j]
                for i in range(1, len(g[0])):
                    if g[1][i] == 0:
                        # opposite mountain/valley assignment as previous fold
                        last = 1 - last
                        g[0][i].type = last
                    elif g[1][i] == 1:
                        # same mountain/valley assignment as previous fold
                        # if previous fold is unassigned, assign based on choices
                        g[0][i].type = g[0][i-1].type
                    else:
                            g[0][i].type = -1

            # check if every vertex not on an edge satisfies Maekawa's theorem
            succeeded = True
            for v in self.none_edge_vertices():
                if not self.maekawa(v):
                    succeeded = False
            if succeeded:
                # return the choice
                print("Assignment Found")
                return groupings

        # if we get here, no choice worked
        print("No choice worked")
        return []

    def export_svg(self, filename):
        # export the square and crease pattern to an svg file
        # first, normalize the crease pattern
        self.scale(100)
        # get the coordinates of the square
        square_coords = []
        square_coords.append([0, 0])
        square_coords.append([self.side, 0])
        square_coords.append([self.side, self.side])
        square_coords.append([0, self.side])

        # start the svg file
        dwg = svgwrite.Drawing('square.svg', profile='tiny')

        # draw the square
        dwg.add(dwg.polygon(square_coords, fill='none', stroke='black'))

        # draw the crease pattern
        # for each fold, draw a line with blue for valley folds and red for mountain folds
        for f in self.folds:
            if f.type == 0:
                color = 'red'
            elif f.type == 1:
                color = 'blue'
            else:
                color = 'black'
            dwg.add(dwg.line((f.v1.x, f.v1.y), (f.v2.x, f.v2.y), stroke=color))

        # export the svg file with the given filename
        dwg.saveas(filename)