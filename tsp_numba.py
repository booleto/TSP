from numba import njit, prange
import numpy as np

np.random.seed(1707)

# ---------------------------------------------------------------------------- #
#                                    GLOBALS                                   #
# ---------------------------------------------------------------------------- #

# ----------------------------------- DEBUG ---------------------------------- #
DEBUG = False

# ----------------------------------- TRUTH ---------------------------------- #
FALSE = 0
TRUE = 1
NULL = -1


# ---------------------------------------------------------------------------- #
#                                NODES CREATION                                #
# ---------------------------------------------------------------------------- #

@njit
def randomNodes(shape : tuple[int], scale : int) -> np.ndarray:
    return np.random.random(shape) * scale

nodes = randomNodes((10, 2), scale=100)
# print(nodes)

@njit
def pertubate(nodes : np.ndarray) -> np.ndarray:
    for i in prange(nodes.shape[0]):
        for j in prange(nodes.shape[1]):
            nodes[i, j] = np.round(nodes[i, j]) * 2 + 1
    return nodes
    
# pertubate(nodes)
# print(nodes)



# ---------------------------------------------------------------------------- #
#                                   QUADTREE                                   #
# ---------------------------------------------------------------------------- #

# -------------------------------- TREE NODES -------------------------------- #
PARENT = 0
ACTIVE = 1
LEVEL = 2
CHILDREN = np.array((3, 4, 5, 6))
VERTEX_COUNT = 7
AABB = np.array((8, 9, 10))
VERTEX = 11

nodes_count = len(nodes)
tree = np.full((500, VERTEX + 1), NULL, dtype=np.int32)
tree_bound = 0


@njit
def divide(nodes, node_idx : np.ndarray, aabb):
    startx, starty, box_size = aabb
    box_size //= 2

    left = np.where(nodes[node_idx, 0] < startx + box_size)[0]
    right = np.where(nodes[node_idx, 0] > startx + box_size)[0]
    top = np.where(nodes[node_idx, 1] < starty + box_size)[0]
    bot = np.where(nodes[node_idx, 1] > starty + box_size)[0]

    topleft = np.intersect1d(top, left)
    topright = np.intersect1d(top, right)
    botleft = np.intersect1d(bot, left)
    botright = np.intersect1d(bot, right)

    return (topleft, topright, botleft, botright)


@njit
def divide_aabb(aabb):
    startx, starty, box_size = aabb
    box_size //= 2

    botleft_aabb = np.array((startx, starty, box_size))
    topleft_aabb =  np.array((startx + box_size, starty, box_size))
    botright_aabb = np.array((startx, starty + box_size, box_size))
    topright_aabb =  np.array((startx + box_size, starty + box_size, box_size))

    return (botleft_aabb, topleft_aabb, botright_aabb, topright_aabb)


@njit
def is_inside_aabb(p, aabb, inclusive=True):
    startx, starty, box_size = aabb
    aabb_start = np.array((startx, starty))
    aabb_end = aabb_start + np.array((box_size, box_size))

    if inclusive:
        return np.all(aabb_start <= p) and np.all(p <= aabb_end)
    else:
        return np.all(aabb_start < p) and np.all(p < aabb_end)
    

# point = np.array((0, 1))
# aabb = (0, 0, 2)
# print(is_inside_aabb(point, aabb, inclusive=True))

@njit
def is_on_edge_aabb(p, aabb):
    return is_inside_aabb(p, aabb, inclusive=True) \
            and not is_inside_aabb(p, aabb, inclusive=False)

# aabb = np.array((0, 0, 8))
# p = np.array((1, 1))
# print(f"is on edge: {is_on_edge_aabb(p, aabb)}")


@njit
def get_node_level(node_idx):
    level = 0
    level_end = 1
    level_start = 0

    while level_end <= node_idx:
        level_start = level_end
        level += 1
        level_end += 4 ** level

    return (level, level_start, level_end)


@njit
def get_children_idx(parent_idx):
    child = parent_idx * 4
    return (child + 1, child + 2, child + 3, child + 4)

@njit
def get_parent_idx(child_idx):
    return (child_idx - child_idx % 4) // 4

# p_idx = 9
# print(get_children_idx(p_idx))
# print(get_parent_idx(p_idx))

@njit
def search_next_power(number : int, power : int) -> int:
    ret = 1
    while ret < number:
        ret *= power
    return ret


@njit
def build_tree(tree, node_idx, aabb, vertices, vertices_idx):
    if DEBUG:
        print("======================")
        print(f"aabb: {aabb}")
        print(f"tree node: {tree[node_idx]}")
        print(f"")
    
    tree[node_idx][ACTIVE] = TRUE
    tree[node_idx][AABB] = aabb
    tree[node_idx][VERTEX_COUNT] = len(vertices_idx)

    if len(vertices_idx) == 1:
        tree[node_idx][VERTEX] = vertices_idx[0]
        return
    if len(vertices_idx) == 0:
        return

    child_idx = get_children_idx(node_idx)
    child_aabb = divide_aabb(aabb)
    child_vert_idx = divide(vertices, vertices_idx, aabb)

    if DEBUG:
        print(f"child idx: {child_idx}")
        print(f"child aabb: {child_aabb}")
        print(f"child vert idx: {child_vert_idx}")
        print("======================")
    
    tree[node_idx][CHILDREN] = child_idx
    for i in prange(len(child_idx)):
        tree[child_idx[i]][PARENT] = node_idx
        build_tree(tree, child_idx[i], child_aabb[i], vertices, child_vert_idx[i])

    return tree


# vertices = randomNodes((10, 2), 100)
# pertubate(vertices)
# vert_idx = np.array(range(len(vertices)))
# bound = search_next_power(np.max(vertices), 2)
# build_tree(tree, 0, (0, 0, bound), vertices, vert_idx)
# print(tree[:4**3])

@njit
def print_tree(tree):
    for i, node in enumerate(tree):
        if node[ACTIVE] != NULL:
            print(node)
        


# ---------------------------------------------------------------------------- #
#                                 PORTALIZATION                                #
# ---------------------------------------------------------------------------- #

RIGHT = 1
LEFT = 2
DOWN = 3
UP = 4

CORNER_TL = 5
CORNER_TR = 6
CORNER_BL = 7
CORNER_BR = 8


@njit
def get_portals(portallist, square):
    length = portallist[square][0]
    if length == 0:
        return None
    
    ret = portallist[square][1 : 2*length + 1]
    return ret.reshape((length, np.int32(2)))


# portallist = np.array(((1, 3, 4, NULL, NULL), (2, 3, 4, 5, 6)))
# port = get_portals(portallist, 0)
# port[0, 1] = 0
# print(portallist)
# print(get_portals(portallist, 0))

@njit
def get_square_portals(square_idx, portalref):
    length = portalref[square_idx][0]
    if length == 0:
        return None
    
    return portalref[square_idx][1 : length+1]

# portalref = np.array(((1, 1, NULL, NULL), (3, 2, 3, 4)))
# print(f"PORTALREF: {get_square_portals(0, portalref)}")


@njit
def calc_corner_portals(orientation_index, aabb):
    startx, starty, bound = aabb
    pos = np.array((startx, starty))

    if orientation_index == CORNER_TL:
        ret = np.add(pos, np.array((bound, 0)))
        return ret
    
    if orientation_index == CORNER_TR:
        ret = np.add(pos, np.array((bound, bound)))
        return ret
    
    if orientation_index == CORNER_BL:
        return pos
    
    if orientation_index == CORNER_BR:
        ret = np.add(pos, np.array((0, bound)))
        return ret


@njit
def calc_line_portals(orientation_index, aabb, num_portals):
    startx, starty, bound = aabb
    pos = np.array((startx, starty))
    m = num_portals + 1
    ret = None
    increment = np.arange(1, m) * bound / m

    if orientation_index == RIGHT: # right
        pos = np.add(pos, np.array((0, bound)))
        incr = np.vstack((increment, np.zeros(m - 1))).T
        ret = np.add(pos, incr)
        return ret

    if orientation_index == LEFT: # left
        pos = np.add(pos, np.array((0, 0)))
        incr = np.vstack((increment, np.zeros(m - 1))).T
        ret = np.add(pos, incr)
        return ret

    if orientation_index == DOWN: # down
        pos = np.add(pos, np.array((0, 0)))
        incr = np.vstack((np.zeros(m - 1), increment)).T
        ret = np.add(pos, incr)
        return ret
    
    if orientation_index == UP: # up
        pos = np.add(pos, np.array((bound, 0)))
        incr = np.vstack((np.zeros(m - 1), increment)).T
        ret = np.add(pos, incr)
        return ret


# aabb = (0, 0, 8)
# print(calc_line_portals(UP, aabb, 1))
# print(calc_corner_portals(CORNER_TR, aabb))


@njit
def calc_inner_portals(aabb, num_portals):
    inner_portals = np.empty((4*num_portals + 1, 2))

    botleft, _, _, topright = divide_aabb(aabb)
    inner_portals[:num_portals, :] = calc_line_portals(UP, botleft, num_portals)
    inner_portals[num_portals : 2*num_portals, :] = calc_line_portals(RIGHT, botleft, num_portals)
    inner_portals[2*num_portals : 3*num_portals, :] = calc_line_portals(LEFT, topright, num_portals)
    inner_portals[3*num_portals : 4*num_portals, :] = calc_line_portals(DOWN, topright, num_portals)
    inner_portals[4*num_portals] = calc_corner_portals(CORNER_TR, botleft)

    return inner_portals


# aabb = (0, 0, 16)
# print(calc_inner_portals(aabb, 3))

@njit
def build_portallist(tree, portallist, num_portals):
    """
    """
    portal_index = 0
    for index, node in enumerate(tree):
        if node[ACTIVE] == NULL:
            continue
        if node[VERTEX_COUNT] <= 1:
            continue
        
        inner_portals = calc_inner_portals(node[AABB], num_portals)
        index_end = len(inner_portals) + portal_index
        portallist[portal_index : index_end] = inner_portals

        if DEBUG:
            print(f"index range: {index_end - portal_index}")
            print(f"portal count: {len(inner_portals)}")
            print("----------------------------------")



# @njit
def build_portalref(tree, portallist, portalref):
    for index, node in enumerate(tree):
        if node [ACTIVE] == NULL:
            continue
        aabb = node[AABB]
        portals = np.full(len(portallist), NULL, dtype=np.int32)
        idx = 0
        for i, port in enumerate(portallist):
            if is_on_edge_aabb(port, aabb):
                portals[idx] = i
                idx += 1
        
        if idx == 0:
            continue
        
        print(f"FOUND PORTALS: {portals[:idx]}")
        start_idx = portalref[index][0] if portalref[index][0] > 0 else 1
        end_idx = start_idx + idx
        print(start_idx, end_idx, sep=" ")
        portalref[index][0] = end_idx
        portalref[index][start_idx: end_idx] = portals[:idx].copy()

        # portalref[index][0] = idx
        # portalref[index][1: idx+1] = portals[:idx].copy()
        print(f"PORTALREF: {portalref[index][1: end_idx]}")
        

vertices = np.array(((1, 1), (7, 7)))
vert_idx = np.array(range(len(vertices)))
bound = search_next_power(np.max(vertices), 2)
portal_count = 3

print(f"bound: {bound}")

build_tree(tree, 0, (0, 0, bound), vertices, vert_idx)

print_tree(tree)
portalref = np.full((len(tree), 4* portal_count + 1), NULL, dtype=np.int32)
portallist = np.full((30, 2), NULL, dtype=np.int32)
portalref[:, 0] = 0

build_portallist(tree, portallist, portal_count)
build_portalref(tree, portallist, portalref)

print(f"portallist: {portallist}")
print(f"portalref: {portalref}")

# from numba.typed import Sets

# @njit
# def get_common_portals(portallist, square1, square2):
#     s1_ports = get_portals(portallist, square1)
#     s2_ports = get_portals(portallist, square2)
#     s1_set = set(s1_ports)
#     s2_set = set(s2_ports)
#     # s2_ports = ((s2_ports[i][0], s2_ports[i][1]) for i in range(len(s2_ports)))

#     intersect = np.array(s1_set & s2_set)

#     return intersect

# print(get_common_portals(portallist, 0, 1))


# def intersect2D(a, b):
#     return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

@njit
def intersect2D(set1, set2):
    mask = np.array([False for i in set1], dtype=np.bool_)
    for i, elem in enumerate(set1):
        for j in range(len(set2)):
            if (elem == set2[j]).all():
                mask[i] = True

    idx = np.argwhere(mask == True).T[0]
    return idx


# s1 = np.array(((1, 0), (2, 1), (1, 1)))
# s2 = np.array(((1, 0), (1, 1), (1, 2)))
# print(intersect2D(s1, s2))



@njit
def get_subset(index, num_elements):
    index *= 3
    subset = np.zeros(num_elements, dtype=np.bool_)
    for j in range(num_elements):
        if (index >> j) & 1:
            subset[j] = True
    return subset


@njit
def count_subset(num_elements):
    count = 0
    while not np.all(get_subset(count, num_elements)):
       count += 1
    return count



# ---------------------------------------------------------------------------- #
#                              DYNAMIC PROGRAMMING                             #
# ---------------------------------------------------------------------------- #

@njit
def catalan(n):
    ret = 1
    for i in range(2, n):
        ret *= (i + n) / i
    return ret

@njit
def factorial(n):
    ret = 1
    for i in prange(2, n + 1):
        ret *= i
    return ret

@njit
def nCk(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))

@njit
def narayana(n, k):
    return int(nCk(n, k) * nCk(n, k-1) / n)


@njit
def calc_subproblem_space(max_portal_count : int):
    ret = 0
    for i in range(2, max_portal_count+1, 2):
        ret += narayana(max_portal_count, i)
    return int(ret)

print(calc_subproblem_space(6))

    


# @njit
# def calc_interface(mask, used)


# def get_subproblem_index(portalmask : np.ndarray):
#     for count in range(start=2, stop=len(portalmask), step=2):
#         for 



# def get_pairing_index():


# @njit
# def build_tree(node_idx, tree, aabb):
#     tree_length = 0
#     current_idx = 0
#     stack = [(node_idx, NULL, aabb)]

#     while current_idx <= tree_length:
#         current_node, parent, aabb = stack.pop()
#         tree[current_idx][ACTIVE] = TRUE
#         tree[current_idx][VERTEX_COUNT] = len(current_node)
#         tree[current_idx][PARENT] = parent

#         if len(current_node) == 0:
#             current_idx += 1
#             continue
#         elif len(current_node) == 1:
#             current_idx += 1
#             continue    
#         else:
#             tl, tr, bl, br = divide(node_idx, aabb)
#             tl_aabb, tr_aabb, bl_aabb, br_aabb = divide_aabb(aabb)

#             stack.append((tl, current_idx, tl_aabb))
#             stack.append((tr, current_idx, tr_aabb))
#             stack.append((bl, current_idx, bl_aabb))
#             stack.append((br, current_idx, br_aabb))

#             current_idx += 1
            


# @njit
# def register_node(node_idx, tree, tree_idx, parent, aabb):
#     treenode = np.full(VERTEX)
#     treenode[VERTEX]

#     if len(node_idx) <= 1:
#         treenode = np.empty(VERTEX)

#         treenode[PARENT] = parent
#         treenode[VERTEX_COUNT] = len(node_idx)
#         for CHILD in CHILDREN:
#             treenode[CHILD] = NULL

#         if len(node_idx == 0):
#             treenode[VERTEX] = NULL
#         else:
#             treenode[VERTEX] = node_idx

#         tree[tree_idx] = treenode
#         tree_idx += 1

#     else:
#         treenode = np.empty(VERTEX)
#         treenode[PARENT] = parent
#         treenode[VERTEX] = NULL
#         treenode[VERTEX_COUNT] = len(node_idx)
#         tree[tree_idx] = treenode

#         startx, starty, bound = aabb
#         children = divide(node_idx, (startx, starty, bound))
#         for child in children:
#             register_tree(child, tree, tree_idx, parent, )
            

# n = np.array([0])
# register_tree(n, tree, tree_bound, NULL)
# print(tree)