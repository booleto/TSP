import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

np.random.seed(42)  # Đảm bảo kết quả nhất quán


def createNodes(N, scale=100):
    """Tạo nút ngẫu nhiên"""
    return np.random.rand(N, 2) * scale


class AroraQuadTree:
    """Quadtree được sử dụng trong Arora PTAS
    
    nodes: Các điểm cần đi qua trong bài toán TSP
    root: Gốc của cây, dạng TreeNode
    bounds: độ to của bounding box. bounds > max(nodes)

    """
    def __init__(self) -> None:
        """Khởi tạo các giá trị mặc định cho cây
        """
        self.nodes: np.ndarray = []
        self.root : TreeNode = TreeNode()
        self.levels = 0
        self.bound = 0

    
    # def solve_tree(self):
    #     node = self.root
    #     for child in self.children:
    #         if len(child.vertex) == 0:
    #             continue
    #         if len(child)
            


    def build_tree(self):
        """Dựng quadtree, dựa trên nodes
        Cần phải nhập nodes và bounds trước khi gọi hàm này
        """
        assert self.bound != 0, "Cần nhập bound của cây trước!"
        assert self.bound > np.max(self.nodes), "Bounds không đủ to để chứa mọi node!"
        assert len(self.nodes), "Cần nhập danh sách node cho cây!"
        self.root.vertex = self.nodes
        self.root.bbox.bound = np.array((self.bound, self.bound))
        TreeNode.divide(self.root)


    def draw(self, figsize=(10, 10)):
        """Vẽ quadtree
        figsize: Độ lớn của hình
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.root.draw(ax)
        plt.show()


class TreeNode:
    """Nút của quadtree

    level: Độ sâu trên cây
    parent: Nút cha
    children: Các nút con
    is_leaf: True nếu là nút lá và vertex != None

    bbox: HCN bao quanh
    vertex: Các điểm nằm trong bbox
    portals: Các portal nằm trên ô
    
    """
    def __init__(self, box=(0,0,0,0)) -> None:
        self.level = 0
        self.parent = None
        self.children = None
        self.neighbors = []
        self.bbox = BBox(box)
        self.is_leaf = False
        self.portals = np.array([]).reshape((0, 2))
        self.portal_cache = {}
        self.local_idx = -1
        self.vertex = []
        self.directions = []
        self.solutions = {}
        self.m = 1


    def divide(node):
        """Tạo 4 nút con
        node: nút đang xét
        """
        node.children = [TreeNode(), TreeNode(), TreeNode(), TreeNode()]
        for child in node.children:
            child.parent = node
            child.bbox.bound = node.bbox.bound // 2
            child.level = node.level + 1
            child.m = node.m
        
        b, _ = node.bbox.bound // 2
        node.children[0].bbox.pos = node.bbox.pos
        node.children[1].bbox.pos = node.bbox.pos + np.array((0, b))
        node.children[2].bbox.pos = node.bbox.pos + np.array((b, 0))
        node.children[3].bbox.pos = node.bbox.pos + np.array((b, b))

        for i in range(4):
            node.children[i].local_idx = i

        for child in node.children:
            child.vertex = node.vertex[[child.bbox.is_inside(v) for v in node.vertex]]
            if len(child.vertex) == 0:
                continue
            if len(child.vertex) == 1:
                child.is_leaf = True
                continue
            TreeNode.divide(child)

        for i1 in range(4): # Thêm portals
            for i2 in range(i1+1, 4):
                if len(node.children[i1].vertex) > 0 and len(node.children[i2].vertex) > 0:
                    dir = TreeNode.idx_to_vec(i1, i2)
                    node.children[i1].directions.append(dir)
                    node.children[i2].directions.append(- dir)
                    portals = node.children[i1].calc_portals(dir, m=node.m)
                    # print(f"portals: {portals}")
                    # print(f"node portals: {node.children[i].portals}")

                    node.children[i1].portals = np.concatenate((node.children[i1].portals, portals), axis=0)
                    node.children[i2].portals = np.concatenate((node.children[i2].portals, portals), axis=0)
                    node.children[i1].neighbors.append(node.children[i2])
                    node.children[i2].neighbors.append(node.children[i1])
            
        

    def calc_portals(self, dir: np.array, m: int=1):
        '''Tính toán vị trí các portal
        '''
        if dir.tobytes() in self.portal_cache:
            return self.portal_cache[dir.tobytes()]

        bound = self.bbox.bound[0]
        pos = self.bbox.pos
        bound_vec = np.array((bound, bound))
        corners = np.array(((-1, -1), (1, 1), (-1, 1), (1, -1))) # trong góc
        if np.any([np.all(np.equal(corner, dir)) for corner in corners]):
            dir = (dir + 1) / 2
            ret = [pos + np.multiply(bound_vec, dir)]
            self.portal_cache[dir.tobytes()] = ret
            return ret
        
        m += 1
        if np.array_equiv(dir, np.array((0, 1))): # bên phải
            ret = [pos + np.array((a*(bound / m), bound)) for a in range(1, m)]
            self.portal_cache[dir.tobytes()] = ret
            return ret
        
        if np.array_equiv(dir, np.array((1, 0))): # bên trên
            ret = [pos + np.array((bound, a*(bound/m))) for a in range(1, m)]
            self.portal_cache[dir.tobytes()] = ret
            return ret
        
        if np.array_equiv(dir, np.array((0, -1))): # bên trái
            ret = [pos + np.array((a*(bound / m), 0)) for a in range(1, m)]
            self.portal_cache[dir.tobytes()] = ret 
            return ret
        
        if np.array_equiv(dir, np.array((-1, 0))): # bên dưới
            ret = [pos + np.array((0, a*(bound / m))) for a in range(1, m)]
            self.portal_cache[dir.tobytes()] = ret
            return ret


    
    def idx_to_vec(i1 : int, i2 : int):
        """Tính vector đi từ local index i1 đến i2
        i1 và i2 cần có chung node mẹ

        i1: int, local index
        i2: int, local index
        """
        i1v = np.array((i1 // 2, i1 % 2))
        i2v = np.array(((i2 // 2, i2 % 2)))
        return i2v - i1v
    

    def solve_leaf(self, ins, out):
        if len(self.vertex) == 1:
            cost = np.linalg.norm(ins - self.vertex[0]) + np.linalg.norm(self.vertex[0] - out)
            # plt.plot((ins[0], self.vertex[0][0]), (ins[1], self.vertex[0][1]), color='g', alpha=1)
            # plt.plot((out[0], self.vertex[0][0]), (out[1], self.vertex[0][1]), color='g', alpha=1)
            return (cost, [self.vertex[0]])
        else:
            # plt.plot((ins[0], out[0]), (ins[1], out[1]), color='g', alpha=1)
            return (np.linalg.norm(ins - out), [])


    def solve_subproblem(self, ins, out):
        key = np.concatenate((ins, out)).tobytes()
        if key in self.solutions:
            return self.solutions[key]

        if len(self.vertex) <= 1:
            return self.solve_leaf(ins, out)

        starts = []
        ends = []
        travel_order = []
        min_path = 0
        for child in self.children:
            # if len(child.vertex) == 0:
            #     continue
            if child.bbox.is_inside_inclusive(ins):
                starts.append(child)
            if child.bbox.is_inside_inclusive(out):
                ends.append(child)
            if len(child.vertex) != 0:
                min_path += 1
        

        queue = [([st], 1) if len(st.vertex) != 0 else ([st], 0) for st in starts]

        # tìm các đường đi khả thi
        while queue:
            current_path, node_count = queue.pop()
            print(f"path: {[node.bbox.pos for node in current_path]}")
            print(f"count: {node_count}")
            next_poss = [child for child in self.children if child not in current_path]

            for nextp in next_poss:
                if nextp in ends:
                    print(f"end: {nextp.bbox.pos}")
                    if node_count == min_path or (node_count == min_path-1 and len(nextp.vertex) != 0):
                        current_path.append(nextp)
                        travel_order.append(current_path)
                        continue

                if len(current_path) < 4:
                    next_count = node_count
                    if len(nextp.vertex) != 0:
                        next_count += 1
                    queue.append((current_path + [nextp], next_count))
            print("-----------------------------")

        # print(f"travel order: {[[node.bbox.pos for node in tour] for tour in travel_order]}")

        # for tour in travel_order:
        #     print(f"tour: {[node.bbox.pos for node in tour]}")

    
        # while queue:
        #     current_path = queue.pop()
        #     next_poss = current_path[-1].neighbors
        #     for nextp in next_poss:
        #         if nextp in ends and len(current_path) == min_path-1 and nextp not in current_path:
        #             current_path.append(nextp)
        #             travel_order.append(current_path)
        #         if nextp not in current_path and len(current_path) < 4:
        #             queue.append(current_path + [nextp])
        # else:
        #     while queue:
        #         current_path = queue.pop()
        #         next_poss = current_path[-1].neighbors
        #         for nextp in next_poss:
        #             if len(current_path) == min_path-1 and nextp not in current_path:
        #                 current_path.append(nextp)
        #                 travel_order.append(current_path)
        #             if nextp not in current_path:
        #                 queue.append(current_path + [nextp])

        
        min_tour = None # tìm đường đi tốt nhất
        min_tour_cost = np.iinfo(np.int64).max # Max value
        for order in travel_order:
            print(f"node order: {[node.vertex for node in order]}")
            portal_tours = TreeNode.get_portal_tour(order, ins, out)
            
            for t, port_tour in enumerate(portal_tours):
                tour_cost = 0
                current_tour = []

                for i, node in enumerate(order):    
                    cost, subtour = node.solve_subproblem(port_tour[i], port_tour[i+1])
                    tour_cost += cost
                    current_tour = current_tour + subtour
                
                if tour_cost < min_tour_cost:
                    min_tour_cost = tour_cost
                    min_tour = current_tour
        
        key = np.concatenate((ins, out)).tobytes()
        self.solutions[key] = (min_tour_cost, min_tour)
        return (min_tour_cost, min_tour)

    
    def get_portal_tour(path, ins, out):
        """Tính danh sách các portal có thể đi qua
        path: các node cần đi qua (theo thứ tự)
        ins: portal khởi đầu
        out: portal kết thúc
        """
        viable_paths = np.array(ins)
        for i in range(len(path) - 1):
            dir = TreeNode.idx_to_vec(path[i].local_idx, path[i+1].local_idx)
            ports = path[i].calc_portals(dir, m=path[i].m)
            viable_paths = add_to_path(viable_paths, np.array(ports))
        viable_paths = add_to_path(viable_paths, np.array(out))

        return viable_paths


    def draw(self, ax):
        """Vẽ bbox và vertex. Làm tương tự với các nút con
        ax: Matplotlib axes
        """
        self.bbox.draw(ax) # vẽ bbox

        plt.scatter(self.portals[:, 0], self.portals[:, 1], marker='x', c='r') # vẽ portals 

        if self.is_leaf: # vẽ các điểm
            for i in range(len(self.vertex)):
                plt.scatter(self.vertex[i, 0], self.vertex[i, 1], color='b')
                # ax.annotate(self.level, (self.vertex[i, 0], self.vertex[i, 1]))
                ax.annotate(f"{(self.vertex[i, 0], self.vertex[i, 1])}", (self.vertex[i, 0], self.vertex[i, 1]))
            return

        if self.children == None:
            return
        for child in self.children:
            child.draw(ax)



class BBox:
    """Hình chữ nhật trong không gian 2D"""
    def __init__(self, box=(0,0,0,0)) -> None:
        """Khởi tạo hình chữ nhật
        box: vị trí và kích cỡ hcn
        """
        self.pos = np.array((box[0], box[1]))
        self.bound = np.array((box[2], box[3]))


    def is_inside(self, p) -> bool:
        """Kiểm tra điểm p có nằm trong hcn không
        Nếu p ở trên cạnh trên hoặc cạnh trái => False
        Nếu ở cạnh dưới hoặc cạnh phải => True

        p: Tọa độ điểm cần kiểm tra
        """
        return np.all(self.pos < p) and np.all(p <= self.pos + self.bound)
    

    def is_inside_inclusive(self, p) -> bool:
        """Kiểm tra điểm p có nằm trong hcn không
        Nếu p ở trên cạnh hcn => True

        p: Tọa độ điểm cần kiểm tra
        """
        return np.all(self.pos <= p) and np.all(p <= self.pos + self.bound)
    

    def is_inside_exclusive(self, p) -> bool:
        """Kiểm tra điểm p có nằm trong hcn không
        Nếu p ở trên cạnh hcn => False

        p: Tọa độ điểm cần kiểm tra
        """
        return np.all(self.pos <= p) and np.all(p <= self.pos + self.bound)
    

    def center(self) -> np.ndarray:
        """Tâm của hình chữ nhật"""
        return (self.pos + self.bound / 2)
    

    def draw(self, ax):
        """Vẽ HCN
        ax: Matplotlib axes
        """
        width, height = self.bound
        rect = plt.Rectangle(self.pos, width, height, edgecolor='k', facecolor='none', linewidth=0.7, linestyle="-")
        ax.add_patch(rect)


# class PortalSet:
#     def __init__(self) -> None:
#         self.portals = []

#     def g


def next_2power(num):
    '''Tìm số n = k**2 > num, gần num nhất'''
    ret = 1
    while ret < num:
        ret *= 2
    return ret

def pertub(nodes, epsilon, d):
    '''Làm tròn tọa độ các nút để độ phức tạp của quadtree là nlog(n), rồi khử các nút trùng vị trí

    nodes: danh sách các nút
    L_0: độ dài cạnh hình vuông bao quanh mọi nút
    epsilon: tham số. epsilon > 1/(len(nodes) ** (1/3))
    d: độ dài max giữa 2 nút

    Trả về: danh sách nút, sắp xếp tăng dần theo node[0, :]
    '''
    scaled_nodes = np.round(nodes / (epsilon * d/(len(nodes)**1.5)))
    return np.unique(scaled_nodes, axis=0) * 2 + 1


# quadtree demo
# epsilon = 1.3
# L_0 = 1000
# d = L_0
# N = 60
# r = 2

# nodes = createNodes(N=N, scale=L_0)
# pnodes = pertub(nodes, epsilon, L_0 * np.sqrt(2))
# plt.scatter(pnodes[:, 0], pnodes[:, 1])
# L = np.max(pnodes)

# print("Lượng node sau khi làm tròn: ", len(pnodes))

# qtree = AroraQuadTree()
# qtree.nodes = pnodes
# qtree.bound = next_2power(np.max(pnodes))
# qtree.build_tree()
# qtree.draw(figsize=(7, 7))




# def solve_subproblem(tnode: TreeNode, ins, out):
#     paths = [[ins]]
#     solution = None

#     starts = []
#     ends = []
#     travel_order = []
#     min_path = 0
#     for child in tnode.children:
#         if len(child.vertex) == 0:
#             continue
#         if child.bbox.is_inside_inclusive(ins):
#             starts.append(child)
#         if child.bbox.is_inside_inclusive(out):
#             ends.append(child)
#         min_path += 1
    
#     queue = [[st] for st in starts]

#     while queue:
#         current_path = queue.pop()
#         next_poss = current_path[-1].neighbors
#         for nextp in next_poss:
#             if nextp in ends and len(current_path) == min_path-1 and nextp not in current_path:
#                 current_path.append(nextp)
#                 travel_order.append(current_path)
#             if nextp not in current_path:
#                 queue.append(current_path + [nextp])

#     return travel_order

    


    # for order in travel_order:
    #     viable_paths = [np.array(ins)]
    #     for i in range(len(order) - 1):
    #         dir = TreeNode.idx_to_vec(order[i].local_idx, order[i+1].local_idx)
    #         ports = TreeNode.calc_portals(order[i].bbox.pos, order[i].bbox.bound[0], dir, m=1)
    #         print(viable_paths)
    #         print(ports)
    #         viable_paths = add_to_path(viable_paths, ports)
    #     print(viable_paths)




# def get_portal_tour(path, ins, out):
#     """Tính danh sách các portal có thể đi qua
#     path: các node cần đi qua (theo thứ tự)
#     ins: portal khởi đầu
#     out: portal kết thúc
#     """
#     viable_paths = np.array(ins)
#     for i in range(len(path) - 1):
#         dir = TreeNode.idx_to_vec(path[i].local_idx, path[i+1].local_idx)
#         ports = TreeNode.calc_portals(path[i].bbox.pos, path[i].bbox.bound[0], dir, m=1)
#         viable_paths = add_to_path(viable_paths, np.array(ports))
#     viable_paths = add_to_path(viable_paths, np.array(out))

#     return viable_paths


path = np.array([[(0, 1), (0, 2)], [(1, 1), (1, 2)]])
next = np.array(((2, 3), (2, 4)))

temppath = np.empty(0)
npath = []

def add_to_path(pathlist : np.ndarray, next : np.ndarray):
    """Thêm các nút (hoặc portal) vào đường đi
    pathlist: các đường đi hiện có
    next: các điểm có thể đến tiếp theo
    """
    npath = []
    # print(f"pathlist: {pathlist}")
    # print(f"next: {next}")

    if len(pathlist.shape) == 1:
        if len(next.shape) == 1:
            npath.append(np.concatenate((pathlist.reshape(1, 2), next.reshape(1, 2)), axis=0))
            return np.array(npath)

        for n in next[:, :]:
            npath.append(np.concatenate((pathlist.reshape(1, 2), n.reshape(1, 2)), axis=0))
            return np.array(npath)
        
    if len(next.shape) == 1:
        for p in pathlist[:, :]:
            npath.append(np.concatenate((p, next.reshape(1, 2)), axis=0))
        return np.array(npath)

    for p in pathlist[:, :]:
        for n in next[:, :]:
            npath.append(np.concatenate((p, n.reshape(1, 2)), axis=0))
    return np.array(npath)



ins = np.array((0, 16))
outs = np.array((12, 0))

tnode = TreeNode(box=(0, 0, 16, 16))
# nodes = np.array(((1.1, 3.1), (3.1, 2.6), (3.1, 1.1), (1.1, 1.4), (1.1, 1.1)))
nodes = np.array(((3, 1), (3, 3), (9, 3), (1, 3), (5, 7)))
tnode.vertex = nodes
tnode.divide()


fig, ax = plt.subplots(figsize=(7, 7))
tnode.draw(ax)
plt.scatter(ins[0], ins[1], c='g', marker='x', s=1000, linewidths=5)
plt.scatter(outs[0], outs[1], c='g', marker='x', s=1000, linewidths=5)
# plt.show()

# path = add_to_path(path, next)
# print(path)

cost, tour = tnode.solve_subproblem(ins, outs)
print(f"cost: {cost}")
print(f"tour: {tour}")
# plt.show()

# def plot_tour(tour, porttour):
#     plt.plot((porttour[0][0], tour[0][0]), (porttour[0][1], tour[0][1]), color='b')
#     for i in range(len(tour) - 1):
#         plt.plot((porttour[i*2][0], tour[i][0]), (porttour[i*2][1], tour[0][1]), color='b')
#         plt.plot((porttour[i*2 + 1][0], tour[i][0]), (porttour[i*2 + 1][1], tour[i][1]), color='b')
#         # plt.plot((tour[i][0], tour[i+1][0]), (tour[i][1], tour[i + 1][1]), color='b')

#     plt.plot((porttour[-2][0], tour[-1][0]), (porttour[-1][1], tour[-1][1]), color='b')
#     plt.show()

plt.plot((ins[0], tour[0][0]), (ins[1], tour[0][1]), color='b')
for i in range(len(tour) - 1):
    plt.plot((tour[i][0], tour[i+1][0]), (tour[i][1], tour[i + 1][1]), color='b')
plt.plot((tour[-1][0], outs[0]), (tour[-1][1], outs[1]), color='b')
plt.show()

# for i in range(len(tour)):
#     for j in range(len(tour[i])):
#         print(tour[i][j][0])
#         plt.plot((tour[i][j][0], tour[i][j][0]), (tour[i][j][1], tour[i][j][1]), color='b')



# plot_tour(tour, porttour)
# for order in travel_order:
#     ret = get_portal_tour(order, ins, outs)
#     print(ret)


# i am slowly losing it

# torder = solve_subproblem(tnode=tnode, ins=ins, out=outs)

# for path in torder:
#     print(" - ")
#     for node in path:
#         print(node.vertex, end=" ")
#     print()

# port1 = np.array(((0, 1), (1, 2)))
# port2 = np.array(((0, 1), (2, 2)))
# print(np.intersect1d(port1, port2)) 