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
    portals:
    """
    def __init__(self, box=(0,0,0,0)) -> None:
        self.level = 0
        self.parent = None
        self.children = None
        self.neighbors = []
        self.bbox = BBox(box)
        self.is_leaf = False
        self.portals = np.array([]).reshape((0, 2))
        self.local_idx = -1
        self.vertex = None
        self.directions = []
        self.solutions = {}


    def divide(node):
        """Tạo 4 nút con
        node: nút đang xét
        """
        node.children = [TreeNode(), TreeNode(), TreeNode(), TreeNode()]
        for child in node.children:
            child.parent = node
            child.bbox.bound = node.bbox.bound // 2
            child.level = node.level + 1
        
        b, _ = node.bbox.bound // 2
        node.children[0].bbox.pos = node.bbox.pos
        node.children[1].bbox.pos = node.bbox.pos + np.array((0, b))
        node.children[2].bbox.pos = node.bbox.pos + np.array((b, 0))
        node.children[3].bbox.pos = node.bbox.pos + np.array((b, b))

        for i in range(4):
            node.children[i].local_idx = i

        for child in node.children:
            child.vertex = node.vertex[[child.bbox.is_inside(m) for m in node.vertex]]
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
                    portals = TreeNode.calc_portals(node.children[i1].bbox.pos, node.children[i1].bbox.bound[0], dir, m=2)
                    # print(f"portals: {portals}")
                    # print(f"node portals: {node.children[i].portals}")

                    node.children[i1].portals = np.concatenate((node.children[i1].portals, portals), axis=0)
                    node.children[i2].portals = np.concatenate((node.children[i2].portals, portals), axis=0)
                    node.children[i1].neighbors.append(node.children[i2])
                    node.children[i2].neighbors.append(node.children[i1])
            
        

    def calc_portals(pos: np.array, bound: int, dir: np.array, m: int=1):
        '''Tính toán vị trí các portal
        '''
        bound_vec = np.array((bound, bound))
        corners = np.array(((-1, -1), (1, 1), (-1, 1), (1, -1))) # trong góc
        if np.any([np.all(np.equal(corner, dir)) for corner in corners]):
            dir = (dir + 1) / 2
            return [pos + np.multiply(bound_vec, dir)]
        
        m += 1
        if np.array_equiv(dir, np.array((0, 1))): # bên phải
            return [pos + np.array((a*(bound / m), bound)) for a in range(1, m)]
        
        if np.array_equiv(dir, np.array((1, 0))): # bên trên            
            return [pos + np.array((bound, a*(bound/m))) for a in range(1, m)]
        
        if np.array_equiv(dir, np.array((0, -1))): # bên trái
            return [pos + np.array((a*(bound / m), 0)) for a in range(1, m)]
        
        if np.array_equiv(dir, np.array((-1, 0))): # bên dưới
            return [pos + np.array((0, a*(bound / m))) for a in range(1, m)]


    
    def idx_to_vec(i1 : int, i2 : int):
        i1v = np.array((i1 // 2, i1 % 2))
        i2v = np.array(((i2 // 2, i2 % 2)))
        return i2v - i1v
    

    # def solve_subproblem(self, ins, out):



    def draw(self, ax):
        """Vẽ bbox và vertex. Làm tương tự với các nút con
        ax: Matplotlib axes
        """
        self.bbox.draw(ax) # vẽ bbox

        plt.scatter(self.portals[:, 0], self.portals[:, 1], marker='x', c='r') # vẽ portals 

        if self.is_leaf: # vẽ các điểm
            for i in range(len(self.vertex)):
                plt.scatter(self.vertex[i, 0], self.vertex[i, 1], color='b')
                ax.annotate(self.level, (self.vertex[i, 0], self.vertex[i, 1]))
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




ins = np.array((0, 3))
outs = np.array((4, 1))

tnode = TreeNode(box=(0, 0, 4, 4))
nodes = np.array(((1, 3), (3, 3), (3, 1)))
tnode.vertex = nodes
tnode.divide()



def solve_subproblem(tnode: TreeNode, ins, out):
    paths = [[ins]]
    solution = None

    starts = []
    ends = []
    travel_order = []
    min_path = 0
    for child in tnode.children:
        if len(child.vertex) == 0:
            continue
        if child.bbox.is_inside_inclusive(ins):
            starts.append(child)
        if child.bbox.is_inside_inclusive(out):
            ends.append(child)
        min_path += 1
    
    queue = [[st] for st in starts]

    while queue:
        current_path = queue.pop()
        next_poss = current_path[-1].neighbors
        for nextp in next_poss:
            if nextp in ends and len(current_path) == min_path-1 and nextp not in current_path:
                current_path.append(nextp)
                travel_order.append(current_path)
            if nextp not in current_path:
                queue.append(current_path + [nextp])

    return travel_order
    
    # for order in travel_order:
    #     viable_paths = [np.array(ins)]
    #     for i in range(len(order) - 1):
    #         dir = TreeNode.idx_to_vec(order[i].local_idx, order[i+1].local_idx)
    #         ports = TreeNode.calc_portals(order[i].bbox.pos, order[i].bbox.bound[0], dir, m=1)
    #         print(viable_paths)
    #         print(ports)
    #         viable_paths = add_to_path(viable_paths, ports)
    #     print(viable_paths)


def solve_paths(path, ins, out):
    viable_paths = np.array(ins)
    for i in range(len(path) - 1):
        dir = TreeNode.idx_to_vec(path[i].local_idx, path[i+1].local_idx)
        ports = TreeNode.calc_portals(path[i].bbox.pos, path[i].bbox.bound[0], dir, m=1)
        viable_paths = add_to_path(viable_paths, np.array(ports))
    viable_paths = add_to_path(viable_paths, np.array(out))

        
    #     viable_paths = add_to_path(viable_paths, np.array(ports))

    # viable_paths = add_to_path(viable_paths, out)
    return viable_paths


path = np.array([[(0, 1), (0, 2)], [(1, 1), (1, 2)]])
next = np.array(((2, 3), (2, 4)))

temppath = np.empty(0)
npath = []

def add_to_path(pathlist : np.ndarray, next : np.ndarray):
    npath = []

    if len(pathlist.shape) == 1:
        if len(next.shape) == 1:
            npath.append(pathlist, next.reshape(1, 2), axis=0)
            return npath

        for n in next[:, :]:
            npath.append
        
    if len(next.shape) == 1:
        for p in pathlist[:, :]:
            npath.append(np.concatenate((p, next.reshape(1, 2)), axis=0))
        return npath

    for p in pathlist[:, :]:
        for n in next[:, :]:
            npath.append(np.concatenate((p, n.reshape(1, 2)), axis=0))
    return npath

# def add_to_path(pathlist : np.ndarray, next : np.ndarray):
#     npath = []
#     for p in pathlist:
#         for n in next:
#             npath.append(np.concatenate((p, n), axis=0))
#     return np.array(npath)


# path = add_to_path(path, next)
# print(path)

travel_order = solve_subproblem(tnode, ins, outs)

print(travel_order[0])

ret = solve_paths(travel_order[0], ins, outs)
print(ret)


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