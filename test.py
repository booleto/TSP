import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

np.random.seed(42)  # Đảm bảo kết quả nhất quán

def createNodes(N, scale=100):
    """Tạo nút ngẫu nhiên"""
    # Tạo một danh sách các thành phố (nút) với tọa độ ngẫu nhiên
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
        assert self.bound > np.max(nodes), "Bounds không đủ to để chứa mọi node!"
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
        self.bbox = BBox(box)
        self.is_leaf = False
        self.portals = []
        self.children = None
        self.vertex = None


    def divide(node):
        """Tạo 4 nút con
        node: nút đang xét
        """
        node.children = [[TreeNode(), TreeNode()], [TreeNode(), TreeNode()]]
        for row in node.children:
            for child in row:
                child.parent = node
                child.bbox.bound = node.bbox.bound // 2
                child.level = node.level + 1
        
        b, _ = node.bbox.bound // 2
        node.children[0][0].bbox.pos = node.bbox.pos
        node.children[0][1].bbox.pos = node.bbox.pos + np.array((0, b))
        node.children[1][0].bbox.pos = node.bbox.pos + np.array((b, 0))
        node.children[1][1].bbox.pos = node.bbox.pos + np.array((b, b))

        for row in node.children:
            for child in row:
                child.vertex = node.vertex[[child.bbox.is_inside(m) for m in node.vertex]]
                if len(child.vertex) == 0:
                    continue
                if len(child.vertex) == 1:
                    child.is_leaf = True
                    continue
                TreeNode.divide(child)

    
    # def define_portals(node, m):


    def draw(self, ax):
        """Vẽ bbox và vertex. Làm tương tự với các nút con
        ax: Matplotlib axes
        """
        self.bbox.draw(ax)

        if self.is_leaf:
            for i in range(len(self.vertex)):
                plt.scatter(self.vertex[i, 0], self.vertex[i, 1], color='b')
                ax.annotate(self.level, (self.vertex[i, 0], self.vertex[i, 1]))
            return

        if self.children == None:
            return
        for row in self.children:
            for child in row:
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
    
    def draw(self, ax):
        """Vẽ HCN
        ax: Matplotlib axes
        """
        width, height = self.bound
        rect = plt.Rectangle(self.pos, width, height, edgecolor='k', facecolor='none', linewidth=0.7, linestyle="-")
        ax.add_patch(rect)



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


# def pertub(nodes, L_0, c):
#     '''Làm tròn tọa độ các nút để độ phức tạp của quadtree là nlog(n), rồi khử các nút trùng vị trí

#     nodes: danh sách các nút
#     L_0: độ dài cạnh hình vuông bao quanh mọi nút
#     epsilon: tham số. epsilon > 1/(len(nodes) ** (1/3))
#     d: độ dài max giữa 2 nút

#     Trả về: danh sách nút, sắp xếp tăng dần theo node[0, :]
#     '''
#     scaled_nodes = np.round(nodes / (L_0/(8*len(nodes)*c))) * 8
#     return np.unique(scaled_nodes, axis=0) + 1


epsilon = 0.5
L_0 = 256
d = L_0
N = 160

nodes = createNodes(N=N, scale=L_0)
pnodes = pertub(nodes, epsilon, L_0 * np.sqrt(2))
plt.scatter(pnodes[:, 0], pnodes[:, 1])
L = np.max(pnodes)

print("Node count post pertub: ", len(pnodes))

qtree = AroraQuadTree()
qtree.nodes = pnodes
qtree.bound = next_2power(np.max(pnodes))
qtree.build_tree()
qtree.draw(figsize=(8, 8))

# print((L_0/(8*N*c)))

    
# a = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [8, 9]])
# bb = BBox()
# bb.pos = np.array((0,0))
# bb.bound = np.array((2,2))
# print(bb.is_inside(a[1]))

# p1 = np.array((0, 1))
# p2 = np.array((4, 5))


# print([a.all(p1 < a)])







