# coding:utf-8
from collections import defaultdict
import numpy as np

class Graph:

    def __init__(self, d_array):
        """
        构造函数
        :param d_array: 整数度数列列表
        """
        self.d_array = d_array  # 度数列
        self.V = len(d_array)  # 顶点个数
        self.graph = defaultdict(list)  # 存储顶点间的关系v-u
        self.matrix = []  # 存储相邻矩阵

        if self.isGraph():
            if self.isSimpleGraph():
                self.makeAdjacentMatrix()
                if self.isEulerGraph():
                    self.findEulerCiruit()

    def isGraph(self):
        """
        该函数用于判断非负整数数列是否为可图化
        :return: True 可图化
                 False 不可图化
        """
        for num in self.d_array:
            if num < 0:
                print("请输入非负整数序列！", end="")
                return False
        if not sum(self.d_array) % 2:
            print("可图化的")
            return True
        else:
            print("不可图化的")
            return False
        # return not (sum(self.d_array) % 2)

    def isSimpleGraph(self):
        """
        该函数用于判断是否为简单图，采用Havel定理
        :return: True 可简单图化
                 False 不可简单图化
        """
        li = self.d_array.copy()
        while True:
            li.sort(reverse=True)
            maxd = max(li)
            n = len(li)

            if (0 <= maxd <= n - 1) and (sum(li) % 2 == 0):
                # print(li)
                index = li.pop(0)
                for i in range(0, index):
                    li[i] -= 1

                for num in li:
                    if num < 0:
                        print("不是简单图")
                        return False

                if (len(li) == 0):
                    print("是简单图")
                    return True
            else:
                print("不是简单图")
                return False

    def makeAdjacentMatrix(self):
        """
        该函数用于生成简单图的相邻矩阵
        :return: numpy.ndarray AdjacentMatrix
        """
        li = self.d_array.copy()
        li_copy = li.copy()
        new_li = []
        new_li.append(li.copy())

        # 生成 顶点数/2 个 列表，模拟Havel定理过程。
        # 但每一轮轮减1后不重新排序，将要删去的数字变为0。
        # 每一轮获得的新列表储存new_li
        j = 0
        n = 0
        li_len = len(li)
        half_len = int(len(li) / 2)

        while n < half_len:
            t_li = []
            for i in range(j + 1, len(li)):
                t_li.append([li[i], i])

            t_li.sort(key=self.takeFirst, reverse=True)

            for elem in t_li[0:li[j]]:
                if elem[0] != 0:
                    li[elem[1]] = elem[0] - 1

            li[j] = 0
            # print(li)
            new_li.append(li.copy())
            j += 1
            n += 1

        # 开始生成相邻矩阵
        # 根据上一步获得的列表,纵向比较两次列表的每一位数
        # 如果两行数相差1，则在新列表填1
        yuan = np.array(new_li)
        ju_zhen = np.zeros([li_len, li_len], dtype=int)

        for row in range(half_len):
            for line in range(li_len):
                if yuan[row][line] - yuan[row + 1][line] == 1:
                    ju_zhen[row][line] = 1
                else:
                    ju_zhen[row][line] = 0
                if row == line:  # 根据性质相邻矩阵行和列相等处为0
                    ju_zhen[row][line] = 0

        # 因为相邻矩阵是斜对称的，所以只用比较1/2行数
        # 根据已经确定的行数，和对应行（顶点）对应的度数生成另一半矩阵
        for j in range(li_len):
            for i in range(li_len):
                ju_zhen[i][j] = ju_zhen[j][i]

        for row in range(half_len, li_len - 1):
            sum_d = 0
            for line in range(li_len):
                sum_d += ju_zhen[row][line]
            if sum_d == new_li[0][row]:
                pass
            else:
                ju_zhen[row][row + 1] = 1
                ju_zhen[row + 1][row] = 1

        print(ju_zhen)
        self.matrix = ju_zhen
        return ju_zhen

    def isConnect(self):
        """
        该函数用于判断简单图是否为连通图
        :return: True 连通图
                 False 不是连通图
        """
        # 算法思路是任意第n行的前n个数和第n列的前n个数相加
        # 如果和为0就是不连通的。没有采用课本上的矩阵次方求和来判断
        ju_zhen = self.matrix
        for l in range(1, len(ju_zhen)):
            sum = 0
            for r in range(0, l + 1):
                sum += ju_zhen[r][l] + ju_zhen[l][r]
            if not sum:
                print("不是连通图")
                return False
        print("是连通图")
        return True

    def isConnect2(self):
        """
        该函数用于判断简单图是否为连通图(课本）
        :return: True 连通图
                 False 不是连通图
        """
        ju_zhen = self.matrix
        sum = ju_zhen
        for i in range(2, len(ju_zhen)):
            sum = np.add(sum, self.matrixPow(ju_zhen, i))
        if 0 in sum:
            print("不是连通图")
            return False
        else:
            print("是连通图")
            return True

    def isEulerGraph(self):
        """
        该函数用于判断是否为欧拉图
        :return: True 是欧拉图
                 False 不是欧拉图
        """
        li = self.d_array
        for num in li:
            if num % 2 != 0:
                print("不是欧拉图")
                return False
        print("是欧拉图")
        return True

    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def makeEdge(self):
        """
        该函数用来构建顶点之间的连接关系字典
        :return: None
        """
        for r in range(0,self.V):
            for l in range(0, r+1):
                if self.matrix[r][l] == 1:
                    self.addEdge(r, l)

    def removeEdge(self, u, v):
        """
        该函数用于将边u-v删除，实质上是删除u和v在字典中的关系
        :param u: 顶点1
        :param v: 顶点2
        :return:  None
        """
        for index, key in enumerate(self.graph[u]):
            if key == v:
                self.graph[u].pop(index)
        for index, key in enumerate(self.graph[v]):
            if key == u:
                self.graph[v].pop(index)

    def DFSCount(self, v, visited):
        """
        该函数用于计算顶点v的可达顶点数
        :param v:
        :param visited:
        :return: 可达顶点个数
        """
        # 借鉴了DFS深度搜索算法
        count = 1
        visited[v] = True
        # 遍历v相邻的顶点
        for i in self.graph[v]:
            if visited[i] == False:
                count = count + self.DFSCount(i, visited)
        return count

    def isValidNextEdge(self, u, v):
        """
        该函数用于检查边u-v是否能够当作欧拉回路的下一条边
        :param u:
        :param v:
        :return:
        """
        # 如果u-v可以用要满足以下两点
        # 1) 如果v是u的唯一连接一个顶点
        if len(self.graph[u]) == 1:
            return True
        else:

            # 2) 如果有很多连接的边， 选择不是桥的边u-v
            # 计算u有多少可达顶点
            visited = [False] * (self.V)
            count1 = self.DFSCount(u, visited)

            # 删除边u-v后，再次计算u有多少可达顶点
            self.removeEdge(u, v)
            visited = [False] * (self.V)
            count2 = self.DFSCount(u, visited)

            # 将删去的边加回来
            self.addEdge(u, v)

            # 如果count1比较大，那么u-v是一座桥，所以不能删去
            # if count1 > count2:
            #     return False
            # else:
            #     return True
            return False if count1 > count2 else True

    def printEulerUtil(self, u):
        """
        该函数用于递归寻找欧拉回路
        :param u: 顶点
        :return: None
        """
        # 所有相邻的点到这个点的循环
        for v in self.graph[u]:
            # 如果边u-v没有被删除然后这是一个合法的下一条边
            if self.isValidNextEdge(u, v):
                # print("%d-%d " % (u, v)),
                print("-v%d" % (v+1), end=""),
                self.removeEdge(u, v)
                self.printEulerUtil(v)

    def findEulerCiruit(self):
        u = 0
        self.makeEdge()
        print("一条欧拉回路：", end="")
        print('v' + str(u + 1), end="")
        self.printEulerUtil(u)

    def matrixPow(self, Matrix, n):
        """
        该函数用于求矩阵的n次方
        :param Matrix: 所求矩阵
        :param n: n次方
        :return: 返回n次方后矩阵
        """
        if (type(Matrix) == list):
            Matrix = np.array(Matrix)
        if (n == 1):
            return Matrix
        else:
            return np.matmul(Matrix, self.matrixPow(Matrix, n - 1))

    def takeFirst(self, elem):
        """
        该函数用于返回数组的第0位，本类中用于.sort()的key
        :param elem: a list[]
        :return: the first element of a list
        """
        return elem[0]

def main():
    test_array = [[3, 3, 2, 2, 1],
                  [3, 3, 2],
                  [1, 1, 1, 1],
                  [3, 2, 2, 1],
                  [4, 4, 2, 2, 2]]

    for arr in test_array:
        print("-----------------")
        print(arr)
        Graph(arr)

if __name__ == '__main__':
    main()