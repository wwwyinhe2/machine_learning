
# coding: utf-8

# In[3]:


# 任意选一个你喜欢的整数，这能帮你得到稳定的结果
seed = 9527


# # 欢迎来到线性回归项目
# 
# 若项目中的题目有困难没完成也没关系，我们鼓励你带着问题提交项目，评审人会给予你诸多帮助。
# 
# 所有选做题都可以不做，不影响项目通过。如果你做了，那么项目评审会帮你批改，也会因为选做部分做错而判定为不通过。
# 
# 其中非代码题可以提交手写后扫描的 pdf 文件，或使用 Latex 在文档中直接回答。

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[4]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何NumPy以及相关的科学计算库来完成作业


# 本项目要求矩阵统一使用二维列表表示，如下：
A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

# 向量也用二维列表表示
C = [[1],
     [2],
     [3]]

#TODO 创建一个 4*4 单位矩阵

I = [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]


# ## 1.2 返回矩阵的行数和列数

# In[5]:


# TODO 返回矩阵的行数和列数
def shape(M):
    row = len(M)
    col = len(M[0])
    return row,col


# In[6]:


# 运行以下代码测试你的 shape 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 每个元素四舍五入到特定小数数位

# In[7]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
from decimal import Decimal
def matxRound(M, decPts=4):
    row = len(M)
    col = len(M[0])
    
    for i in range(row):
        for j in range(col):
            M[i][j] = round(Decimal(M[i][j]),decPts)

#下面方法虽好，但有返回值，与题设不符：
# N =[[round(Decimal(elem),decPts) for elem in row] for row in M]
# return N


# In[8]:


# 运行以下代码测试你的 matxRound 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 计算矩阵的转置

# In[9]:


# TODO 计算矩阵的转置
def transpose(M):
    row = len(M)
    col = len(M[0])
    
    #首先生成一个正确的转置0矩阵很重要，
    N = [[0 for x in range(row)] for y in range(col)]
    
    for i in range(row):
        for j in range(col):
            N[j][i] = M[i][j]
            
    return N


# In[10]:


# 运行以下代码测试你的 transpose 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 计算矩阵乘法 AB

# In[11]:


# TODO 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):
    if len(A[0])!= len(B):
        raise ValueError
    else:
        #先生成一个0矩阵,需要注意的是，非常容易把矩阵的行列弄反~~~~： 
        C = [[0 for x in range(len(B[0]))] for y in range(len(A))]
        
        #计算AB乘积
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    C[i][j] += A[i][k]*B[k][j]
        return C
    
    


# In[12]:


# 运行以下代码测试你的 matxMultiply 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[13]:


# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):

    C = [[0 for x in range(len(A[0])+1)] for y in range(len(A))]

    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j]

    for k in range(len(A)):
        #需要注意b[k][0]的定义~
        C[k][len(A[0])] = b[k][0]   


    return C


# In[14]:


# 运行以下代码测试你的 augmentMatrix 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# # 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[15]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1],M[r2] = M[r2],M[r1]


# In[16]:


# 运行以下代码测试你的 swapRows 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_swapRows')


# In[17]:


# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    else:
        for i in range(len(M[r])):
            M[r][i] = M[r][i] * scale


# In[18]:


# 运行以下代码测试你的 scaleRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[19]:


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    
    #需要注意不要直接scale r2 到r2，不要改变r2的值。
    
    C = [0 for i in range(len(M[r2]))]
    
    for i in range(len(M[r2])):
        C[i] = M[r2][i]*scale
        
    for i in range(len(M[r1])):
        M[r1][i] = M[r1][i] +C[i]


# In[20]:


# 运行以下代码测试你的 addScaledRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_addScaledRow')


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 2.3.1 算法
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None (你可以在选做问题2.4中证明为什么这里A一定是奇异矩阵)
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# **注：** 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# ### 2.3.2 算法推演
# 
# 为了充分了解Gaussian Jordan消元法的计算流程，请根据Gaussian Jordan消元法，分别手动推演矩阵A为***可逆矩阵***，矩阵A为***奇异矩阵***两种情况。

# In[5]:


# 不要修改这里！
from helper import *

A = generateMatrix(4,seed,singular=False)
b = np.ones(shape=(4,1)) # it doesn't matter
Ab = augmentMatrix(A.tolist(),b.tolist()) # please make sure you already correct implement augmentMatrix
printInMatrixFormat(Ab,padding=4,truncating=0)


# 请按照算法的步骤3，逐步推演***可逆矩阵***的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵。
# 
# 要求：
# 1. 做分数运算
# 2. 使用`\frac{n}{m}`来渲染分数，如下：
#  - $\frac{n}{m}$
#  - $-\frac{a}{b}$
# 
# 可逆矩阵
# 
# $ A = \begin{bmatrix}
#     -9 & -2 & 6 & 6\\
#     2 & -2 & 6 & -9\\
#     1 & -1 & -8 & 6\\
#     3 & 2 & -10 & -9\end{bmatrix}$
# $ b = \begin{bmatrix}
#     1\\
#     1\\
#     1\\
#     1\end{bmatrix}$
# 
# 
# 增广矩阵
# $ Ab = \begin{bmatrix}
#     -9 & -2 & 6 & 6 & 1\\
#     2 & -2 & 6 & -9 & 1\\
#     1 & -1 & -8 & 6 & 1\\
#     3 & 2 & -10 & -9 & 1\end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     1 & \frac{2}{9} & -\frac{2}{3} & -\frac{2}{3} & -\frac{1}{9}\\
#     0 & -\frac{22}{9} & \frac{22}{3} & -\frac{23}{3} & \frac{11}{9}\\
#     0 & -\frac{11}{9} & -\frac{22}{3} & \frac{20}{3} & \frac{10}{9}\\
#     0 & \frac{4}{3} & -8 & -7 & \frac{4}{3}\end{bmatrix}$
#     
# $ --> \begin{bmatrix}
#     1 & 0 & 0 & -\frac{4}{3} & 0\\
#     0 & 1 & -3 & \frac{22}{7} & -\frac{1}{2}\\
#     0 & 0 & -11 & \frac{21}{2} & \frac{1}{2}\\
#     0 & 0 & -4 & -\frac{67}{6} & 2\end{bmatrix}$
#     
# $--> \begin{bmatrix}
#     1 & 0 & 0 & -\frac{4}{3} & 0\\
#     0 & 1 & 0 & \frac{1}{4} & -\frac{2}{3}\\
#     0 & 0 & 1 & -1 & 0\\
#     0 & 0 & 0 & -15 & \frac{9}{5}\end{bmatrix}$
# 
# $--> \begin{bmatrix}
#     1 & 0 & 0 & 0 & -\frac{1}{6}\\
#     0 & 1 & 0 & 0 & -\frac{3}{5}\\
#     0 & 0 & 1 & 0 & -\frac{1}{6}\\
#     0 & 0 & 0 & 1 & -\frac{1}{8}\end{bmatrix}$
#     

# In[9]:


# 不要修改这里！
A = generateMatrix(4,seed,singular=True)
b = np.ones(shape=(4,1)) # it doesn't matter
Ab = augmentMatrix(A.tolist(),b.tolist()) # please make sure you already correct implement augmentMatrix
printInMatrixFormat(Ab,padding=4,truncating=0)


# 请按照算法的步骤3，逐步推演***奇异矩阵***的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵。
# 
# 要求：
# 1. 做分数运算
# 2. 使用`\frac{n}{m}`来渲染分数，如下：
#  - $\frac{n}{m}$
#  - $-\frac{a}{b}$
# 
# 增广矩阵
# $ Ab = \begin{bmatrix}
#     -4 & -5 & -1 & 4 & 1\\
#     -4 & -5 & -4 & 0 & 1\\
#     -7 & -6 & 0 & 7 & 1\\
#     3 & -10 & -2 & 5 & 1\end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     1 & \frac{6}{7} & 0 & -1 & -\frac{1}{7}\\
#     0 & -\frac{11}{7} & -4 & -4 & \frac{3}{7}\\
#     0 & -\frac{11}{7} & -1 & 0 & \frac{3}{7}\\
#     0 & -\frac{88}{7} & -2 & 8 & \frac{10}{7}\end{bmatrix}$
#     
# $ --> \begin{bmatrix}
#     1 & 0 & -\frac{1}{7} & -\frac{4}{9} & 0\\
#     0 & 1 & \frac{1}{6} & -\frac{2}{3} & -\frac{1}{9}\\
#     0 & 0 & -\frac{3}{4} & -1 & \frac{1}{4}\\
#     0 & 0 & -\frac{15}{4} & -5 & \frac{1}{4}\end{bmatrix}$
#     
# $--> \begin{bmatrix}
#     1 & 0 & 0 & -\frac{1}{4} & 0\\
#     0 & 1 & 0 & -\frac{6}{7} & -\frac{1}{9}\\
#     0 & 0 & 1 & \frac{4}{3} & 0\\
#     0 & 0 & 0 & 0 & \frac{1}{5}\end{bmatrix}$
#     
# 

# ### 2.3.3 实现 Gaussian Jordan 消元法

# In[47]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""


def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    #返回None，如果A,b高度不同
    if len(A)!= len(b):
        return None
    else:
        #构建增广矩阵
        Ab = augmentMatrix(A,b)
        
        #先每一列,不包括最后1列
        for j in range(len(Ab[0])-1):
            
            #其他行最大值
            max_num = abs(Ab[j][j])
            for i in range(j+1,len(Ab)):
                if abs(Ab[i][j]) > max_num:
                    max_num = abs(Ab[i][j])
                    
            #其他最大行如比对角线行绝对值大，互换
            for i in range(j+1,len(Ab)):
                if abs(Ab[i][j]) == max_num:
                    swapRows(Ab,j,i)
    
            #如果A为奇异矩阵，返回None
            if abs(Ab[j][j]) < epsilon :
                return None
            else:
                #将对角线行系数变1
                num = Ab[j][j]
                scale = 1./num  #注意要用float
                scaleRow(Ab,j, scale)
                
                #将其余列数变零
                for k in range(len(Ab)):
                    if k != j :
                        num_2 = -Ab[k][j]
                        Ab[k][j] =addScaledRow(Ab,k,j,num_2)
                        Ab[k][j] = 0
        #将值传到列b里面
        for i in range(len(Ab)):
            b[i][0] = Ab[i][len(Ab)]
        #计算b的精度
        for i in range(len(b)):
            b[i][0] = round(Decimal(b[i][0]),decPts)
        
    return b


# In[48]:


# 运行以下代码测试你的 gj_Solve 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_gj_Solve')


# ## (选做) 2.4 算法正确判断了奇异矩阵：
# 
# 在算法的步骤3 中，如果发现某一列对角线和对角线以下所有元素都为0，那么则断定这个矩阵为奇异矩阵。
# 
# 我们用正式的语言描述这个命题，并证明为真。
# 
# 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 证明：

# # 3  线性回归

# ## 3.1 随机生成样本点

# In[30]:


# 不要修改这里！
# 运行一次就够了！
from helper import *
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

X,Y = generatePoints(seed,num=100)

## 可视化
plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')
plt.show()


# ## 3.2 拟合一条直线
# 
# ### 3.2.1 猜测一条直线

# In[49]:


#TODO 请选择最适合的直线 y = mx + b
m = -1.75
b = 9.55

# 不要修改这里！
plt.xlim((-5,5))
x_vals = plt.axes().get_xlim()
y_vals = [m*x+b for x in x_vals]
plt.plot(x_vals, y_vals, '-', color='r')

plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')

plt.show()


# ### 3.2.2 计算平均平方误差 (MSE)

# 我们要编程计算所选直线的平均平方误差(MSE), 即数据集中每个点到直线的Y方向距离的平方的平均数，表达式如下：
# $$
# MSE = \frac{1}{n}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$

# In[51]:


# TODO 实现以下函数并输出所选直线的MSE

def calculateMSE(X,Y,m,b):
    #初设个空列表和sum：
    c = [0]*100
    sum = 0
    
    #使用zip将yi-m*xi-b的平方填入列表 
    for x, y,z in zip(X,Y,range(100)):
        c[z] = (y-m*x-b)*(y-m*x-b)
    #将列表值求和
    for i in range(100):
        sum += c[i]
    
    #求出平均值
    sum = 1./100*sum
    
    #返回sum
    return sum

print(calculateMSE(X,Y,m,b))


# ### 3.2.3 调整参数 $m, b$ 来获得最小的平方平均误差
# 
# 你可以调整3.2.1中的参数 $m,b$ 让蓝点均匀覆盖在红线周围，然后微调 $m, b$ 让MSE最小。

# ## 3.3 (选做) 找到参数 $m, b$ 使得平方平均误差最小
# 
# **这一部分需要简单的微积分知识(  $ (x^2)' = 2x $ )。因为这是一个线性代数项目，所以设为选做。**
# 
# 刚刚我们手动调节参数，尝试找到最小的平方平均误差。下面我们要精确得求解 $m, b$ 使得平方平均误差最小。
# 
# 定义目标函数 $E$ 为
# $$
# E = \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 因为 $E = \frac{n}{2}MSE$, 所以 $E$ 取到最小值时，$MSE$ 也取到最小值。要找到 $E$ 的最小值，即要找到 $m, b$ 使得 $E$ 相对于 $m$, $E$ 相对于 $b$ 的偏导数等于0. 
# 
# 因此我们要解下面的方程组。
# 
# $$
# \begin{cases}
# \displaystyle
# \frac{\partial E}{\partial m} =0 \\
# \\
# \displaystyle
# \frac{\partial E}{\partial b} =0 \\
# \end{cases}
# $$
# 
# ### 3.3.1 计算目标函数相对于参数的导数
# 首先我们计算两个式子左边的值
# 
# 证明/计算：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-(y_i - mx_i - b)}
# $$

# TODO 证明:

# ### 3.3.2 实例推演
# 
# 现在我们有了一个二元二次方程组
# 
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# $$
# 
# 为了加强理解，我们用一个实际例子演练。
# 
# 我们要用三个点 $(1,1), (2,2), (3,2)$ 来拟合一条直线 y = m*x + b, 请写出
# 
# - 目标函数 $E$, 
# - 二元二次方程组，
# - 并求解最优参数 $m, b$

# TODO 写出目标函数，方程组和最优参数

# ### 3.3.3 将方程组写成矩阵形式
# 
# 我们的二元二次方程组可以用更简洁的矩阵形式表达，将方程组写成矩阵形式更有利于我们使用 Gaussian Jordan 消元法求解。
# 
# 请证明 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = X^TXh - X^TY
# $$
# 
# 其中向量 $Y$, 矩阵 $X$ 和 向量 $h$ 分别为 :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 证明:

# 至此我们知道，通过求解方程 $X^TXh = X^TY$ 来找到最优参数。这个方程十分重要，他有一个名字叫做 **Normal Equation**，也有直观的几何意义。你可以在 [子空间投影](http://open.163.com/movie/2010/11/J/U/M6V0BQC4M_M6V2AJLJU.html) 和 [投影矩阵与最小二乘](http://open.163.com/movie/2010/11/P/U/M6V0BQC4M_M6V2AOJPU.html) 看到更多关于这个方程的内容。

# ### 3.4 求解 $X^TXh = X^TY$ 
# 
# 在3.3 中，我们知道线性回归问题等价于求解 $X^TXh = X^TY$ (如果你选择不做3.3，就勇敢的相信吧，哈哈)

# In[52]:


# TODO 实现线性回归
'''
参数：X, Y
返回：m，b
'''
def linearRegression(X,Y):
    #将X,Y生成为矩阵100x1矩阵：
    X_ju = [[0 for x in range(2)] for y in range(100)]
    for x,i in zip(X,range(100)):
        X_ju[i][0] = x
        X_ju[i][1] = 1

    
    Y_ju = [[0 for x in range(1)] for y in range(100)]
    for y,i in zip(Y,range(100)):
        Y_ju[i][0] = y
    
    #求出X_ju矩阵的转置
    X_t = transpose(X_ju)
    
    #求出X_t转置和X的积，X_t和Y的积
    A = matxMultiply(X_t,X_ju)
    b = matxMultiply(X_t,Y_ju)

    #使用高斯销项法求解
    h = gj_Solve(A,b)
    
    #将解赋值给m，b
    m = h[0][0]
    b = h[1][0]    
    
    return m,b

m,b = linearRegression(X,Y)
print(m,b)


# 你求得的回归结果是什么？
# 请使用运行以下代码将它画出来。

# In[53]:


# 请不要修改下面的代码
x1,x2 = -5,5
y1,y2 = x1*m+b, x2*m+b

plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')
plt.plot((x1,x2),(y1,y2),'r')
plt.text(1,2,'y = {m}x + {b}'.format(m=m,b=b))
plt.show()


# 你求得的回归结果对当前数据集的MSE是多少？

# In[54]:


calculateMSE(X,Y,m,b)


# In[ ]:


0.9775816480290155

