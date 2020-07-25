#  nn model 可以分成两种
# 1）logistics regression找到最优theta，此时为single layer
# 2） 用已知theta feed forward求出prediction


# Theta not given------------------------------------------
# 数据处理
# 0) 用sio读取mat file
# 1) matshow plot mat file
# 2) y 在读取时候必须先reshape成（m,)的column vector，以便后面的one-hot转码
# 3）y 需要处理成 one-hot coding
# 4) y 需要array化, shape = (k,m)   k=no. of types, 单种type的y则为（m,)
# 5) X 有一步transpose?
# 6）X 要加入x0=1 cols

# 1-d model
# 只区分 one type or not that type ( use y[0] (5000,) classify the type y[0] represent as 1
# 0) cost function 需要额外的 regularization part (no theta_0 regularization)
# 1) minimize function 不要求  1d array args, theta 可以是 (n+1,1)
# 2) gradient 需要额外的 regularization term，同样无视theta_0，输出时加上theta_0 =0

# k-d model nnk
# 0) the theta output is now (10,401), in 1-d is (401,)
# 1) find the largest values' index as our y_pred (5000,)
# 2) use raw_y as y_answer, change 10 to 0
# 3) classification_report



# Theta given------------------------------------------
# 数据处理
# X 加x0=1 cols
# Y reshape 从（m,1) 化成(m,)
# a1 = input X
# z2 = X @ theta1.T
# a2 = sigmoid (z2) + (a0=1 cols)
