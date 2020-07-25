# 数据处理
# 1) panda只接受1d array数据，所以plot前需要ravel处理  （12，1）to (12,)
# 2) 使用panda和seaborn包来plot数据
# 3）+x0 要在normalization之后

# leaning curves
# 1) 先算出+regularized term时的theta
# 2）算出当前theta在没有regularization term时的 cv cost和 train cost
# 3）随着m增长 plot

# 改进
# 1）high bias 增加 ploy degree
# 2）根据不同的lambda进行两个cost的计算，找出最优的lambda
