# 处理数据
# 1） onehot包
# 2） y不需要改成1-index，只需要最后predict时结果+1就好了
# 3) X he y_onehot 都matrix化， 使用minimize 不需要 array

# forward
# 算出每一个a和z 包括最后output h
# 每次算新层z时记得给当前层a加a0=0
# cost需要加regularization term，用y_onehot去做参数

# back prop
# theta的init
# sigmoid的inverse function
# 求在init theta时的grad去优化


