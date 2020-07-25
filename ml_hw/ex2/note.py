# 在第一部分中
# 使用SciPy's truncated newton (TNC) 失败 1.2.3 1.2.4
# 原因未知
# 重新使用原始GDS函数 accuracy远低于TNC


# 更新
# 0) theta要放在argument第一位！
# 1) opt.fmin_tnc 函数的x0必须为1D array，所以theta必须要shape为1*(n+1)
# 2) 在preprocess 时候都把X，y给设置为array，放入function时候转化成matrix
        原因可能是opt.fmin_tnc接受的args必须为array格式？