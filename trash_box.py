# exercise 8.1.2



# 定义λ（alpha）值的范围

alphas = lambda_interval

# 初始化存储平均验证误差的列表
mean_errors = []

# 执行K折交叉验证
for alpha in alphas:
    model = Ridge(alpha=alpha)
    errors = -cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    mean_errors.append(np.mean(errors))

# 绘制 λ 函数的估计泛化误差图
plt.figure(figsize=(8, 6))
plt.semilogx(alphas, mean_errors, marker='o')
plt.xlabel('λ (alpha)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Estimated Generalization Error vs. λ')
plt.grid(True)
plt.show()

# 找到最小泛化误差对应的λ值
optimal_alpha = alphas[np.argmin(mean_errors)]
print(f"Optimal λ (alpha): {optimal_alpha}")