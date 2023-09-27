import seaborn as sns
import matplotlib.pyplot as plt

# 创建一个示例的数据矩阵（DataFrame）
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 创建一个自定义的颜色映射
custom_cmap = sns.color_palette("coolwarm", as_cmap=True)

# 绘制热图，并设置颜色映射和颜色标度范围
sns.heatmap(data, cmap=custom_cmap, annot=True)

# 在不同的区域应用不同的颜色映射
# 使用 add_patch 方法添加矩形区域并设置不同的颜色映射
ax = plt.gca()
ax.add_patch(plt.Rectangle((0, 1), 2, 1, fill=True, color='yellow', alpha=0.5))
ax.add_patch(plt.Rectangle((1, 1), 2, 2, fill=True, color='green', alpha=0.5))

# 设置轴标签和标题
plt.xlabel('列')
plt.ylabel('行')
plt.title('不同区域设置不同颜色渐变的热图')

# 显示热图
plt.show()
