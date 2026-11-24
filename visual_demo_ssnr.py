import numpy as np
import matplotlib.pyplot as plt

# 设置绘图风格与中文字体支持 (兼容 Win/Mac)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 重新定义安全、防崩溃的矢量绘制函数
# ==========================================
def draw_vector(ax, start, vec, color, label, linestyle='-'):
    # 1. 绘制一条空线条用于生成标准、安全的图例
    ax.plot([], [], color=color, label=label, linestyle=linestyle, linewidth=2)
    
    # 2. 使用 annotate 绘制带有精确箭头的矢量线 (完美支持虚线)
    ax.annotate('', 
                xy=(start[0] + vec[0], start[1] + vec[1]), 
                xytext=(start[0], start[1]),
                arrowprops=dict(edgecolor=color, facecolor=color, 
                                arrowstyle="-|>", ls=linestyle, lw=2, mutation_scale=15))

# ==========================================
# 左图：未归一化空间 (尺度差异巨大)
# ==========================================
ax1.set_xlim(-1, 8)
ax1.set_ylim(-1, 8)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_aspect('equal')
ax1.set_title("1. 未归一化语义空间 (Scale Variance)", fontsize=14, fontweight='bold')

# 原始向量1 (大尺度 - 比如高对比度图像)
V_large_orig = np.array([6.0, 2.0])
V_large_rec  = np.array([5.5, 4.0]) # 接收向量
N_large = V_large_rec - V_large_orig # 干扰向量

# 原始向量2 (小尺度 - 比如纯色背景图像)
V_small_orig = np.array([1.5, 0.5])
V_small_rec  = np.array([1.0, 1.5])
N_small = V_small_rec - V_small_orig

# 绘制大向量及干扰
draw_vector(ax1, [0,0], V_large_orig, 'blue', '原始语义 (大尺度)')
draw_vector(ax1, [0,0], V_large_rec, 'lightblue', '接收语义')
draw_vector(ax1, V_large_orig, N_large, 'red', '语义干扰 (大尺度)', linestyle='--')

# 绘制小向量及干扰
draw_vector(ax1, [0,0], V_small_orig, 'green', '原始语义 (小尺度)')
draw_vector(ax1, [0,0], V_small_rec, 'lightgreen', '接收语义')
draw_vector(ax1, V_small_orig, N_small, 'orange', '语义干扰 (小尺度)', linestyle='--')

ax1.legend(loc='upper left', framealpha=0.9)

# ==========================================
# 右图：归一化空间 (映射到单位圆)
# ==========================================
ax2.set_xlim(-0.2, 1.2)
ax2.set_ylim(-0.2, 1.2)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_aspect('equal')
ax2.set_title("2. L2归一化空间与 S-SNR 等价性", fontsize=14, fontweight='bold')

# 画单位圆
circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='-.', linewidth=1.5, alpha=0.5)
ax2.add_patch(circle)
ax2.text(0.6, 0.85, "Unit Circle ($||V||_2 = 1$)", color='gray', fontsize=12)

# 归一化操作
V_norm_orig = V_large_orig / np.linalg.norm(V_large_orig)
V_norm_rec  = V_large_rec / np.linalg.norm(V_large_rec)
N_norm = V_norm_rec - V_norm_orig

# 绘制归一化向量
draw_vector(ax2, [0,0], V_norm_orig, 'blue', '归一化原始语义')
draw_vector(ax2, [0,0], V_norm_rec, 'lightblue', '归一化接收语义')
draw_vector(ax2, V_norm_orig, N_norm, 'red', '归一化语义干扰 (欧氏距离)', linestyle='--')

# 标注角度 theta
angle = np.linspace(np.arctan2(V_norm_orig[1], V_norm_orig[0]), np.arctan2(V_norm_rec[1], V_norm_rec[0]), 100)
x_arc = 0.3 * np.cos(angle)
y_arc = 0.3 * np.sin(angle)
ax2.plot(x_arc, y_arc, color='purple', linewidth=2)
ax2.text(0.35, 0.25, r"$\theta$", color='purple', fontsize=18)

ax2.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig("ssnr_concept_illustration.png", dpi=300)
print("✅ 图表已成功生成并保存为 ssnr_concept_illustration.png")
plt.show()