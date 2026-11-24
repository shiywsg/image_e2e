import numpy as np
import matplotlib.pyplot as plt

# 设置绘图风格与中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

tau = 0.4 # 三值擦除的阈值

def plot_ternary_step(ax):
    """绘制三值阶跃函数"""
    x1 = np.linspace(-1.5, -tau, 100)
    x2 = np.linspace(-tau, tau, 100)
    x3 = np.linspace(tau, 1.5, 100)
    
    ax.plot(x1, -np.ones_like(x1), 'b-', linewidth=3)
    ax.plot(x2, np.zeros_like(x2), 'b-', linewidth=3)
    ax.plot(x3, np.ones_like(x3), 'b-', linewidth=3)
    
    # 画断点圆圈
    ax.scatter([-tau, tau], [-1, 0], facecolors='none', edgecolors='b', s=80, zorder=5)
    ax.scatter([-tau, tau], [0, 1], color='b', s=80, zorder=5)

# ==========================================
# 左图：真实的前向传播 (梯度消失的死胡同)
# ==========================================
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title("1. 真实正向传播 (阶跃导致梯度消失)", fontsize=14, fontweight='bold')
ax1.set_xlabel("连续输入 (Noisy Symbols)", fontsize=12)
ax1.set_ylabel("离散输出 (Ternary Symbols)", fontsize=12)

plot_ternary_step(ax1)

# 标注梯度为 0 的区域
ax1.text(-1.0, -0.8, r"导数 $\nabla=0$", color='red', fontsize=14, fontweight='bold')
ax1.text(-0.2, 0.2, r"导数 $\nabla=0$", color='red', fontsize=14, fontweight='bold')
ax1.text(0.8, 1.2, r"导数 $\nabla=0$", color='red', fontsize=14, fontweight='bold')

# 标注断点
ax1.axvline(-tau, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(tau, color='gray', linestyle='--', alpha=0.5)

# ==========================================
# 右图：STE 梯度直通 (反向传播的“偷梁换柱”)
# ==========================================
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_title("2. STE 梯度直通 (反向传播的“替身”)", fontsize=14, fontweight='bold')
ax2.set_xlabel("连续输入 (Noisy Symbols)", fontsize=12)
ax2.set_ylabel("反向传播回传的等效梯度路径", fontsize=12)

# 绘制浅色的阶跃函数作为背景
x1 = np.linspace(-1.5, -tau, 100)
x2 = np.linspace(-tau, tau, 100)
x3 = np.linspace(tau, 1.5, 100)
ax2.plot(x1, -np.ones_like(x1), 'b-', linewidth=3, alpha=0.2)
ax2.plot(x2, np.zeros_like(x2), 'b-', linewidth=3, alpha=0.2)
ax2.plot(x3, np.ones_like(x3), 'b-', linewidth=3, alpha=0.2)

# 绘制 STE 代理函数 (y = x)
ax2.plot([-1.5, 1.5], [-1.5, 1.5], 'r--', linewidth=3, label="STE 代理函数 (y = x)")

# 解释文字
ax2.text(-0.5, 0.8, "无视阶跃\n强行回传!", color='red', fontsize=14, fontweight='bold', 
         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

# 代码对应
code_text = "rx_input = (ternary - noisy).detach() + noisy"
ax2.text(-1.4, -1.2, f"PyTorch魔法:\n{code_text}", color='green', fontsize=12, family='monospace',
         bbox=dict(facecolor='lightgreen', alpha=0.3))

ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig("ste_illustration.png", dpi=300)
print("✅ 图表已生成：ste_illustration.png")
plt.show()