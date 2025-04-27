import matplotlib.pyplot as plt

# 你的数据
n = [64, 128, 256, 512, 1024]
speedup = [3.45, 10.48, 38.98, 144.80, 261.44]

# 画图
plt.figure(figsize=(8,6))
plt.plot(n, speedup, marker='o')  # 带点的折线图
plt.xscale('log', base=2)          # x轴用对数刻度（因为是64,128,256这种）
plt.xlabel('Matrix size (n = m)')
plt.ylabel('Speedup (CPU time / GPU time)')
plt.title('Speedup vs Matrix Size')
plt.grid(True)
plt.tight_layout()
plt.show()
