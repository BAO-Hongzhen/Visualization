import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import random

# 用于顺序选择波动线的计数器
wave_line_counter = 0

# --- 读取潮汐数据 ---
data = pd.read_csv(r'd:/MScIME25/SD5913-PFAD/pfad/week02/tides_processed.csv')
tide_levels = data['tide_level'].values
datetimes = data['datetime'].values

WIDTH, HEIGHT = 800, 180
N_LINES = 20
LINE_LEN = 700
Y_MARGIN = 5
ys = np.linspace(Y_MARGIN, HEIGHT - Y_MARGIN, N_LINES)

# --- 物理参数 ---
DECAY = 0.98  # 衰减系数
BASE_WAVE_SPEED = 1.0  # 基础波速（更快）
N_POINTS = LINE_LEN

# --- 弦的状态 ---
positions = np.zeros((N_LINES, N_POINTS))  # 当前位移
velocities = np.zeros((N_LINES, N_POINTS))  # 当前速度

# --- 初始化画布 ---
fig, ax = plt.subplots(figsize=(WIDTH/100, HEIGHT/100))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, LINE_LEN)
ax.set_ylim(0, HEIGHT)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.axis('off')

lines = []
x = np.linspace(0, LINE_LEN, N_POINTS)
for y in ys:
    (line,) = ax.plot(x, np.full(N_POINTS, y), lw=1, color='white')
    lines.append(line)

# --- 左下角潮汐+时间文本 ---
txt_bottom = ax.text(10, 10, '', color='white', fontsize=10, ha='left', va='bottom', backgroundcolor='black', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))


# --- 动画更新函数 ---
def animate(i):
    # 当前潮汐数据
    # 每24帧推进一个数据点
    data_speed = 6
    idx = (i // data_speed) % len(tide_levels)
    tide = tide_levels[idx]
    dt = datetimes[idx]
    txt_bottom.set_text(f'Tide: {tide:.2f}m\n{dt}')
    wave_speed = BASE_WAVE_SPEED

    # 每个数据点只产生一次波动
    global wave_line_counter
    if i % data_speed == 0:
        line_idx = N_LINES - 1 - wave_line_counter
        wave_line_counter = (wave_line_counter + 1) % N_LINES
        pos = random.randint(N_POINTS//4, 3*N_POINTS//4)
        amp = 10 + 25 * (tide / tide_levels.max())
        width = 12
        left = max(0, pos-width//2)
        right = min(N_POINTS, pos+width//2)
        positions[line_idx, left:right] += amp * np.hanning(right-left)

    # 物理模拟：一维波动方程（简化）
    for j in range(N_LINES):
        # 差分近似二阶导数
        laplacian = np.zeros(N_POINTS)
        laplacian[1:-1] = positions[j, :-2] - 2*positions[j, 1:-1] + positions[j, 2:]
        velocities[j] += wave_speed * laplacian
        velocities[j] *= DECAY  # 衰减
        positions[j] += velocities[j]
        # 边界固定
        positions[j, 0] = 0
        positions[j, -1] = 0
        # 更新线条
        lines[j].set_ydata(ys[j] + positions[j])

    return lines + [txt_bottom]

ani = animation.FuncAnimation(fig, animate, frames=len(tide_levels), interval=120, blit=True)
plt.show()
