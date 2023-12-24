import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def find_v_with_boundary_conditions(XL, YL, Nx, Ny, epsilon, rho, er, e0, t, max_iter=500, tolerance=1e-6):
    # 그리드 생성
    x = np.linspace(0, XL, Nx)
    y = np.linspace(0, YL, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # 초기 조건 설정
    phi = np.zeros_like(X)

    # 경계 조건 설정
    phi[:, 0] = 0  # 왼쪽 경계
    phi[:, -1] = 0  # 오른쪽 경계
    phi[0, :] = 0  # 위쪽 경계

    # 유전체의 위치 정의
    x1_dielect_i = int(0.2 * Nx)
    x1_dielect_f = int(0.4 * Nx)
    x2_dielect_i = int(0.6 * Nx)
    x2_dielect_f = int(0.8 * Nx)

    # PDE 해결
    for _ in range(max_iter):
        # 이전 전위 값 저장
        phi_old = np.copy(phi)

        # Jacobi 방법을 사용한 PDE 해결
        for i in range(1, Nx // 2):  # 반만 계산
            for j in range(1, Ny - 1):
                # 경계 조건이면 전위를 고정
                if (i > x1_dielect_f and i < Nx / 2 and j >= 0 and j <= int(0.4 * Ny)):
                    phi[j, i] = np.sin(t)
                    # 좌우대칭
                    phi[j, Nx - i - 1] = phi[j,i]
                # 경계조건
                elif (i == x1_dielect_f and j >= 0 and j <= int(0.4 * Ny)):  # 유전체 좌측
                    phi[j, i] = (-rho[j, i] / e0 + er * phi[j + 1, i] + phi[j - 1, i]) / (1 + er)
                    # 좌우대칭
                    phi[j, Nx - i - 1] = phi[j, i]
                elif ((j == int(0.4 * Ny) and i >= x1_dielect_i and i <= x1_dielect_f)):  # 유전체 상단
                    phi[j, i] = (-rho[j, i] / e0 + phi[j, i + 1] + er * phi[j, i - 1]) / (1 + er)
                    # 좌우대칭
                    phi[j, Nx - i - 1] = phi[j, i]
                else:
                    phi[j, i] = 0.25 * (
                            phi[j, i + 1] + phi[j, i - 1] + phi[j + 1, i] + phi[j - 1, i] - dx * dy * rho[j, i] / e0)
                    # 좌우대칭
                    phi[j, Nx - i - 1] = phi[j, i]

        # 수렴 확인
        if np.max(np.abs(phi - phi_old)) < tolerance:
            break
    return X, Y, phi


def electric_field(phi, dx, dy):
    Ex = -np.gradient(phi, axis=1) / dx  # x방향 전기장
    Ey = -np.gradient(phi, axis=0) / dy  # y방향 전기장
    return Ex, Ey


def velocity_model(y, t, Ex, Ey, e_charge=1.602e-19, m=9.109e-31):
    vy, vx = y[0], y[1]
    dvxdt = (e_charge / m) * vx * Ex
    dvydt = (e_charge / m) * vy * Ey
    return [dvydt, dvxdt]


def update_charge_density(x, y, Nx, Ny, dx, dy, electron_charge=1.602e-19):
    
    """
    위치 배열 만들어 전자들 랜덤 생성
    x값 y값 저장하고 rho를 dx dy간격으로 q/s 를 통해 전하밀도 구함
    rho 배열에 개수 X전하량으로 update
    """
    
    rho = np.zeros((Ny, Nx))
    i_idx = (x / dx).astype(int)
    j_idx = (y / dy).astype(int)
    valid_indices = (0 <= i_idx) & (i_idx < Nx) & (0 <= j_idx) & (j_idx < Ny)
    i_idx = i_idx[valid_indices]
    j_idx = j_idx[valid_indices]
    rho[j_idx, i_idx] += electron_charge / (dx * dy)
    return rho


# 매개변수 설정
XL = 10
YL = 10
Nx = 100
Ny = 100
epsilon = 1.0

# 전자의 초기 위치
num_electrons = 1000
x0 = np.random.uniform(low=0, high=XL, size=num_electrons)
y0 = np.random.uniform(low=4, high=YL, size=num_electrons)

# 시뮬레이션 시간 설정
t = np.arange(0, 10, 1)

# 전자의 초기 위치 및 속도
x_pos, y_pos = x0.tolist(), y0.tolist()
vx0, vy0 = np.zeros_like(x0), np.zeros_like(y0)
vx, vy = vx0.tolist(), vy0.tolist()


def update(frame, x_pos, y_pos, vx, vy):
    current_time = t[frame]
    X, Y, phi = find_v_with_boundary_conditions(XL, YL, Nx, Ny, epsilon, np.zeros((Ny, Nx)), 4.0, 8.854e-12, current_time)
    dt = t[frame] - t[frame - 1]  # Calculate time step

    # Electric field calculation
    Ex, Ey = electric_field(phi, X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0])

    # Update positions using the velocity model (Euler method)
    vx_temp, vy_temp = velocity_model([vx[frame - 1], vy[frame - 1]], current_time, Ex, Ey)
    x_temp, y_temp = x_pos[frame - 1] + vx_temp * dt, y_pos[frame - 1] + vy_temp * dt

    # Update charge density
    rho = update_charge_density(x_temp, y_temp, Nx, Ny, X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0])

    # Extend lists with new values
    x_pos.extend(x_temp)
    y_pos.extend(y_temp)
    vx.extend(vx_temp)
    vy.extend(vy_temp)

    # Plotting
    levels = np.linspace(np.min(phi), np.max(phi), 100)
    plt.clf()
    plt.contourf(X, Y, phi, levels=levels, cmap='viridis')
    plt.colorbar()
    plt.quiver(X, Y, Ex, Ey, scale=20, color='white', width=0.007)
    plt.scatter(x_temp, y_temp, color='red', label='Electron')
    plt.title(f'Time: {current_time} seconds')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()


# 나머지 코드는 그대로 사용

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=np.arange(1, len(t)), fargs=(x_pos, y_pos, vx, vy), interval=200, repeat=False)
plt.show()