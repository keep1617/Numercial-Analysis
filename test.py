import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def find_v_with_boundary_conditions(XL, YL, Nx, Ny, epsilon, rho, er, e0,t, max_iter=500, tolerance=1e-6):
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
    x1_dielect_i = int(0.1 * Nx)
    x1_dielect_f = int(0.2 * Nx)
    x2_dielect_i = int(0.8 * Nx)
    x2_dielect_f = int(0.9 * Nx)
    
    # PDE 해결
    # PDE 해결
    for _ in range(max_iter):
    # 이전 전위 값 저장
        phi_old = np.copy(phi)
        
    # Jacobi 방법을 사용한 PDE 해결
        for i in range(1, Nx//2):  # 반만 계산
            for j in range(1, Ny-1):
            # 경계 조건이면 전위를 고정
                if (i > x1_dielect_f and i < Nx/2 and j >= 0 and j <= int(0.4 * Ny)):
                    phi[j, i] = 1
                    
                # 좌우대칭
                    phi[j, Nx-i-1] = 1
            # 경계조건
                elif (i == x1_dielect_f and j >= 0 and j <= int(0.4 * Ny)):  # 유전체 좌측
                    phi[j, i] = (-rho[j, i] / e0 + er * phi[j + 1, i] + phi[j - 1, i]) / (1 + er)
                # 좌우대칭
                    phi[j, Nx-i-1] = phi[j, i]
                elif ((j == int(0.4 * Ny) and i >= x1_dielect_i and i <= x1_dielect_f)):  # 유전체 상단
                    phi[j, i] = (-rho[j, i] / e0 + phi[j, i + 1] + er * phi[j, i - 1]) / (1 + er)
                # 좌우대칭
                    phi[j, Nx-i-1] = phi[j, i]
                else:
                    phi[j, i] = 0.25 * (phi[j, i + 1] + phi[j, i - 1] + phi[j + 1, i] + phi[j - 1, i] - dx * dy * rho[j, i] / e0)
                # 좌우대칭
                    phi[j, Nx-i-1] = phi[j, i]
                ## 전기장 계산
                
        
# 수렴 확인
        if np.max(np.abs(phi - phi_old)) < tolerance:
            break
    return X, Y, phi
        




# 매개변수 설정
XL = 10
YL = 10
Nx = 100
Ny = 100
epsilon = 1.0

# 유전체의 위치 정의
x1_dielect_i = int(0.2 * Nx)
x1_dielect_f = int(0.4 * Nx)
x2_dielect_i = int(0.6 * Nx)
x2_dielect_f = int(0.8 * Nx)

rho = np.zeros((Nx, Ny))
rho[:, :] = 0.1
rho[0:x1_dielect_f, x1_dielect_i:x2_dielect_f+1] = 0

er = 4.0  # 유전율
e0 = 8.854e-12  # 자유공간의 유전율
for t in np.arange(0,6,1):
# 전위 계산
    X, Y, phi = find_v_with_boundary_conditions(XL, YL, Nx, Ny, epsilon, rho, er, e0,t)

    # 결과 시각화
    levels = np.linspace(np.min(phi), np.max(phi), 100)  # 전위 값에 따라 100단계로 나누어 색 지정
    plt.contourf(X, Y, phi, levels=levels, cmap='viridis')
    plt.colorbar()
    plt.title('Voltage')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
