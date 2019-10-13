import numpy as np
import cv2
import matplotlib.pyplot as plt


# 菲涅尔衍射S-FFT算法
def fresnel_diff_s_fft(U0, z, dx, r):
    k = 2*np.pi/r
    [M, N] = np.shape(U0)

    # 建立初始面坐标系
    u = dx*(np.linspace(0, N-1, N)-N/2)
    v = dx*(np.linspace(0, M-1, M)-M/2)
    [u, v] = np.meshgrid(u, v)

    # 傅里叶积分内的指数相位因子与物函数相乘
    fresnel = np.exp(1j*k/(2*z)*(u**2+v**2))
    if z>0:
        Uf = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(U0 * fresnel)))
    else:
        Uf = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(U0 * fresnel)))

    # 建立观测面坐标系
    L = r*z/dx  # 傅里叶变换后对应的观测面宽度
    x = (L/N)*(np.linspace(0, N-1, N)-N/2)
    y = (L/M)*(np.linspace(0, M-1, M)-M/2)
    [x, y] = np.meshgrid(x, y)

    # 二次相位因子
    phase = np.exp(1j*k*z)/(1j*r*z)*np.exp(1j*k/(2*z)*(x**2+y**2))
    Uf = Uf*phase

    I = Uf*np.conj(Uf)

    return Uf, I, L/N


# 菲涅尔衍射T-FFT算法
def fresnel_diff_t_fft(U0, z, dx, r):
    k = 2 * np.pi / r
    [M, N] = np.shape(U0)

    # 建立衍射面坐标
    x = dx * (np.linspace(0, N - 1, N) - N / 2)
    y = dx * (np.linspace(0, M - 1, M) - M / 2)
    [x, y] = np.meshgrid(x, y)

    # 频域相乘，卷积
    fresnel = np.exp(1j * k / (2 * z) * (x ** 2 + y ** 2))
    Uf = np.fft.fftshift(np.fft.fft2(U0)) * np.fft.fftshift(np.fft.fft2(fresnel))

    phase = np.exp(1j * k * z) / (1j * r * z)
    U = phase * np.fft.ifftshift(np.fft.ifft2(Uf))

    I = U * np.conj(U)

    return U, I


# 菲涅尔衍射D-FFT算法
def fresnel_diff_d_fft(U0, z, dx, r):
    k = 2*np.pi/r
    [M, N] = np.shape(U0)

    # 建立频谱坐标系
    fx = (1/(N*dx))*(np.linspace(0, N-1, N)-N/2)
    fy = (1/(M*dx))*(np.linspace(0, N-1, N)-N/2)
    [fx, fy] = np.meshgrid(fx, fy)

    Uf = np.fft.fftshift(np.fft.fft2(U0))
    H = np.exp(1j*k*z*(1-r**2/2*(fx**2+fy**2))) # 传递函数
    U = np.fft.ifft2(np.fft.ifftshift(Uf*H))

    I = U*np.conj(U)

    return U, I


# 角谱衍射算法
def angular_diff(U0, z, dx, r):
    [M, N] = np.shape(U0)

    # 建立频谱坐标系
    fx = (1 / (N * dx)) * (np.linspace(0, N - 1, N) - N / 2)
    fy = (1 / (M * dx)) * (np.linspace(0, N - 1, N) - N / 2)
    [fx, fy] = np.meshgrid(fx, fy)

    trans = np.exp(1j*2*np.pi*z/r*np.sqrt(1-(r*fx)**2-(r*fy)**2))
    Uf = np.fft.fftshift(np.fft.fft2(U0))*trans
    U = np.fft.ifft2(np.fft.ifftshift(Uf))

    I = U*np.conj(U)
    return U, I


if __name__ == "__main__":
    r = 0.532e-3    # 波长
    dx = 6.4e-3     # 初始面采样间隔

    U0 = cv2.imread("Imgs/lenna.bmp", 0)
    [M, N] = np.shape(U0)

    plt.figure()
    plt.imshow(U0, 'gray'), plt.axis('off'), plt.title('Raw')

    z = dx**2*N/r   # 完全满足采样定律的距离
    z = 100       # 单位 mm

    [U, I, dfx] = fresnel_diff_s_fft(U0, z, dx, r)
    [U, I, dx] = fresnel_diff_s_fft(U, -z, dfx, r)

    # [U, I] = fresnel_diff_t_fft(U0, z, dx, r)

    # [U, I] = fresnel_diff_d_fft(U0, z, dx, r)
    # [U, I] = fresnel_diff_d_fft(U, -z, dx, r)

    # [U, I] = angular_diff(U0, z, dx, r)
    # [U, I] = angular_diff(U, -z, dx, r)

    plt.figure()
    plt.imshow(np.abs(I), 'gray'), plt.axis('off'), plt.title('Diffraction')

    # Variable_Sampling_rate
    q = 1.2     # Variable_Sampling_rate
    pad_U = np.zeros((round(q**2*M), round(q**2*N)))
    [Mp, Np] = np.shape(pad_U)
    pad_U[round(Np/2-N/2):round(Np/2+N/2), round(Mp/2-M/2):round(Mp/2+M/2)] = U0
    [UU, II, dfx] = fresnel_diff_s_fft(pad_U, z, dx, r)        # 初始面光场分布
    [U1, I1, dx] = fresnel_diff_s_fft(UU, -z, dfx, r)       # 重新衍射
    [U2, I2, dx] = fresnel_diff_s_fft(UU, -q * z, dfx, r)   # z = q*z
    [U3, I3, dx] = fresnel_diff_s_fft(UU, -z, q * dfx, r)   # dfx = q*dfx

    plt.figure()
    plt.subplot(131), plt.imshow(np.abs(I1), 'gray'), plt.axis('off'), plt.title('original')
    plt.subplot(132), plt.imshow(np.abs(I2), 'gray'), plt.axis('off'), plt.title('1.2*z')
    plt.subplot(133), plt.imshow(np.abs(I3), 'gray'), plt.axis('off'), plt.title('1.2*dx')

    plt.show()

