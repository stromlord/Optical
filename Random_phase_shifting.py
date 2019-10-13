from Holography_Simulation import *


def hologram_design(M, N, dx, phase_shifting=0):
    '''
    全息图生成
    :param M, N:    全息图尺寸
    :param dx:      像素尺寸
    :param phase_shifting:  相移值
    :return:        全息图
    '''
    x = np.linspace(-(dx * N) / 2, (dx * N) / 2, N)
    y = np.linspace(-(dx * M) / 2, (dx * M) / 2, M)
    [x, y] = np.meshgrid(x, y)

    I0 = 145 * np.exp(-0.2 * (x ** 2 + y ** 2))
    I_p = 100 * np.exp(-0.2 * (x ** 2 + y ** 2))
    I_phase = 2 * np.pi * (x ** 2 + y ** 2)
    hologram = I0 + I_p * np.cos(I_phase + phase_shifting)

    return hologram


def GPSA(hologram, theta, Num=4):
    '''
    通用移相算法
    Thesis: 苏志德 (2013): 高精度干涉测量随机移相技术研究.
    博士. 中国科学院大学. P31
    Available online at http://www.wanfangdata.com.cn/details/detail.do?_type=degree&id=Y2369285.
    :param hologram:    全息图列表
    :param theta:       相移值列表
    :param N:           全息图个数
    :return:            振幅，相位
    '''
    B_1 = np.zeros((M, N))
    B_2 = np.zeros((M, N))
    B_3 = np.zeros((M, N))
    A = np.zeros((3, 3))
    A[0][0] = Num
    for i in range(4):
        A[0][1] = A[0][1] + np.cos(theta[i])
        A[0][2] = A[0][2] + np.sin(theta[i])
        A[1][0] = A[1][0] + np.cos(theta[i])
        A[1][1] = A[1][1] + np.cos(theta[i]) * np.cos(theta[i])
        A[1][2] = A[1][2] + np.cos(theta[i]) * np.sin(theta[i])
        A[2][0] = A[2][0] + np.sin(theta[i])
        A[2][1] = A[2][1] + np.sin(theta[i]) * np.cos(theta[i])
        A[2][2] = A[2][2] + np.sin(theta[i]) * np.sin(theta[i])

        B_1 = B_1 + hologram[i]
        B_2 = B_2 + hologram[i] * np.cos(theta[i])
        B_3 = B_3 + hologram[i] * np.sin(theta[i])

    A_inv = np.linalg.inv(A)

    B = [B_1, B_2, B_3]
    X_1 = 0
    X_2 = 0
    X_3 = 0
    for i in range(3):
        X_1 = X_1 + A_inv[0, i] * B[i]
        X_2 = X_2 + A_inv[1, i] * B[i]
        X_3 = X_3 + A_inv[2, i] * B[i]

    amplitude = X_1
    phase = np.arctan(-X_3 / X_2)
    X = [X_1, X_2, X_3]

    # plt.figure()
    # plt.subplot(121), plt.imshow(np.abs(amplitude), 'gray'), plt.axis('off'), plt.title('Amplitude')
    # plt.subplot(122), plt.imshow(np.abs(phase), 'gray'), plt.axis('off'), plt.title('Phase')
    # plt.show()

    return phase


def Least_square_phase_shifting(hologram, phase):
    A = np.zeros((3, 3))
    A[0][0] = M * N
    num = np.shape(hologram)[0]
    B = np.zeros((num, 3))
    for m in range(M):
        for n in range(N):
            A[0][1] = A[0][1] + np.cos(phase[m][n])
            A[0][2] = A[0][2] + np.sin(phase[m][n])
            A[1][0] = A[1][0] + np.cos(phase[m][n])
            A[1][1] = A[1][1] + np.cos(phase[m][n]) * np.cos(phase[m][n])
            A[1][2] = A[1][2] + np.sin(phase[m][n]) * np.cos(phase[m][n])
            A[2][0] = A[2][0] + np.sin(phase[m][n])
            A[2][1] = A[2][1] + np.sin(phase[m][n]) * np.cos(phase[m][n])
            A[2][2] = A[2][2] + np.sin(phase[m][n]) * np.sin(phase[m][n])

            for step in range(num):
                B[step][0] = B[step][0] + hologram[step][m][n]
                B[step][1] = B[step][1] + hologram[step][m][n] * np.cos(phase[m][n])
                B[step][2] = B[step][2] + hologram[step][m][n] * np.sin(phase[m][n])

    A_inv = np.linalg.inv(A)
    X = []
    delta = []
    for i in range(num):
        X.append(np.dot(A_inv, B[i][:]))
        delta.append(np.arctan(-X[i][2] / X[i][1]))
    print(delta)



def AIA(phase, alpha):
    '''
    Wang, Zhaoyang; Han, Bongtae (2004):
    Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms.
    In Opt. Lett. 29 (14), pp. 1671–1673.
    DOI: 10.1364/OL.29.001671.
    :return:
    '''
    [M, N] = np.shape(phase)



if __name__ == "__main__":
    r = 0.532e-3  # 波长
    k = 2 * np.pi / r  # 波矢
    pixel = 6.4e-3  # 像素间隔

    # 模拟全息图
    # M = 512
    # N = 512
    # theta = [0, np.pi / 3, np.pi / 2, np.pi]
    #
    # hologram = []
    # for i in range(4):
    #     hologram.append(hologram_design(M, N, pixel, phase_shifting=theta[i]))

    # 模拟干涉全息图
    I0 = cv2.imread('Imgs/lenna.bmp', 0)
    I0 = I0 / 255  # 归一化
    I0 = I0 * np.exp(1j * np.pi * I0 / np.max(I0))  # 添加相位
    [M, N] = np.shape(I0)
    I_filter = low_filter(I0, M / 6)

    theta = [0, np.pi/3, np.pi/2, 3 * np.pi / 2]
    ref = []
    holograms = []
    for i in range(4):
        ref.append(reference_beam(M, N, pixel, k, shift_phase=theta[i]))
        hologram = np.abs(interfence(I_filter, ref[i]))
        hologram = hologram / np.max(hologram)
        holograms.append(hologram)

    # plt.figure()
    # plt.subplot(221), plt.imshow(np.abs(hologram[0]), 'gray'), plt.axis('off')
    # plt.subplot(222), plt.imshow(np.abs(hologram[1]), 'gray'), plt.axis('off')
    # plt.subplot(223), plt.imshow(np.abs(hologram[2]), 'gray'), plt.axis('off')
    # plt.subplot(224), plt.imshow(np.abs(hologram[3]), 'gray'), plt.axis('off')
    # plt.show()

    alpha = [0, np.pi / 3, np.pi/2, np.pi]      # 预估初始相移值

    phase = GPSA(holograms, theta, 4)
    Least_square_phase_shifting(holograms, phase)


