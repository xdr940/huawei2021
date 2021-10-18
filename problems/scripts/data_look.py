import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nf
from dataset import data_load

feat = 'O3'


def plotfft(arr):
    plt.subplot(2, 1, 1)
    plt.plot(arr)
    plt.subplot(2, 1, 2)

    plt.ylim([0,400])
    plt.xlim([0,300])

    comp_arr = nf.fft(arr)
    y2 = nf.ifft(comp_arr).real
    freqs = nf.fftfreq(arr.size, 1/24/24)
    pows = np.abs(comp_arr)  # 复数的模
    plt.plot(freqs[freqs > 0], pows[freqs > 0], color='orangered', label='frequency')

    plt.show()
def main():
    df = data_load('A_h_CW_detection')
    arr =  np.array(df[feat])

    arr = (arr - arr.min())/(arr.max()-arr.min())
    plotfft(arr)


def main2():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y = np.zeros(x.size)
    for i in range(1, 1000):
        y += 4 * np.pi / (2 * i - 1) * np.sin((2 * i - 1) * x)

    plt.figure('FFT', facecolor='lightgray')
    plt.subplot(121)
    plt.title('Time Domain', fontsize=16)
    plt.grid(linestyle=':')
    plt.plot( x,y, label=r'$y$')
    # 针对方波y做fft
    comp_arr = nf.fft(y)
    y2 = nf.ifft(comp_arr).real
    plt.plot(x, y2, color='orangered', linewidth=5, alpha=0.5, label=r'$y$')
    # 绘制频域图形
    plt.subplot(122)
    freqs = nf.fftfreq(y.size, x[1] - x[0])
    pows = np.abs(comp_arr)  # 复数的模
    plt.title('Frequency Domain', fontsize=16)
    plt.grid(linestyle=':')
    plt.plot(freqs[freqs > 0], pows[freqs > 0], color='orangered', label='frequency')

    plt.legend()
    plt.savefig('fft.png')
    plt.show()
if __name__ == '__main__':
    main()
