import matplotlib.pyplot as plt
import numpy as np
from dataset.dataloader import data_load

from mpl_toolkits.mplot3d import Axes3D
# 定义figure
fig = plt.figure()

from scipy import interpolate

def main3():
    x = np.arange(-5.01, 5.01, 0.25)
    y = np.arange(-5.01, 5.01, 0.25)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2+yy**2)
    f = interpolate.interp2d(x, y, z, kind='cubic')



    xnew = np.arange(-5.01, 5.01, 1e-2)
    ynew = np.arange(-5.01, 5.01, 1e-2)
    znew = f(xnew, ynew)
    
    
    X,Y = np.meshgrid(x,y)
    ZNEW = f(X,Y)

    plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
    plt.show()
  
def main2():
    x = np.asarray([0,1,2])
    y = np.asarray([0, 3])  #! 注意: 这里是正序
    z = np.asarray([[1,2,3], [4,5,6]])
    xx, yy = np.meshgrid(x, y)
    plt.contourf(xx, yy, z)
    plt.show()
    # interp
    xn = np.linspace(0, 2, num=8)
    yn = np.linspace(0, 3, num=6)  #! 注意: 这里是正序
    interpfunc = interpolate.interp2d(x, y, z, kind='linear')
    zn = interpfunc(xn, yn)
    xxn, yyn = np.meshgrid(xn, yn)
    plt.contourf(xxn, yyn, zn)  # 展示插值结果
    plt.show()


def main():
    A1 = data_load('A1_h_CW_detection')
    A2 = data_load('A2_h_CW_detection')
    A3 = data_load('A3_h_CW_detection')
    A1p = data_load('A1_h_CW_prediction')
    a1_COp = np.array(A1p['CO'])
    a1_CO =  np.array(A1['CO'])
    plt.ylim([-1,6])
    plt.subplot(2,1,1)
    plt.plot(a1_COp,'r*')
    plt.ylim([-1,6])
    plt.subplot(2,1,2)
    plt.plot(a1_CO,'b*')

    plt.show()


    pass



if __name__ == '__main__':
    main()