from matplotlib import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G = 9.8


def data_generate(n_sample: int, v0: float, sigma: float = 1,
                  decimals=2, seed=None):
    '''
    n_sample: số lượng dữ liệu

    v0: vận tốc ban đầu

    sigma: độ lệch chuẩn của sai số

    seed: random seed

    return: vị trí theo trục y, thời gian
    '''
    np.random.seed(seed)

    time_of_fall = 2 * v0 / G

    epsilon = np.random.normal(scale=sigma, size=n_sample)

    t = np.random.uniform(low=0, high=time_of_fall, size=n_sample)

    # tính y theo công thức và thêm sai số
    y = v0*t - (G/2)*np.power(t, 2) + epsilon

    y[y < 0] = 0

    # thêm sai số dụng cụ
    t = t.round(decimals)
    y = y.round(decimals)

    return y, t


def to_csv(y, t, file_path, index=False):
    df = pd.DataFrame({"t": t, "y": y})

    df.to_csv(file_path, index=index)


if __name__ == "__main__":
    y_train, t_train = data_generate(n_sample=500, v0=10, sigma=.15, seed=188)
    y_test, t_test = data_generate(n_sample=200, v0=10, sigma=.1, seed=202)

    # plt.plot(t_train, y_train, 'ro')
    # plt.plot(t_test, y_test, 'gx')
    # plt.show()

    to_csv(y_train, t_train, "./data/train.csv")
    to_csv(y_test, t_test, "./data/test.csv")
