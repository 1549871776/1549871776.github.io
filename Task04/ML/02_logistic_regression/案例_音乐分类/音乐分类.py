import numpy as np
from scipy.fft import fft  # 引入傅里叶变换提取声音信息
from scipy.io import wavfile  # 读取音乐文件
from sklearn.linear_model import LogisticRegression


# 加载训练集数据，分割训练集以及测试集，进行分类器的训练
genre_list = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']
X = []
Y = []
for g in genre_list:
    for n in range(100):
        rad = r"G:/data_home/trainset/" + g + "." + str(n).zfill(5) + ".fft.npy"
        fft_features = np.load(rad)
        X.append(fft_features)
        Y.append(genre_list.index(g))  # index索引

X = np.array(X)
Y = np.array(Y)


print("starting build model...")
# model = LogisticRegression(max_iter=1000)  # 调大迭代次数，使其收敛
model = LogisticRegression(max_iter=1000)
model.fit(X, Y)
print("build model successfully")

# 存储模型
# import pickle
# s = pickle.dumps(model)
# clf2 = pickle.loads(s)
# clf2.predict(X[0])

print('starting read wavfile...')
sample_rate, test = wavfile.read('G:/data/trainset/sample/heibao-wudizirong-remix.wav')
sample_rate_1, test_1 = wavfile.read('G:/data/trainset/sample/sine_mix.wav')

testdata_fft_features = abs(fft(test))[:1000]
testdata_fft_features_1 = abs(fft(test_1))[:1000]
# print(sample_rate, testdata_fft_features, len(testdata_fft_features))

type_index = model.predict([testdata_fft_features])[0]
type_index_1 = model.predict([testdata_fft_features_1])[0]
print("读取音乐的种类是：", genre_list[type_index])
print("读取音乐的种类是：", genre_list[type_index_1])


