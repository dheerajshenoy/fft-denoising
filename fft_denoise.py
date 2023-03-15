import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import pi
import scipy as sp
from scipy.fft import fft, ifft
from scipy.io.wavfile import write
import sympy as smp
from sympy.abc import x as sym_x

plt.rcParams['axes.grid'] = True
plt.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(4, 1)

SR = 44100 # Sampling rate
T = 1/SR # Time period
t = np.arange(0, 1, T) # time space
x = 0

# for i in range(100):
#     amp = np.random.randint(100)
#     freq = np.random.randint(10)
#     x += amp * np.sin(2 * pi * freq * t)

# Waves

F = [20, 40, 60, 80, 200, 800, 1500, 1200]
A = [10, 5, 0.5, 10, 20, 100, 50, 20]

for i in range(len(F)):
    k = 2 * pi * F[i] * t
    x += A[i] * np.sin(k)

def sound(data, fileName, sampling_rate):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(fileName, sampling_rate, scaled)

sound(x, 'test1.wav', SR)

ax[0].plot(t, x, 'b')
ax[0].set_xlabel("t (sec)")
ax[0].set_ylabel("x(t)")

X = fft(x)
N = len(X)
n = np.arange(N)
T = N/SR
X = 2 * np.abs(X)/SR
freq = n/T
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlim(20, 1000)
ax[1].stem(freq, X, 'r', markerfmt="")

for i in range(len(freq)):
    if freq[i] >= 800:
        X[i] = 0

ax[2].set_xlabel('Freq (Hz)')
ax[2].set_ylabel('Amplitude')
ax[2].set_xlim(20, 1000)
ax[2].stem(freq, X, 'r', markerfmt="")


w = ifft(X * SR/2)
ax[3].set_ylabel('x(t)')
ax[3].set_xlabel('t')
ax[3].plot(t, w, 'b')

sound(w, 'test2.wav', SR)

plt.show()
