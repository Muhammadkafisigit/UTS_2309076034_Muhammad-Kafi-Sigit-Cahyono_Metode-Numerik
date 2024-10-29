import numpy as np
import matplotlib.pyplot as plt

# Konstanta
L = 0.5  # Induktansi dalam Henries
C = 10e-6  # Kapasitansi dalam Farads
target_f = 1000  # Frekuensi target dalam Hertz
tolerance = 0.1  # Toleransi kesalahan dalam Ohm

# Fungsi untuk menghitung frekuensi resonansi berdasarkan resistansi R
def f_R(R):
    term = 1 / (L * C) - (R*2) / (4 * L*2)
    if term <= 0:
        return None  # Jika term negatif, kembalikan None
    return (1 / (2 * np.pi)) * np.sqrt(term)

# Turunan dari f(R) untuk metode Newton-Raphson
def f_prime_R(R):
    term = 1 / (L * C) - (R*2) / (4 * L*2)
    if term <= 0:
        return None  # Jika turunan tidak terdefinisi, kembalikan None
    sqrt_term = np.sqrt(term)
    return -R / (4 * np.pi * L**2 * sqrt_term)

# Metode Newton-Raphson
def newton_raphson_method(initial_guess, tolerance):
    R = initial_guess
    while True:
        f_val = f_R(R)
        if f_val is None:
            return None  # Kasus tidak valid
        f_value = f_val - target_f
        f_prime_value = f_prime_R(R)
        if f_prime_value is None:
            return None  # Kasus tidak valid
        new_R = R - f_value / f_prime_value
        if abs(new_R - R) < tolerance:
            return new_R
        R = new_R

# Metode Bisection
def bisection_method(a, b, tolerance):
    while (b - a) / 2 > tolerance:
        mid = (a + b) / 2
        f_mid = f_R(mid) - target_f
        if f_mid is None:
            return None  # Kasus tidak valid
        if abs(f_mid) < tolerance:
            return mid
        if (f_R(a) - target_f) * f_mid < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

# Eksekusi kedua metode
initial_guess = 50  # Tebakan awal untuk Newton-Raphson
interval_a, interval_b = 0, 100  # Interval untuk Bisection

# Hasil dari Newton-Raphson
R_newton = newton_raphson_method(initial_guess, tolerance)
f_newton = f_R(R_newton) if R_newton is not None else "Tidak ditemukan"

# Hasil dari metode Bisection
R_bisection = bisection_method(interval_a, interval_b, tolerance)
f_bisection = f_R(R_bisection) if R_bisection is not None else "Tidak ditemukan"

# Tampilkan hasil
print("Metode Newton-Raphson:")
print(f"Nilai R: {R_newton} ohm, Frekuensi Resonansi: {f_newton} Hz")

print("\nMetode Bisection:")
print(f"Nilai R: {R_bisection} ohm, Frekuensi Resonansi: {f_bisection} Hz")

# Plot hasil
plt.figure(figsize=(10, 5))
plt.axhline(target_f, color="purple", linestyle="--", label="Frekuensi Target 1000 Hz")

# Plot hasil Newton-Raphson
if R_newton is not None:
    plt.scatter(R_newton, f_newton, color="orange", label="Newton-Raphson", zorder=5)
    plt.text(R_newton, f_newton + 30, f"NR: R={R_newton:.2f}, f={f_newton:.2f} Hz", color="orange")

# Plot hasil Bisection
if R_bisection is not None:
    plt.scatter(R_bisection, f_bisection, color="teal", label="Bisection", zorder=5)
    plt.text(R_bisection, f_bisection + 30, f"Bisection: R={R_bisection:.2f}, f={f_bisection:.2f} Hz", color="teal")

# Labeling plot
plt.xlabel("Nilai R (Ohm)")
plt.ylabel("Frekuensi Resonansi f(R) (Hz)")
plt.title("Perbandingan Metode Newton-Raphson dan Bisection")
plt.legend()
plt.grid(True)
plt.show()

# Perbandingan Kesalahan untuk Berbagai Metode Numerik

# Fungsi untuk menghitung R berdasarkan suhu T
def R(T):
    return 5000 * np.exp(3500 * (1/T - 1/298))

# Metode diferensiasi numerik

# Metode beda maju
def forward_difference(T, h):
    return (R(T + h) - R(T)) / h

# Metode beda mundur
def backward_difference(T, h):
    return (R(T) - R(T - h)) / h

# Metode beda pusat
def central_difference(T, h):
    return (R(T + h) - R(T - h)) / (2 * h)

# Penghitungan turunan eksak
def exact_derivative(T):
    return 5000 * np.exp(3500 * (1/T - 1/298)) * (-3500 / T**2)

# Rentang suhu dan interval
temperatures = np.arange(250, 351, 10)
h = 1e-3  # Interval kecil untuk perbedaan

# Mengumpulkan hasil untuk setiap metode
results = {
    "Suhu (K)": temperatures,
    "Beda Maju": [forward_difference(T, h) for T in temperatures],
    "Beda Mundur": [backward_difference(T, h) for T in temperatures],
    "Beda Pusat": [central_difference(T, h) for T in temperatures],
    "Turunan Eksak": [exact_derivative(T) for T in temperatures],
}

# Menghitung kesalahan relatif
errors = {
    "Kesalahan Beda Maju": np.abs((np.array(results["Beda Maju"]) - np.array(results["Turunan Eksak"])) / np.array(results["Turunan Eksak"])) * 100,
    "Kesalahan Beda Mundur": np.abs((np.array(results["Beda Mundur"]) - np.array(results["Turunan Eksak"])) / np.array(results["Turunan Eksak"])) * 100,
    "Kesalahan Beda Pusat": np.abs((np.array(results["Beda Pusat"]) - np.array(results["Turunan Eksak"])) / np.array(results["Turunan Eksak"])) * 100,
}

# Memplot kesalahan relatif
plt.figure(figsize=(10, 6))
plt.plot(temperatures, errors["Kesalahan Beda Maju"], label="Kesalahan Beda Maju", marker='o')
plt.plot(temperatures, errors["Kesalahan Beda Mundur"], label="Kesalahan Beda Mundur", marker='s')
plt.plot(temperatures, errors["Kesalahan Beda Pusat"], label="Kesalahan Beda Pusat", marker='^')
plt.xlabel("Suhu (K)")
plt.ylabel("Kesalahan Relatif (%)")
plt.legend()
plt.title("Kesalahan Relatif dari Metode Turunan Numerik dibandingkan Turunan Eksak")
plt.grid()
plt.show()