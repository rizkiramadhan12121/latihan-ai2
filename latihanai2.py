# Program Prediksi Tipe Kepribadian dengan Tiga Dimensi
# Menggunakan Scikit-learn untuk melatih tiga model terpisah

# Langkah 1: Mengimpor library yang dibutuhkan
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Langkah 2: Menyiapkan Data Pelatihan untuk Tiga Kategori
# Data input (fitur):
# Kolom: [suka_bersosialisasi (1-10), terorganisir (1-10), suka_berpikir_abstrak (1-10)]
# Ini adalah data yang sama untuk semua model
X = np.array([
    [9, 4, 3],  # Tipe 1: Ekstrovert, Spontan, Praktis
    [2, 8, 7],  # Tipe 2: Introvert, Terorganisir, Teoritis
    [8, 3, 2],  # Tipe 1
    [3, 9, 8],  # Tipe 2
    [10, 5, 4], # Tipe 1
    [1, 7, 9],  # Tipe 2
    [7, 6, 5],  # Tipe 1
    [4, 10, 10],# Tipe 2
])

# Label Target untuk Kategori Ekstrovert (E) vs. Introvert (I)
# 1 = E, 0 = I
y_ei = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Label Target untuk Kategori Spontan (S) vs. Terorganisir (T)
# 1 = S, 0 = T
y_st = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Label Target untuk Kategori Praktis (P) vs. Teoritis (T)
# 1 = P, 0 = T
y_pt = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Langkah 3: Melatih Tiga Model AI Terpisah
# Model untuk memprediksi Ekstrovert/Introvert
model_ei = DecisionTreeClassifier()
model_ei.fit(X, y_ei)

# Model untuk memprediksi Spontan/Terorganisir
model_st = DecisionTreeClassifier()
model_st.fit(X, y_st)

# Model untuk memprediksi Praktis/Teoritis
model_pt = DecisionTreeClassifier()
model_pt.fit(X, y_pt)

# Langkah 4: Meminta input dari pengguna dan melakukan prediksi
print("--- Prediksi Tipe Kepribadian Berdasarkan Tiga Dimensi ---")
print("Jawab pertanyaan berikut dengan skala 1-10 (1 = Sangat tidak setuju, 10 = Sangat setuju)")
try:
    skor_sosialisasi = float(input("1. Seberapa suka Anda bersosialisasi?: "))
    skor_organisir = float(input("2. Seberapa terorganisir Anda?: "))
    skor_abstrak = float(input("3. Seberapa suka Anda ide-ide abstrak?: "))

    # Mengubah input menjadi format yang bisa diproses oleh model
    data_pengguna = np.array([[skor_sosialisasi, skor_organisir, skor_abstrak]])
    
    # Melakukan prediksi dengan tiga model AI
    prediksi_ei = model_ei.predict(data_pengguna)
    prediksi_st = model_st.predict(data_pengguna)
    prediksi_pt = model_pt.predict(data_pengguna)
    
    # Menampilkan hasil prediksi
    print("\n--- Hasil Prediksi AI ---")
    
    # Menentukan hasil Ekstrovert/Introvert
    if prediksi_ei[0] == 1:
        tipe_ei = "Ekstrovert (E)"
    else:
        tipe_ei = "Introvert (I)"
    
    # Menentukan hasil Spontan/Terorganisir
    if prediksi_st[0] == 1:
        tipe_st = "Spontan (S)"
    else:
        tipe_st = "Terorganisir (T)"
        
    # Menentukan hasil Praktis/Teoritis
    if prediksi_pt[0] == 1:
        tipe_pt = "Praktis (P)"
    else:
        tipe_pt = "Teoritis (T)"
        
    print(f"Berdasarkan input, Anda diprediksi memiliki tipe kepribadian:")
    print(f"1. E/I: {tipe_ei}")
    print(f"2. S/T: {tipe_st}")
    print(f"3. P/T: {tipe_pt}")
        
except ValueError:
    print("\nInput tidak valid. Pastikan Anda memasukkan angka.")

print("\n----------------------------------")
print("Catatan: Ini adalah contoh sederhana. Hasil prediksi tidak akurat dan bukan diagnosis psikologis.")