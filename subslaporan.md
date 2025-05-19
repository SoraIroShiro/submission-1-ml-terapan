# Laporan Proyek Machine Learning - Mochammad Fiqi Fahrudillah

## Project Overview

Perubahan suhu harian di kota-kota besar Asia, seperti Jakarta, menjadi isu penting yang berdampak pada berbagai sektor, mulai dari kesehatan masyarakat, transportasi, hingga perencanaan kota. Prediksi suhu yang akurat sangat dibutuhkan untuk mendukung pengambilan keputusan dan mitigasi risiko cuaca ekstrem.  
Proyek ini bertujuan membangun model machine learning untuk memprediksi suhu rata-rata harian di Jakarta berdasarkan data historis, sehingga dapat memberikan estimasi yang lebih baik untuk kebutuhan masyarakat dan pemerintah.

**Referensi:**  
- [City Temperature Kaggle Dataset](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities)  
- BMKG. (2022). "Perubahan Iklim dan Dampaknya di Indonesia".  
- World Meteorological Organization, 2021.

---

## Business Understanding

### Problem Statements

- Bagaimana memprediksi suhu rata-rata harian di Jakarta secara akurat menggunakan data historis?
- Algoritma machine learning apa yang paling efektif untuk forecasting suhu harian di Jakarta?
- Sejauh mana model machine learning dapat mengungguli metode prediksi sederhana (naive forecast)?

### Goals

- Menghasilkan model prediksi suhu rata-rata harian di Jakarta dengan error serendah mungkin.
- Membandingkan performa model machine learning (XGBoost) dengan baseline naive.
- Memberikan insight fitur apa yang paling berpengaruh terhadap prediksi suhu.

**Solution Approach:**

- Baseline: Naive forecast (prediksi suhu hari ini = suhu kemarin).
- Machine Learning: XGBoostRegressor dengan fitur waktu dan rolling mean.

---

## Data Understanding

Dataset yang digunakan adalah [City Temperature Kaggle Dataset](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities) yang berisi data suhu harian dari berbagai kota besar dunia, termasuk Jakarta, dari tahun 1995 hingga 2020.  
Jumlah data untuk Jakarta: lebih dari 9.000 baris (setiap baris = 1 hari).

**Fitur utama pada dataset:**
- `Region`, `Country`, `State`, `City`: Identitas lokasi
- `Month`, `Day`, `Year`: Informasi waktu
- `AvgTemperature`: Suhu rata-rata harian (Fahrenheit)
- `Date`: Kolom tanggal hasil penggabungan

**EDA singkat:**  
Visualisasi tren suhu, distribusi suhu, dan statistik deskriptif dilakukan untuk memahami pola musiman dan outlier pada data.

![Visualisasi Prediksi vs Aktual](https://raw.githubusercontent.com/SoraIroShiro/submission-1-ml-terapan/0a5794865d7d0d31a355f2bf190559089add7219/prediksivsactual.png)

---

## Data Preparation

- **Pembersihan data:** Menghapus data tidak valid (`AvgTemperature <= -50`).
- **Konversi suhu:** Fahrenheit ke Celsius.
- **Pembuatan fitur waktu:** Bulan (`Month`), hari dalam minggu (`DayOfWeek`).
- **Fitur rolling mean:** Rata-rata suhu 7 hari terakhir (`RollingMean_7`).
- **Split data:** Data diurutkan berdasarkan tanggal, lalu dibagi train-test (80:20) tanpa shuffle.

---

## Modeling

- **Baseline:** Naive forecast (prediksi suhu hari ini = suhu kemarin).
- **Machine Learning:** XGBoostRegressor dengan fitur waktu dan rolling mean.
- **Evaluasi:** Menggunakan MAE dan RMSE.

**Contoh kode baseline:**
```python
df_city['Naive_Pred'] = df_city['AvgTemperature_C'].shift(1)
df_eval = df_city.dropna(subset=['Naive_Pred'])
mae_naive = mean_absolute_error(df_eval['AvgTemperature_C'], df_eval['Naive_Pred'])
rmse_naive = np.sqrt(mean_squared_error(df_eval['AvgTemperature_C'], df_eval['Naive_Pred']))
```

**Contoh kode XGBoost:**
```python
features = ['Month', 'DayOfWeek', 'RollingMean_7']
X = df_ml[features]
y = df_ml['AvgTemperature_C']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae_ml = mean_absolute_error(y_test, y_pred)
rmse_ml = np.sqrt(mean_squared_error(y_test, y_pred))
```

---

## Evaluation

- **Baseline Naive:**  
  MAE: 0.85°C, RMSE: 1.14°C
- **XGBoostRegressor:**  
  MAE: 0.65°C, RMSE: 0.84°C

Visualisasi prediksi vs aktual menunjukkan model mampu mengikuti pola suhu harian dengan baik.

![Visualisasi Prediksi vs Aktual](https://raw.githubusercontent.com/SoraIroShiro/submission-1-ml-terapan/0a5794865d7d0d31a355f2bf190559089add7219/prediksivsactual.png)

**Feature Importance:**  
Fitur rolling mean 7 hari paling berpengaruh, diikuti oleh bulan dan hari dalam minggu.

---

## Kesimpulan

- Model XGBoost mampu memprediksi suhu rata-rata harian di Jakarta dengan performa lebih baik dibanding baseline naive.
- Fitur waktu dan rolling mean sangat membantu model dalam mengenali pola musiman dan tren suhu.
- Model ini dapat digunakan untuk mendukung perencanaan aktivitas berbasis cuaca di Jakarta.

---

## Saran Pengembangan

- Lakukan tuning hyperparameter pada model XGBoost untuk meningkatkan akurasi prediksi.
- Tambahkan fitur eksternal seperti kelembapan, curah hujan, atau event khusus.
- Benchmarking dengan model lain seperti Random Forest, Linear Regression, atau ARIMA.
- Terapkan model pada kota lain di Asia untuk menguji generalisasi model.

---

## Referensi

- Dataset: [City Temperature Kaggle Dataset](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities)
- Dokumentasi XGBoost: https://xgboost.readthedocs.io/
- Dokumentasi scikit-learn: https://scikit-learn.org/
- BMKG: https://www.bmkg.go.id/