# Tugas 1 Machine Learning

Pada Tugas 1 ini, Anda akan bekerja dengan sebuah dataset berisi informasi mahasiswa dari berbagai program studi. Dataset ini sengaja dirancang dengan beberapa permasalahan umum dalam dunia nyata, seperti data yang hilang (missing values), inkonsistensi format data, serta tipe data kategorikal yang belum siap diproses oleh model machine learning.

Anda diharapkan dapat menerapkan langkah-langkah data preprocessing untuk menyiapkan dataset ini agar siap digunakan dalam proses pelatihan model machine learning.

## Petunjuk Pengerjaan

Kerjakan tugas1 ini secara berurutan sesuai dengan tahapan berikut:

1. Load Dataset
   - Impor library yang diperlukan
   - Baca file dataset_tugas1_preprocessing.csv ke dalam DataFrame
2. Eksplorasi Awal (EDA)
   - Cek struktur dan tipe data
   - Tampilkan jumlah nilai hilang per kolom
   - Sajikan statistik deskriptif
3. Menangani Missing Values
   - Imputasi nilai kosong pada kolom Nilai_Akhir dan Umur dengan metode yang sesuai
4. Normalisasi Format Tanggal
   - Pastikan seluruh data pada kolom Tanggal_Ujian terkonversi ke format datetime
5. Encoding Label
   - Lakukan encoding (Label Encoding) pada kolom kategorikal: Nama, Jenis_Kelamin, Prodi, Status, Nilai_Akhir
6. Split Data
   - Bagi data menjadi data latih dan data uji (80% - 20%)
7. Visualisasi
   - Buat histogram untuk kolom Umur
   - Buat grafik batang jumlah mahasiswa per Prodi

Pengumpulan:
- Simpan hasil pengerjaan Anda dalam format .ipynb (jika ingin diunggah di Tuton) dan atau ke upload filenya github dan infokan link untuk diperiksa Tutor.

---

```yaml
Nama: Muhammad Hidayat
NIM: 052747132
```
Hasil pengerjaan beserta penjelasan lengkap dapat dilihat di file Jupyter Notebook (Google Colab) di GitHub berikut: <https://github.com/return215/machine-learning-tugas-ut>. Lokasi tuugas ada di folder "tugas1".

Environment dibuat dengan perintah:

```bash
conda create --name "$(basename "$PWD")" python
```
