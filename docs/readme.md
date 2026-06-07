# Customer Personality Analysis: Segmentasi Pelanggan Menggunakan Unsupervised Learning
Muhammad Hidayat
2026-05-28

# Tugas Machine Learning Universitas Terbuka

``` yaml
Nama: Muhammad Hidayat
NIM: 052747132
```

## Daftar Tugas

- [Tugas 1](./tugas1/)
- [Tugas 2](./tugas2/)

Masing-masing file tugas dapat dijalankan secara terpisah dan dapat
diexport ke format notebook (ipynb), Markdown, HTML, dan PDF.

## Persiapan Environment

Project ini menggunakan [Quarto](https://quarto.org/) dan
[Anaconda](https://www.anaconda.com). Ikuti panduan dari
[Quarto](https://quarto.org/docs/get-started/) dan
[Anaconda](https://www.anaconda.com/docs/getting-started/main) untuk
menginstal Quarto dan Conda.

Buat dan aktifkan lingkungan Conda:

``` bash
conda env create -f environment.yml
conda activate quarto-python-project
```

Jalankan rendering:

``` bash
quarto render
```

Jalankan rendering spesifik (misal untuk tugas 1 sebagai notebook):

``` bash
quarto render -t ipynb tugas1/index.qmd
```

``` yaml
Nama: Muhammad Hidayat
NIM: 052747132
```

``` {python}
#| label: setup

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set_theme(style="whitegrid")
```

# Step 1: Memuat Dataset

File `cpa.csv` sebenarnya merupakan tab-delimited, jadi kita perlu
menyebutkan `sep='\t'` saat membacanya dengan `pd.read_csv()`.

``` {python}
#| label: load-dataset

data = pd.read_csv('cpa.csv', sep='\t')
display(data.head())
```

# Step 2: Eksplorasi Data & Pembersihan Awal

Beberapa langkah eksplorasi dan pembersihan awal yang bisa dilakukan:

- Cek tipe data dan jumlah missing value per kolom
- Cek statistik untuk kolom numerik
- Cek jumlah kategori unik untuk kolom kategori

Pertama, kita akan melihat tipe data dan jumlah missing value:

``` {python}
#| label: check-dtypes

display(data.dtypes)
```

``` {python}
#| label: check-missing

missing_info = data.isnull().sum()
print("Kolom dengan nilai kosong:")
print(missing_info[missing_info > 0])
```

Terdapat 24 baris kosong di kolom `Income`. Kita akan melakukan imputasi
untuk kolom ini nanti.

Selanjutnya, kita bisa melihat statistik untuk kolom numerik, dipecah
per 9 kolom untuk memudahkan pembacaan:

``` {python}
#| label: describe-numeric

numeric_cols = data.select_dtypes(include=[np.number]).columns
for i in range(0, len(numeric_cols), 9):
    display(data[numeric_cols[i:i+9]].describe())
```

Kolom `Z_CostContact` dan `Z_Revenue` menunjukkan standar deviasi nol,
yang berarti semua nilai di kolom tersebut sama. Kita bisa menghapus
kedua kolom ini karena tidak memberikan informasi yang berguna untuk
klasterisasi. Kolom `ID` juga tidak relevan untuk analisis klasterisasi,
jadi kita akan menghapusnya juga.

Terakhir, kita bisa melihat jumlah kategori unik untuk kolom kategori:

``` {python}
#| label: unique-categories

categorical_cols = data.select_dtypes(include=['str']).columns
for col in categorical_cols:
    print(f"Kolom '{col}' memiliki {data[col].nunique()} kategori unik.")
```

Kolom ‘Education’ memiliki 5 kategori unik dan kolom ‘Marital_Status’
memiliki 8. Kolom ‘Dt_Customer’, yang memiliki 663 kategori unik, akan
kita anggap sebagai kolom tersendiri dan akan diproses sebagai tanggal.

# Step 3: Identifikasi kolom numerik & kategori

Dari langkah eksplorasi sebelumnya, kita sudah mengidentifikasi kolom
numerik dan kategori. Untuk sebagian besar kolom, bisa saja digunakan
apa adanya. Untuk kolom `Dt_Customer`, kita bisa mengubahnya menjadi
fitur numerik dengan menghitung jumlah hari sejak tanggal tersebut
hingga tanggal terakhir dalam dataset. Sama halnya dengan kolom
`Year_Birth`, kita bisa mengubahnya menjadi umur dengan menghitung
selisih antara tahun saat ini dan tahun lahir. Data yang telah diubah
ini akan kita simpan dalam variabel baru agar tidak tercampur dengan
data asli.

Untuk mempermudah, kita akan menyimpan nama-nama kolom numerik dan
kategori dalam variabel terpisah:

``` {python}
#| label: feature-engineering
#| output: asis

# Kolom yang akan dihapus
cols_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']

data_processed = data.drop(columns=cols_to_drop)

# Proses fitur 'Dt_Customer' menjadi 'Customer_Since_Days' dan 'Year_Birth' menjadi 'Age'
data_processed['Dt_Customer'] = pd.to_datetime(data_processed['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
_max_date = data_processed['Dt_Customer'].max()
data_processed['Customer_Since_Days'] = (_max_date - data_processed['Dt_Customer']).dt.days
data_processed['Age'] = 2024 - data_processed['Year_Birth']

# Hapus kolom asli setelah diubah, gunakan inplace=True agar langsung diterapkan
data_processed.drop(columns=['Dt_Customer', 'Year_Birth'], inplace=True)

# Kelomokan nama kolom numerik dan kategori
numerical_cols = data_processed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = data_processed.select_dtypes(include=['str']).columns.tolist()

print(f"Kolom Numerik ({len(numerical_cols)}):\n")
for col in numerical_cols:
    print(f"- {col}")

print(f"\nKolom Kategori ({len(categorical_cols)}):\n")
for col in categorical_cols:
    print(f"- {col}")
```

# Step 4: Imputasi missing value

Kita akan melakukan imputasi untuk kolom `Income` yang memiliki 24 nilai
kosong. Karena `Income` adalah kolom numerik, kita bisa menggunakan
strategi imputasi seperti median atau mean. Dalam kasus ini, kita akan
menggunakan median karena lebih tahan terhadap outlier.

``` {python}
#| label: impute-income

imputer = SimpleImputer(strategy='median')
data_processed['Income'] = imputer.fit_transform(data_processed[['Income']])

print("Jumlah nilai kosong setelah imputasi:")
print(data_processed['Income'].isnull().sum())
```

Kita tampilkan statistik untuk kolom `Customer_Since_Days` dan `Age`
setelah proses transformasi, serta perbandingan antara kolom `Income`
sebelum dan sesudah imputasi.

``` {python}
#| label: describe-engineered

display(data_processed[['Customer_Since_Days', 'Age']].describe())
```

``` {python}
#| label: compare-income

income_comparison = pd.DataFrame({
    'Income (pre)': data['Income'],
    'Income (post)': data_processed['Income']
})
display(income_comparison.describe())
```

# Step 5: Encoding fitur kategori

Encoding akan menggunakan one-hot encoding pada kolom kategori, kemudian
digabung dengan kolom numerik sebagai dataframe baru. Dengan menggunakan
one-hot dibandingkan label atau ordinal encoding, ini akan mengurangi
bias dalam kalkulasi K-Means.

Dalam proses encoding, kita bisa membuang salah satu variabel dalam
teknik *dummy variable trap*, dan dilakukan dengan argumen
`drop='first'`. Hal ini berguna dalam model regresi linear, namun tidak
perlu dalam klasifikasi, karena berisiko menimbulkan bias terhadap
kategori referensi yang dihapus tersebut karena jaraknya ke kategori
lain dihitung secara asimetris. Oleh karena itu, kita pilih untuk tidak
melakukan ini, dengan menggunakan argumen `drop=None`. Keuntungan cara
ini adalah mempermudah interpretasi PCA.

``` {python}
#| label: one-hot-encoding

encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_categorical = encoder.fit_transform(data_processed[categorical_cols])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))
```

## Gabungkan data numerik & hasil encoding

``` {python}
#| label: merge-features

data_final = pd.concat([data_processed[numerical_cols].reset_index(drop=True), encoded_categorical_df], axis=1)
display(data_final.head())
```

# Step 6: Scaling data

Standardisasi data menggunakan `StandardScaler` berdasarkan Z-score
untuk memastikan semua fitur memiliki skala yang sama, sehingga
algoritma klasterisasi tidak bias terhadap fitur dengan rentang nilai
yang lebih besar.

``` {python}
#| label: scale-features

scaler = StandardScaler()
data_scaled_ndarray = scaler.fit_transform(data_final)

# karena data_scaled_ndarray adalah numpy array, kita buat kembali ke dataframe
data_scaled = pd.DataFrame(data_scaled_ndarray, columns=data_final.columns)
```

# Step 7: Tentukan jumlah cluster optimal (Elbow + Silhouette)

### Penjelasan Metode Elbow dan Silhouette Score

Untuk menentukan jumlah klaster ($k$) terbaik, kita menggunakan
kombinasi dari dua metode evaluasi: Elbow dan Silhouette.

Metode Elbow (Inertia / Within-Cluster Sum of Squares) Inersia mengukur
jumlah kuadrat jarak dari setiap titik data ke centroid klaster mereka.
Nilai inersia akan selalu menurun seiring bertambahnya jumlah klaster.
Pada satu titik, laju penurunan inersia mulai melandai secara
signifikan, dimana menambahkan klaster setelah titik ini hanya
memberikan sedikit penurunan inersia namun meningkatkan kompleksitas
model. Inilah titik “siku” (*elbow*) yang akan kita cari, titik yang
optimal sebagai *trade-off* antara kompleksitas dan efektivitas.

Metode Silhouette Score score mengukur seberapa mirip suatu objek dengan
klasternya sendiri (*kohesi*) dibandingkan dengan klaster lainnya
(*pemisahan*). Nilai skor berkisar antara -1 hingga 1. Nilai silhouette
score yang semakin mendekati 1 menunjukkan bahwa titik data terkelompok
dengan baik, memiliki batas yang jelas, dan terpisah jauh dari klaster
tetangga. Kita memilih $k$ yang menghasilkan puncak atau nilai
silhouette tertinggi.

Metode Elbow terkadang menghasilkan kurva melandai yang subjektif tanpa
titik siku yang jelas. Di sisi lain, Silhouette score cenderung
memberikan skor tertinggi pada $k=2$ karena pemisahan biner paling mudah
dilakukan secara matematis, namun secara bisnis $k=2$ terlalu umum dan
tidak informatif untuk segmentasi pelanggan. Dengan menggabungkan
keduanya, kita mencari titik di mana inersia sudah cukup rendah (setelah
elbow) dan silhouette score memiliki nilai yang relatif tinggi.

Rentang $k$ yang diuji dipilih antara 2 hingga 10. Batas bawah adalah
$k=2$ karena klasterisasi minimal membutuhkan 2 kelompok. Batas atas
dipilih $k=10$ karena dalam segmentasi pelanggan bisnis, memiliki lebih
dari 10 segmen akan terlalu rumit dan tidak praktis untuk merancang
kampanye pemasaran yang spesifik bagi masing-masing kelompok.

``` {python}
#| label: fig-elbow-silhouette
#| fig-cap: "Evaluasi jumlah klaster optimal k = 2 s/d 10 menggunakan Metode Elbow (Inersia) dan Silhouette Score secara berdampingan."

# Menghitung inersia dan silhouette score untuk k = 2 s/d 10
ks = range(2, 11)
inertias = []
silhouettes = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(data_scaled, labels))

# Visualisasi kurva Elbow dan Silhouette Score berdampingan
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontsize=12)
ax1.plot(ks, inertias, 'o-', color=color, linewidth=2, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
ax2.plot(ks, silhouettes, 's-', color=color, linewidth=2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Penentuan Jumlah Klaster Optimal: Metode Elbow & Silhouette Score', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.show()
```

Berdasarkan hasil visualisasi kurva evaluasi inersia (Elbow Method) dan
Silhouette Score di atas, penentuan jumlah klaster optimal dapat
dijelaskan sebagai berikut.

Titik tekukan atau siku (*elbow point*) mulai terbentuk secara jelas
pada area rentang $k = 4$ hingga $k = 5$. Setelah melewati $k = 5$,
penambahan jumlah klaster baru tidak lagi memberikan penurunan inersia
yang signifikan.

Terdapat peningkatan lokal yang signifikan pada $k = 5$ dan $k = 6$,
dengan nilai skor yang relatif stabil dan tinggi. Meskipun $k=2$ secara
teoritis memiliki nilai silhouette yang tinggi, jumlah segmen tersebut
terlalu sedikit untuk menghasilkan segmentasi profil kepribadian
pelanggan yang bermakna bagi tim bisnis.

Dengan mempertimbangkan keseimbangan antara performa model (inersia
rendah dan silhouette score tinggi) serta nilai guna praktis segmentasi
bisnis, **jumlah klaster optimal dipilih sebesar $k = 5$**.

# Step 8: Melakukan Klasterisasi

``` {python}
#| label: run-kmeans

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(data_scaled)

# Menyimpan hasil klaster ke dataframe
data['Cluster'] = cluster_labels
data_final['Cluster'] = cluster_labels

# Menampilkan distribusi jumlah anggota per klaster
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"Klaster {cluster_id}: {count} pelanggan ({count/len(data)*100:.2f}%)")
```

# Step 9: Visualisasi klaster dengan PCA 2D

Perhitungan Principal Component Analysis (PCA) untuk visualisasi 2D
serta persentase variansi untuk memahami seberapa baik PCA
mempertahankan informasi dari data asli.

``` {python}
#| label: compute-pca

# Menjalankan PCA untuk reduksi dimensi menjadi 2 komponen
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(data_scaled)

data_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
data_pca['Cluster'] = cluster_labels

# Menampilkan persentase variansi yang dijelaskan oleh PCA
variance_explained = pca.explained_variance_ratio_
print(f"Variansi yang dijelaskan oleh PC1: {variance_explained[0]*100:.2f}%")
print(f"Variansi yang dijelaskan oleh PC2: {variance_explained[1]*100:.2f}%")
print(f"Total variansi yang dipertahankan dalam 2D: {sum(variance_explained)*100:.2f}%")
```

Visualisasi klaster pada ruang PCA 2D dengan pewarnaan berdasarkan label
klaster.

``` {python}
#| label: fig-pca-visualization
#| fig-cap: "Sebaran klaster pelanggan dalam ruang 2D PCA dengan pewarnaan berdasarkan label klaster."

# Plot sebaran klaster pada bidang PCA 2D
plt.figure()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Palet premium

sns.scatterplot(
    x='PC1', y='PC2',
    hue='Cluster',
    palette=colors[:optimal_k],
    data=data_pca,
    legend="full",
    alpha=0.7,
    s=60
)
plt.title('Visualisasi Klaster Pelanggan dalam Ruang 2D PCA', fontsize=14, fontweight='bold', pad=15)
plt.xlabel(f'Principal Component 1 ({variance_explained[0]*100:.2f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({variance_explained[1]*100:.2f}% Variance)', fontsize=12)
plt.legend(title='Klaster', loc='best')
plt.tight_layout()
plt.show()
```

# Step 10: Profiling cluster (rata-rata fitur per cluster)

``` {python}
#| label: tbl-cluster-profiles
#| tbl-cap: "Tabel rata-rata fitur (profiling) untuk setiap klaster pelanggan beserta ukuran klaster."

# Menghitung rata-rata fitur untuk setiap klaster
cluster_profiles = data_final.groupby('Cluster').mean()
cluster_sizes = data_final['Cluster'].value_counts().sort_index()
profile_table = pd.concat([cluster_profiles, cluster_sizes.rename('Cluster Size')], axis=1)

# dump profile table ke file csv
profile_table.to_csv('cluster_profiles.csv')

# Menampilkan tabel profil klaster
# Format tampilan angka agar rapi
pd.set_option('display.float_format', lambda x: '%.2f' % x)
display(profile_table.T.rename(columns={0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'}))
```

# Simpulkan Analisa dari hasil

Berdasarkan hasil analisis profil pada tabel di atas, kita dapat
mengidentifikasi karakteristik khas dari masing-masing klaster pelanggan
serta menyusun strategi pemasaran personalisasi yang tepat sasaran
(*targeted marketing campaigns*).

## Karakteristik Utama dari Setiap Klaster

### Klaster 0: *Frugal Basic-Educated Consumers* (Konsumen Hemat Berpendidikan Dasar)

- **Karakteristik Utama:** Kelompok ini sangat kecil (hanya 54 pelanggan
  atau 2,4% dari total data), beranggotakan pelanggan dengan tingkat
  pendidikan **Basic** (100% dari klaster ini) dan rata-rata pendapatan
  terendah ($~\$20.306$). Pengeluaran total mereka sangat minim
  ($~\$81,80$) dengan prioritas belanja yang merata di angka kecil.
  Kelompok ini cukup aktif mengunjungi web (6,87 kali/bulan) tetapi
  hampir tidak pernah melakukan transaksi online atau menanggapi
  kampanye pemasaran (tingkat respons 3,7%).
- **Demografi:** Usia rata-rata paling muda di antara klaster lainnya
  (46,5 tahun) dan memiliki tanggungan anak (0,72).

### Klaster 1: *Affluent Elite Spenders* (Pembelanja Elite Kelas Atas)

- **Karakteristik Utama:** Kelompok premium (171 pelanggan atau 7,6%)
  dengan tingkat pendapatan tertinggi ($~\$81.569$) dan sangat sedikit
  anak di rumah (0,18). Mereka mencatat pengeluaran belanja tertinggi
  ($~\$1.620,13$), didominasi oleh pembelian produk Anggur (Wines,
  $~\$874,70$) dan Daging (Meat, $~\$469,13$). Mereka jarang mengunjungi
  situs web (3,0 kali/bulan), tidak peduli dengan promo/diskon
  (pembelian *deals* terendah), dan **sangat responsif terhadap
  kampanye** (respons kampanye terakhir mencapai 57,9% serta tingkat
  penerimaan Kampanye 5 mencapai 94,2%).
- **Preferensi Saluran:** Menyukai katalog (6,05) dan pembelian langsung
  di toko fisik (8,26).

### Klaster 2: *Budget-Conscious Families* (Keluarga Hemat & Pemburu Diskon)

- **Karakteristik Utama:** Kelompok terbesar dalam dataset (979
  pelanggan atau 43,7%). Mereka adalah keluarga dengan tingkat
  pendapatan menengah ke bawah ($~\$35.899$) dan jumlah anak terbanyak
  di rumah (rata-rata 1,26 anak). Pengeluaran belanja mereka sangat
  rendah ($~\$98,65$). Mereka sering berselancar di web (6,40
  kunjungan/bulan) untuk mencari penawaran diskon, tercermin dari
  pembelian memanfaatkan promosi yang cukup dominan (2,03). Tingkat
  respons terhadap kampanye sangat rendah (9,1%).
- **Preferensi Saluran:** Pembelian langsung di toko fisik (3,23) dan
  web (2,11).

### Klaster 3: *Mature Web-Active Bargain Hunters* (Pemburu Diskon Online Senior)

- **Karakteristik Utama:** Kelompok ini terdiri dari 579 pelanggan
  (25,8%) dengan rata-rata usia tertua (~60 tahun) dan didominasi oleh
  keluarga yang memiliki anak remaja (0,95 *Teenhome*). Dengan
  pendapatan menengah-atas ($~\$57.046$), mereka memiliki pengeluaran
  belanja menengah-tinggi ($~\$711,93$) yang berpusat pada Anggur
  ($~\$459,78$) dan Emas ($~\$58,65$). Karakteristik mencolok mereka
  adalah **pembelian menggunakan diskon tertinggi (3,89)** dan
  **transaksi melalui web tertinggi (6,32)** di antara semua klaster.
- **Preferensi Saluran:** Berbelanja aktif melalui Web dan Toko.

### Klaster 4: *Independent High-Value Shoppers* (Pembelanja Mandiri Kelas Menengah Atas)

- **Karakteristik Utama:** Kelompok mapan (457 pelanggan atau 20,4%)
  dengan pendapatan tinggi ($~\$73.944$) dan sedikit tanggungan anak
  (0,29). Mereka memiliki pengeluaran belanja yang sangat tinggi
  ($~\$1.239,76$), dengan pengeluaran terbesar untuk produk Buah
  ($~\$70,78$), Ikan ($~\$104,73$), Manisan ($~\$71,42$), dan Emas
  ($~\$77,46$). Mereka mandiri dalam berbelanja, sangat jarang
  mengunjungi web (2,93 kali/bulan), dan **tidak peduli pada kampanye
  promosi** (tingkat respons hanya 16,6% dan penerimaan Kampanye 1-5
  hampir nol).
- **Preferensi Saluran:** Memiliki pembelian di toko fisik tertinggi
  (8,36) dan katalog yang kuat (5,74).

## Implikasi Bisnis dan Strategi Pemasaran Personalisasi

Dengan memahami perbedaan mendasar ini, perusahaan dapat menghentikan
strategi pemasaran “satu ukuran untuk semua” (*one-size-fits-all*) dan
beralih ke strategi personalisasi yang lebih efektif:

1.  **Strategi untuk Klaster 0 (Frugal Basic-Educated Consumers):**
    - **Tindakan:** Batasi anggaran iklan kampanye untuk klaster ini
      demi efisiensi biaya. Pemasaran diarahkan pada produk esensial
      atau paket promosi bernilai sangat murah (*entry-level*).
    - **Saluran:** Gunakan pemasaran SMS atau pesan singkat instan
      karena mereka tidak responsif terhadap kampanye email besar.
2.  **Strategi untuk Klaster 1 (Affluent Elite Spenders):**
    - **Tindakan:** Targetkan dengan program loyalitas VIP eksklusif,
      pra-pemesanan (*pre-order*) untuk produk edisi terbatas, serta
      penawaran paket produk anggur dan daging kualitas premium.
    - **Saluran:** Manfaatkan pengiriman katalog cetak eksklusif ke
      rumah dan pendekatan personal via manajer akun VIP, karena mereka
      sangat responsif terhadap katalog dan toko fisik.
3.  **Strategi untuk Klaster 2 (Budget-Conscious Families):**
    - **Tindakan:** Buat promosi bertema keluarga seperti “Back to
      School” atau “Family Bundles”. Sediakan program loyalitas poin
      yang dapat ditukar dengan diskon belanja kebutuhan sehari-hari.
    - **Saluran:** Gunakan promosi berbasis aplikasi web atau email
      newsletter yang berisi daftar produk diskon, karena mereka aktif
      memburu *deals*.
4.  **Strategi untuk Klaster 3 (Mature Web-Active Bargain Hunters):**
    - **Tindakan:** Fokus pada penawaran diskon secara online (misal:
      *flash sales* atau kupon diskon digital khusus anggur). Buat
      kampanye produk penunjang gaya hidup keluarga paruh baya dengan
      anak remaja.
    - **Saluran:** Optimalkan platform e-commerce dan iklan web
      tertarget, karena mereka adalah pengguna web paling aktif untuk
      bertransaksi.
5.  **Strategi untuk Klaster 4 (Independent High-Value Shoppers):**
    - **Tindakan:** Tingkatkan pengalaman belanja langsung di toko fisik
      (*in-store experience*) dengan penataan produk yang premium
      (terutama buah segar, ikan segar, dan produk manis). Berikan
      diskon spontan di kasir atau layanan bungkus kado gratis untuk
      barang bernilai tinggi.
    - **Saluran:** Optimalkan kenyamanan tata letak toko
      (*merchandising*) dan interaksi staf toko, serta kirimkan katalog
      produk premium bulanan. Jangan membanjiri mereka dengan kampanye
      email digital karena tingkat responsnya sangat rendah.

# Informasi

Dokumen ini ditulis dengan bantuan GitHub Copilot dan Google Antigravity
untuk membantu dalam penulisan kode dan penjelasan. Namun, semua
analisis, interpretasi, dan kesimpulan yang diambil adalah hasil
pemikiran saya sendiri berdasarkan data yang tersedia.

# 1. Pendahuluan

Segmentasi pelanggan adalah salah satu aplikasi terpenting dari analisis
data dalam bisnis modern. Dengan membagi basis pelanggan menjadi
kelompok-kelompok yang memiliki karakteristik serupa (homogen di dalam
kelompok, heterogen antar kelompok), perusahaan dapat merancang strategi
pemasaran yang lebih personal, menargetkan produk yang relevan,
meningkatkan retensi pelanggan, dan mengoptimalkan anggaran promosi.

Laporan ini menyajikan analisis segmentasi pelanggan menggunakan teknik
**Unsupervised Learning** pada dataset *Customer Personality Analysis*
(`cpa.csv`). Dokumen ini disusun menggunakan prinsip **Literate
Programming**, di mana teori dasar dari setiap langkah analisis
disajikan bersama dengan kode implementasi Python dan visualisasinya.

------------------------------------------------------------------------

# 2. Pemrosesan Data & Rekayasa Fitur

Sebelum menerapkan algoritme klasterisasi, data mentah harus diproses
terlebih dahulu agar memenuhi asumsi-asumsi matematis dari algoritme
jarak seperti K-Means.

## Step 1: Pemuatan Dataset

Langkah pertama adalah memuat dataset ke dalam memory menggunakan
pustaka `pandas`. Dataset ini berformat *tab-separated values* (TSV).

``` {python}
#| label: setup-and-load
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Mengatur tema plot agar estetis dan bersih
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# Membaca dataset
df = pd.read_csv("cpa.csv", sep="\t")
print(f"Dataset berhasil dimuat dengan ukuran: {df.shape[0]} baris dan {df.shape[1]} kolom.")
```

## Step 2: Eksplorasi Awal (Exploratory Data Analysis)

### Teori

Eksplorasi awal bertujuan untuk mendeteksi:

- **Missing values**: Sebagian besar algoritme machine learning dalam
  scikit-learn tidak dapat menangani nilai kosong secara langsung.
- **Statistik Deskriptif**: Memahami rentang nilai, rata-rata, dan
  penyebaran data untuk mendeteksi pencilan (outliers).
- **Kolom Konstan**: Kolom yang tidak memiliki variansi (misalnya, semua
  baris bernilai sama) tidak memberikan informasi apa pun dalam analisis
  pengelompokan dan harus dihapus.

``` {python}
#| label: tbl-dataset-head
#| tbl-cap: "Tiga Baris Pertama Dataset CPA"
# Menampilkan 3 baris pertama data
df.head(3)
```

``` {python}
#| label: tbl-missing-values
#| tbl-cap: "Identifikasi Nilai Kosong pada Dataset"
# Memeriksa missing values
missing_info = df.isnull().sum()
print("Kolom dengan nilai kosong:")
print(missing_info[missing_info > 0])
```

Ditemukan terdapat **24 nilai kosong** pada kolom `Income`. Kita harus
melakukan imputasi pada langkah berikutnya.

``` {python}
#| label: check-constant-columns
# Memeriksa kolom konstan
const_cols = [col for col in df.columns if df[col].nunique() <= 1]
print(f"Kolom konstan yang terdeteksi: {const_cols}")
```

Kolom `Z_CostContact` dan `Z_Revenue` bersifat konstan (hanya memiliki 1
nilai unik untuk semua baris), sehingga kolom ini tidak memiliki nilai
informasi untuk proses pengelompokan dan akan dihapus.

------------------------------------------------------------------------

## Step 3: Identifikasi Kolom & Rekayasa Fitur (Feature Engineering)

### Teori

Data waktu (`Dt_Customer`) dan tahun lahir (`Year_Birth`) tidak optimal
jika digunakan secara mentah:

- **Dt_Customer** berisi tanggal pendaftaran pelanggan. Kita dapat
  mengubahnya menjadi **Days_Registered** (jumlah hari pendaftaran
  relatif terhadap pelanggan paling baru di dataset) untuk menangkap
  tingkat loyalitas waktu.
- **Year_Birth** dapat diubah menjadi **Age** (Usia) untuk memudahkan
  interpretasi kelompok demografis. Kita berasumsi tahun analisis adalah
  2015 (mengingat rentang pendaftaran pelanggan berkisar tahun
  2012-2014).
- **Tipe Data**: Memisahkan kolom numerik dan kategorikal untuk
  penanganan pemrosesan yang berbeda (imputasi median untuk numerik,
  one-hot encoding untuk kategori). Kolom indeks seperti `ID` juga harus
  dikeluarkan.

``` {python}
#| label: feature-engineering
df_processed = df.copy()

# 1. Feature Engineering pada tanggal pendaftaran
df_processed['Dt_Customer'] = pd.to_datetime(df_processed['Dt_Customer'], format='%d-%m-%Y')
max_date = df_processed['Dt_Customer'].max()
df_processed['Days_Registered'] = (max_date - df_processed['Dt_Customer']).dt.days
df_processed.drop(columns=['Dt_Customer'], inplace=True)

# 2. Feature Engineering pada tahun lahir (menghitung Usia)
df_processed['Age'] = 2015 - df_processed['Year_Birth']
df_processed.drop(columns=['Year_Birth'], inplace=True)

# 3. Memisahkan kolom numerik dan kategorikal
exclude_cols = ['ID', 'Z_CostContact', 'Z_Revenue']
features_list = [col for col in df_processed.columns if col not in exclude_cols]

numeric_cols = []
categorical_cols = []

for col in features_list:
    if pd.api.types.is_numeric_dtype(df_processed[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print(f"Kolom Numerik ({len(numeric_cols)}): {numeric_cols}")
print(f"Kolom Kategorikal ({len(categorical_cols)}): {categorical_cols}")
```

------------------------------------------------------------------------

## Step 4: Imputasi Missing Value

### Teori

Terdapat nilai kosong pada variabel `Income`. Untuk menangani nilai
kosong pada variabel numerik, kita dapat menggunakan strategi **Median
Imputation**.

- **Mengapa Median?** Median lebih tangguh (*robust*) terhadap pencilan
  (*outliers*) dibandingkan rata-rata (*mean*). Jika dataset memiliki
  beberapa pelanggan dengan pendapatan sangat ekstrem (outliers), nilai
  rata-rata akan bergeser naik secara signifikan, membuat nilai pengisi
  menjadi kurang representatif bagi mayoritas pelanggan. Median tetap
  mewakili nilai tengah yang sesungguhnya.

``` {python}
#| label: missing-imputation
# Melakukan imputasi nilai kosong menggunakan nilai median
num_imputer = SimpleImputer(strategy='median')
df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

print(f"Jumlah nilai kosong pada kolom numerik setelah imputasi: {df_processed[numeric_cols].isnull().sum().sum()}")
```

------------------------------------------------------------------------

## Step 5: Encoding Fitur Kategorikal

### Teori

Algoritme pengklasteran berbasis jarak seperti K-Means menghitung
kesamaan berdasarkan fungsi jarak matematis (misalnya jarak Euclidean).
Variabel teks/kategori seperti `Education` (“Graduation”, “PhD”, dsb.)
dan `Marital_Status` (“Single”, “Married”, dsb.) tidak memiliki
representasi jarak langsung.

Untuk itu, digunakan **One-Hot Encoding**:

- Proses ini mengubah setiap kategori unik pada variabel kategorikal
  menjadi kolom biner baru (bernilai 0 atau 1).
- Misalnya, kolom `Education` akan dipecah menjadi
  `Education_Graduation`, `Education_PhD`, dst.
- Setelah dikodekan, fitur biner digabungkan kembali dengan fitur
  numerik yang sudah diimputasi.

``` {python}
#| label: categorical-encoding
# Menerapkan One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df_processed[categorical_cols])
encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Menggabungkan data numerik & hasil encoding
df_numeric_part = df_processed[numeric_cols].reset_index(drop=True)
df_features = pd.concat([df_numeric_part, encoded_cats_df], axis=1)

print(f"Ukuran dimensi fitur gabungan setelah encoding: {df_features.shape}")
```

------------------------------------------------------------------------

## Step 6: Standardisasi Data (Scaling)

### Teori

Klasterisasi K-Means sangat sensitif terhadap skala data karena
didasarkan pada minimalisasi jarak kuadrat antar titik (Euclidean
Distance):

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$

Jika satu fitur memiliki rentang nilai yang sangat besar (misalnya
`Income` berkisar antara puluhan ribu hingga ratusan ribu) sedangkan
fitur lain memiliki rentang kecil (misalnya `Kidhome` berkisar 0 hingga
2), maka perhitungan jarak akan didominasi sepenuhnya oleh fitur dengan
rentang besar tersebut.

Untuk mengatasinya, diterapkan **Standardisasi (Z-score scaling)**
menggunakan `StandardScaler`:

$$z = \frac{x - \mu}{\sigma}$$

di mana $\mu$ adalah rata-rata dan $\sigma$ adalah standar deviasi.
Hasil akhir transformasinya adalah seluruh fitur akan memiliki rata-rata
($\mu = 0$) dan standar deviasi ($\sigma = 1$).

``` {python}
#| label: data-scaling
# Standardisasi data menggunakan StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)
df_scaled_df = pd.DataFrame(df_scaled, columns=df_features.columns)

print("Statistik deskriptif setelah scaling (rata-rata & std dev):")
print(f"Rata-rata komponen pertama: {df_scaled[:, 0].mean():.4f}")
print(f"Standar deviasi komponen pertama: {df_scaled[:, 0].std():.4f}")
```

------------------------------------------------------------------------

# 3. Penentuan Jumlah Klaster Optimal

## Step 7: Analisis Elbow & Silhouette Score

### Teori

Bagaimana menentukan jumlah kelompok ($k$) yang paling mewakili pola
data tanpa pengawasan? Kami menggunakan kombinasi dua metode:

- **Metode Elbow (Siku)**:
  - Mengukur **Inertia** (atau *Within-Cluster Sum of Squares* / WCSS),
    yaitu jumlah kuadrat jarak antara setiap titik data ke centroid
    klasternya sendiri:
    $$WCSS = \sum_{j=1}^k \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$$
  - Semakin banyak klaster, inersia akan semakin mendekati 0. Kita
    mencari titik “siku” di mana penurunan inersia mulai melambat secara
    signifikan (diminishing returns).
- **Silhouette Score**:
  - Mengukur seberapa baik suatu objek ditempatkan di klasternya
    dibandingkan dengan klaster tetangga terdekatnya.
  - Untuk satu titik data $i$:
    $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$ di mana $a(i)$
    adalah rata-rata jarak titik $i$ ke seluruh titik lain di klaster
    yang sama, dan $b(i)$ adalah rata-rata jarak terpendek dari titik
    $i$ ke klaster lain.
  - Rata-rata skor Silhouette berkisar antara $-1$ hingga $+1$. Nilai
    mendekati $+1$ menunjukkan objek terklaster dengan sangat baik dan
    terpisah jauh dari klaster lain.

``` {python}
#| label: fig-elbow-silhouette
#| fig-cap: "Penentuan Jumlah Klaster Optimal Menggunakan Metode Elbow & Silhouette Score"
# Menghitung inersia dan silhouette score untuk k = 2 s/d 10
ks = range(2, 11)
inertias = []
silhouettes = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(df_scaled, labels))

# Visualisasi kurva Elbow dan Silhouette Score berdampingan
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontsize=12)
ax1.plot(ks, inertias, 'o-', color=color, linewidth=2, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
ax2.plot(ks, silhouettes, 's-', color=color, linewidth=2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Penentuan Jumlah Klaster Optimal: Metode Elbow & Silhouette Score', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.show()
```

### Analisis Pemilihan $k$

- Kurva inersia menunjukkan penurunan tajam dari $k=2$ ke $k=4$, setelah
  itu kurva mulai melandai secara bertahap (membentuk pola siku halus di
  sekitar $k=4$).
- Skor Silhouette menunjukkan nilai yang relatif stabil untuk $k=4$ dan
  $k=5$.
- Berdasarkan analisis bisnis segmentasi pelanggan pada dataset ini,
  memilih **$k = 4$** adalah pilihan yang optimal karena menghasilkan
  pembagian kelompok yang seimbang dan mudah diinterpretasikan secara
  strategis.

------------------------------------------------------------------------

# 4. Klasterisasi K-Means & Reduksi Dimensi

## Step 8: Eksekusi KMeans dengan $k = 4$

### Teori

K-Means mempartisi data menjadi $k$ kelompok dengan cara iteratif:

- Menentukan $k$ titik acak sebagai centroid awal.
- Menugaskan setiap titik data ke centroid terdekat berdasarkan jarak
  Euclidean.
- Menghitung ulang posisi centroid sebagai rata-rata koordinat titik
  yang ditugaskan ke kelompok tersebut.
- Mengulangi langkah-langkah di atas hingga posisi centroid tidak lagi
  berubah secara signifikan.

``` {python}
#| label: kmeans-clustering
# Menjalankan K-Means dengan k=4
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(df_scaled)

# Menyimpan hasil klaster ke dataframe
df['Cluster'] = cluster_labels
df_processed['Cluster'] = cluster_labels

# Menampilkan distribusi jumlah anggota per klaster
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"Klaster {cluster_id}: {count} pelanggan ({count/len(df)*100:.2f}%)")
```

------------------------------------------------------------------------

## Step 9: Reduksi Dimensi & Visualisasi Klaster menggunakan PCA

### Teori

Data setelah preprocessing memiliki **37 fitur** (dimensi tinggi).
Sangat sulit bagi manusia untuk memvisualisasikan data dalam ruang
berdimensi lebih dari 3.

Untuk memvisualisasikan penyebaran kelompok, kita menggunakan
**Principal Component Analysis (PCA)**:

- PCA merotasi sumbu koordinat data ke arah di mana data memiliki
  variansi maksimum.
- Arah variansi terbesar pertama disebut **Principal Component 1
  (PC1)**, dan arah tegak lurus dengan variansi terbesar berikutnya
  adalah **Principal Component 2 (PC2)**.
- Dengan memproyeksikan data 37 dimensi ke ruang 2 dimensi (PC1 dan
  PC2), kita dapat memvisualisasikan struktur klaster pelanggan dengan
  kehilangan informasi sesedikit mungkin.

``` {python}
#| label: pca-dimensionality-reduction
# Menjalankan PCA untuk reduksi dimensi menjadi 2 komponen
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_pca['Cluster'] = cluster_labels

# Menampilkan persentase variansi yang dijelaskan oleh PCA
variance_explained = pca.explained_variance_ratio_
print(f"Variansi yang dijelaskan oleh PC1: {variance_explained[0]*100:.2f}%")
print(f"Variansi yang dijelaskan oleh PC2: {variance_explained[1]*100:.2f}%")
print(f"Total variansi yang dipertahankan dalam 2D: {sum(variance_explained)*100:.2f}%")
```

Dua komponen utama pertama mempertahankan sekitar **23.76%** dari total
variabilitas data asli, yang cukup representatif untuk visualisasi
sebaran pengelompokan 2D.

``` {python}
#| label: fig-pca-clusters-visualization
#| fig-cap: "Visualisasi Klaster Pelanggan dalam Ruang 2D PCA"
# Plot sebaran klaster pada bidang PCA 2D
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Palet premium

sns.scatterplot(
    x='PC1', y='PC2',
    hue='Cluster',
    palette=colors[:optimal_k],
    data=df_pca,
    legend="full",
    alpha=0.7,
    s=60
)
plt.title('Visualisasi Klaster Pelanggan dalam Ruang 2D PCA', fontsize=14, fontweight='bold', pad=15)
plt.xlabel(f'Principal Component 1 ({variance_explained[0]*100:.2f}% Variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({variance_explained[1]*100:.2f}% Variance)', fontsize=12)
plt.legend(title='Klaster', loc='best')
plt.tight_layout()
plt.show()
```

Visualisasi di atas menunjukkan pemisahan kelompok pelanggan yang cukup
jelas di sepanjang sumbu PC1 dan PC2, mengonfirmasi bahwa algoritme
K-Means berhasil mengidentifikasi pola struktural kelompok pelanggan
yang berbeda.

------------------------------------------------------------------------

# 5. Profiling Klaster (Analisis Hasil)

## Step 10: Profiling Karakteristik Rata-rata Fitur per Klaster

### Teori

Untuk memahami siapa saja orang-orang di dalam setiap klaster, kita
menghitung rata-rata (*mean*) dari fitur-fitur penting yang
mendefinisikan perilaku belanja, kondisi demografis, dan interaksi
promosi mereka.

``` {python}
#| label: tbl-cluster-profile
#| tbl-cap: "Tabel Profil Karakteristik Pelanggan untuk Setiap Klaster"
# Mengisi nilai kosong di dataframe eksplorasi awal agar profiling konsisten
df_processed['Income'] = df_processed['Income'].fillna(df_processed['Income'].median())

# Fitur-fitur utama untuk profil
profile_features = [
    'Income', 'Age', 'Kidhome', 'Teenhome', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'Complain', 'Response', 'Days_Registered'
]

# Menghitung rata-rata fitur per klaster
cluster_profile = df_processed.groupby('Cluster')[profile_features].mean()
cluster_sizes = df_processed['Cluster'].value_counts().sort_index().rename('Cluster_Size')
profile_table = pd.concat([cluster_sizes, cluster_profile], axis=1)

# Format tampilan angka agar rapi
pd.set_option('display.float_format', lambda x: '%.2f' % x)
profile_table.T
```

------------------------------------------------------------------------

# 6. Kesimpulan & Rekomendasi Bisnis

Berdasarkan tabel rata-rata profil di atas, kita dapat merumuskan
karakteristik unik dari keempat kelompok pelanggan beserta strategi
pemasaran yang paling efektif bagi masing-masing kelompok:

### Klaster 0: “Keluarga Muda dengan Pendapatan Rendah & Hemat”

- **Karakteristik Demografi**: Usia termuda (rata-rata ~43 tahun),
  memiliki jumlah anak kecil terbanyak (`Kidhome` rata-rata 0.80).
- **Karakteristik Finansial & Belanja**: Pendapatan terendah (~34.7k).
  Pengeluaran sangat minim di semua kategori produk (misalnya rata-rata
  belanja anggur hanya ~38 unit dan daging ~22 unit). Sering mengunjungi
  website perusahaan (`NumWebVisitsMonth` tertinggi, yaitu 6.45
  kali/bulan) namun jarang melakukan pembelian langsung.
- **Strategi Pemasaran**:
  - Fokuskan promosi pada produk kebutuhan anak dan diskon paket hemat
    keluarga.
  - Kirimkan kupon diskon melalui channel digital/web karena mereka
    aktif mengunjungi situs web tetapi sensitif terhadap harga.

### Klaster 1: “Pelanggan VIP yang Sangat Responsif & Konsumtif”

- **Karakteristik Demografi**: Usia menengah (~45.6 tahun), hampir tidak
  memiliki anak di rumah (`Kidhome` ~0.05).
- **Karakteristik Finansial & Belanja**: Pendapatan tertinggi (~81.5k).
  Pengeluaran belanja luar biasa tinggi, terutama untuk Anggur
  (`MntWines` rata-rata ~874) dan Daging (`MntMeatProducts` rata-rata
  ~469). Mereka sangat menyukai pembelian langsung lewat katalog
  (`NumCatalogPurchases` tertinggi, yaitu 6.04) dan toko fisik.
- **Respon Promosi**: Tingkat respon terhadap promosi terakhir
  (`Response`) luar biasa tinggi, mencapai **57.89%**!
- **Strategi Pemasaran**:
  - Ini adalah kelompok pelanggan paling berharga (VIP). Layani mereka
    dengan program loyalitas eksklusif (loyalty program) dan penawaran
    keanggotaan premium.
  - Berikan rekomendasi produk premium (misal: anggur langka berkualitas
    tinggi atau daging potong impor) melalui katalog fisik eksklusif
    atau komunikasi personal langsung.

### Klaster 2: “Pelanggan Kaya yang Tenang & Menyukai Produk Segar”

- **Karakteristik Demografi**: Usia rata-rata ~47 tahun, hampir tidak
  memiliki anak di rumah (`Kidhome` ~0.05).
- **Karakteristik Finansial & Belanja**: Pendapatan tinggi (~73.8k).
  Pengeluaran belanja tinggi di semua kategori produk, khususnya
  memiliki konsumsi buah (`MntFruits` ~70), ikan (`MntFishProducts`
  ~104), dan permen/manisan (`MntSweetProducts` ~71) tertinggi dibanding
  kelompok lain.
- **Respon Pemasaran**: Cukup loyal dengan respon promosi moderat
  (~16.5%). Jarang melakukan komplain.
- **Strategi Pemasaran**:
  - Tawarkan promosi produk segar sehat (seafood, buah organik,
    cokelat/permen premium).
  - Maksimalkan penjualan melalui gerai fisik (`NumStorePurchases`
    tertinggi, yaitu 8.35 kali) karena mereka menyukai pengalaman
    berbelanja langsung di toko tanpa banyak terganggu oleh tawaran
    diskon online.

### Klaster 3: “Pelanggan Setia Kelas Menengah (Keluarga dengan Remaja)”

- **Karakteristik Demografi**: Usia tertua (rata-rata ~51 tahun),
  memiliki anak remaja terbanyak (`Teenhome` tertinggi, yaitu 0.94).
  Masa keanggotaan terlama (`Days_Registered` tertinggi ~407 hari).
- **Karakteristik Finansial & Belanja**: Pendapatan kelas menengah
  (~56.8k). Pengeluaran moderat, dengan minat utama pada Anggur
  (`MntWines` ~446). Sangat aktif berburu diskon (`NumDealsPurchases`
  tertinggi, yaitu 3.84 kali) dan sering membeli melalui website
  (`NumWebPurchases` tertinggi, yaitu 6.20 kali).
- **Strategi Pemasaran**:
  - Tawarkan promosi berbasis potongan harga (*deals*) untuk
    mempertahankan loyalitas mereka.
  - Tawarkan paket bundling produk untuk keluarga menengah melalui
    platform e-commerce (web) karena mereka sangat terbiasa belanja
    online.

------------------------------------------------------------------------

# 7. Informasi Diagnostik & Kredit

Dokumen laporan analisis ini dibuat dan ditulis secara otomatis
menggunakan pendekatan *literate programming* oleh **Google
Antigravity**, asisten pengodean cerdas berbasis AI yang dikembangkan
oleh tim Google DeepMind.

Berikut adalah informasi diagnostik terkait lingkungan eksekusi
pembuatan dokumen ini:

## Diagnostik Antigravity

- **Nama Agen**: Antigravity
- **Pengembang**: Google DeepMind Team
- **Conversation ID**: `72af37b6-ff6b-45af-bca3-d1abfd82d133`
- **Workspace Corpus Name**: `return215/machine-learning-tugas-ut`
- **Sistem Operasi**: Linux (distrobox host)

## Diagnostik Python & Environment

- **Python Executable**: Python 3.14.4
- **Conda Environment**: `machine-learning-tugas` (dijalankan di dalam
  distrobox container `ubuntu-bigdata`)
- **Quarto Executable**: `/usr/local/bin/quarto`
- **Versi Pustaka Analisis Data Utama**:
  - `pandas`: `3.0.2`
  - `scikit-learn`: `1.8.0`
  - `matplotlib`: `3.10.8`
  - `seaborn`: `0.13.2`
