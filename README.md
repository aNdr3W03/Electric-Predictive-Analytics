# Laporan Proyek Machine Learning - Andrew Benedictus Jamesie

## Domain Proyek

Domain proyek ini akan membahas mengenai permasalahan dalam bidang lingkungan dan energi yang dibuat untuk mengetahui prediksi penggunaan daya listrik (*electric power consumption*) berdasarkan data cuaca yang telah dikumpulkan di kota Tétouan, Maroko.

<img src="https://user-images.githubusercontent.com/64983961/188515112-debf11cd-90f1-434a-a6f1-fc448e2304c8.png" alt="Energy Power Illustration" title="Energy Power Illustration" width="100%">

Tétouan adalah sebuah kota yang terletak di bagian utara Maroko dengan luas wilayah sekitar $10.375km^2$ dan jumlah penduduk sekitar 380.787 jiwa, menurut data sensus terakhir tahun 2014, dan mengalami peningkatan sebesar 1,96% setiap tahun. [[1]](https://en.wikipedia.org/wiki/T%C3%A9touan 'Tétouan') Tétouan terletak di sepanjang Laut Mediterania, sehingga memiliki kondisi cuaca yang sejuk dan musim dingin, panas, dan kering selama musim panas.

Konsumsi energi per kapita Maroko sebesar 0,56 toe (*tonne of oil equivalent*), sekitar 42% di bawah rata-rata Afrika Utara, termasuk juga listrik sekitar 900 kWh (38% di bawah rata-rata regional) pada tahun 2020. Perkembangan konsumsi energi total melambat antara 2010 dan 2019 (+3% per tahun, dibandingkan 4,5% per tahun selama 2000-2010), dan turun sebesar 7% pada tahun 2020 menjadi sekitar 21 Mtoe (*Million tonnes of oil equivalent*). [[2]](https://www.enerdata.net/estore/energy-market/morocco 'Morocco Energy Information')

Produksi nasional hidrokarbon cukup rendah dan semua produk minyak diimpor sejak dilakukannya penutupan kilang minyak tunggal negara itu pada tahun 2015 (200.000 bbl/d). Impor produk minyak meningkat pesat dari tahun 2015 hingga tahun 2019 (+6% per tahun) dan turun sebesar 12% di tahun 2020 karena krisis COVID-19. [[2]](https://www.enerdata.net/estore/energy-market/morocco 'Morocco Energy Information') Data konsumsi daya dikumpulkan oleh Supervisory Control and Data Acquisition System (SCADA) Amendis yang merupakan penyelenggaraan layanan publik dan bertanggung jawab atas distribusi air minum dan listrik sejak tahun 2002. Energi yang disalurkan berasal dari National Office of Electricity and Drinking Water (Dinas Listrik dan Air Minum Nasional). Setelah tegangan tinggi (63 kV) diubah menjadi tegangan menengah (20 kV), diperbolehkan untuk mendistribusikan energi.

Dengan konsumsi listrik tersebut, berdasarkan data dan latar belakang di atas, maka di dalam proyek ini akan dibuat sebuah model *machine learning* untuk melakukan analisis prediksi terhadap penggunaan energi atau daya listrik. Dengan adanya model *machine learning* yang telah dibangun, diharapkan dapat membantu dalam memperkirakan besarnya daya listrik yang dikonsumsi berdasarkan atribut data konsumsi energi listrik di kota Tétouan, Maroko.

# Business Understanding

## Problem Statements

Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap persiapan data sebelum digunakan untuk membuat model *machine learning*?
2. Bagaimana cara membuat model *machine learning* untuk melakukan prediksi konsumsi daya listrik?

## Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:
1. Melakukan tahap persiapan data (*data preparation*) sehingga data dapat digunakan pada model *machine learning* dengan baik.
2. Membuat model *machine learning* untuk melakukan analisis prediksi konsumsi daya listrik dengan tingkat *error* yang cukup rendah.

## Solution Statements

Berdasarkan penjelasan di atas, terdapat beberapa solusi yang dapat dilakukan untuk dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data (*data preparation*) dapat dilakukan dengan beberapa teknik, sebagai berikut:
   - Melakukan pembagian data menjadi 2, yaitu data latih (*training data*) dan data uji (*testing data*) dengan perbandingan rasio sebesar 90 : 10 yang akan digunakan ketika membangun model *machine learning*.
   - Melakukan standarisasi nilai pada data fitur numerik untuk mencegah terjadinya penyimpangan nilai data yang cukup besar.
2. Tahap pembuatan model *machine learning* akan digunakan 3 model dengan algoritma *machine learning* yang berbeda. Algoritma yang akan digunakan adalah K-Nearest Neighbor Algorithm, Random Forest Algorithm, dan Adaptive Boosting Algorithm. Dari ketiga model tersebut akan dilakukan evaluasi performa dan kinerja masing-masing algoritma dan akan dipilih satu algoritma yang memberikan hasil prediksi yang terbaik.
   - **Algoritma K-Nearest Neighbor**  
     Sesuai dengan namanya, yaitu "sejumlah k-tetangga terdekat" adalah algoritma *machine learning* yang tergolong ke dalam *supervised learning* yang bekerja dengan cara mengelompokkan data berdasarkan kemiripan antar data baru dengan sejumlah data (k) yang terdekat. [[3]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning') Cara kerja algoritma K-Nearest Neighbor, sebagai berikut: [[3]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Tentukan jumlah tetangga terdekat (`k`) yang akan dipertimbangkan sebagai dasar klasifikasi.
     - Hitung jarak antara data baru terhadap semua titik data dalam *dataset* (tetangga terdekat).
     - Urutkan jarak pada dari kecil ke besar, lalu diambil titik data dengan jarak terkecil dari sejumlah `k` titik.
     - Hitung jumlah titik pada `k` setiap kelas atau kategori.
     - Masukkan data baru ke kelas dengan jumlah `k` terbanyak.
     
     <br>
     <img src="https://user-images.githubusercontent.com/64983961/188507827-0f729ab6-61a5-4dbc-9be2-afa424f6c294.png" alt="Ilustrasi Algoritma K-Nearest Neighbor" title="Ilustrasi Algoritma K-Nearest Neighbor">
     
     Perhitungan jarak ke tetangga terdekat dapat dilakukan dengan menggunakan metrik sebagai berikut:
     - *Euclidean distance*
       $$d(x,y)=\sqrt{\sum_{i=1}^n (x_i-y_i)^2}$$
     - *Manhattan distance*
       $$d(x,y)=\sum_{i=1}^n |x_i-y_i|$$
     - *Hamming distance*
       $$d(x,y)=\frac{1}{n}\sum_{n=1}^{n=n} |x_i-y_i|$$
     - *Minkowski distance*
       $$d(x,y)=\left(\sum_{i=1}^n |x_i-y_i|^p\right)^\frac{1}{p}$$
     
     Adapun kelebihan dari algoritma K-Nearest Neighbor, yaitu: [[3]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Sangat sederhana dan mudah untuk dipahami
     - Sangat mudah dalam penerapannya
     - Dapat digunakan dalam kasus klasifikasi maupun regresi
     - Dapat digunakan dalam jumlah kelas yang berbeda-beda
     - Tidak memerlukan proses trainig dan pembangunan model, karena data baru secara langsung akan dikelaskan
     - Mudah jika ingin untuk melakukan penambahan data
     - Parameter yang dibutuhkan hanya sedikit, yaitu jumlah k-tetangga (`n_neighbors`), dan metode perhitungan metrik jaraknya (`metric`) [[4]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html 'sklearn.neighbors.KNeighborsRegressor')
     - Hasil pemodelan tidak linear, sehingga cocok untuk klasifikasi data yang batasannya tidak linear.
     
     Adapun kelemahan dari algoritma K-Nearest Neighbor, yaitu: [[3]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Perlu untuk menentukan nilai `k` yang tepat
     - *Computation cost* yang cukup tinggi
     - Waktu pemrosesan akan berlangsung lama jika *dataset* yang digunakan sangat besar
     - Kurang bagus untuk diterapkan pada *high dimensional data*
     - Sangat sensitif pada data yang memiliki banyak *noise* (*noisy data*), data yang hilang (*missing data*), dan data dengan nilai yang ekstrem serta kemunculannya yang jarang (*outliers*).
     
   - **Algoritma Random Forest**  
     Metode Random Forest merupakan jenis algoritma *supervised learning* dan termasuk ke dalam metode Decision Tree yang menggunakan kombinasi dari masing-masing model tree dan akan digabungkan menjadi sebuah model dalam membuat hasil prediksi akhir. Algoritma Random Forest menggunakan teknik *bagging* (*bootstrap aggregating*), di mana beberapa model akan dilatih dengan cara *random sampling with replacement*. [[5]](https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest 'Random Forest')
     
     <img src="https://user-images.githubusercontent.com/64983961/188504775-b7e4aa9b-f1cd-41ef-8a70-a977db8f3d60.png" alt="Ilustrasi Algoritma Random Forest" title="Ilustrasi Algoritma Random Forest">
     
     Setelah dilakukan pelatihan, prediksi untuk sampel yang tidak terlihat ($x'$) dapat dibuat dengan menghitung rata-rata prediksi dari semua pohon setiap individu model pada $x'$. [[6]](https://en.wikipedia.org/wiki/Random_forest#Bagging 'Random Forest - Bagging')
     $$\hat{f}=\frac{1}{B}\sum_{b=1}^{B} f_b(x^{'})$$
     
   - **Algoritma Adaptive Boosting**  
     Algoritma Adaptive Boosting atau biasanya disingkat AdaBoost merupakan algoritma yang melakukan pelatihan model secara berurutan dan dengan proses iteratif atau berulang. Data latih (*training data*) akan mempunyai bobot atau *weight* yang sama, kemudian model akan melakukan pemeriksaan. Bobot yang lebih tinggi akan dimasukkan ke dalam model yang salah, sehingga akan lanjut ke tahap selanjutnya. Proses iteratif tersebut akan terus berlanjut hingga model mencapai tingkat akurasi yang diinginkan.
     
     <img src="https://user-images.githubusercontent.com/64983961/188507801-30224052-cac2-4e99-9c67-2aec18de8e59.png" alt="Ilustrasi Algoritma Adaptive Boosting" title="Ilustrasi Algoritma Adaptive Boosting">
     
     Algoritma AdaBoost mengacu kepada metode tertentu untuk melakukan pelatihan *classifier* yang di-*boosted*. Pengklasifikasian tersebut adalah pengklasifikasian dalam bentuk, [[7]](https://en.wikipedia.org/wiki/AdaBoost#Training 'AdaBoost - Training')
     $$F_T(x)=\sum_{t=q}^{T}f_t(x)$$
     di mana setiap $F_T$ adalah *learner* yang lemah yang mengambil objek $x$ sebagai input dan mengembalikan nilai yang menunjukkan kelas objek. Demikian juga pada pengklasifikasi $T$ merupakan nilai positif jika sampel berada dalam kelas positif, dan negatif jika sebaliknya.

## Data Understanding

<img src="https://user-images.githubusercontent.com/64983961/188505289-4725df5e-9e3a-48b9-b261-e538fd0c6fb9.png" alt="Electric Power Consumption Kaggle Dataset" title="Electric Power Consumption Kaggle Dataset" width="100%">

Data yang digunakan dalam proyek ini adalah *dataset* yang diambil dari Kaggle Dataset [Electric Power Consumption](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption 'Time series analysis of power consumption') dengan kategori *dataset*, yaitu *Energy* dan *Electricity*. Dalam *dataset* tersebut terdapat sebuah *file* atau berkas dengan nama `powerconsumption.csv` yang berekstensi (*file format*) `.csv` atau [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values 'Comma-separated values') berukuran 4,33 MB.

Dari *dataset* tersebut, masih perlu dilakukan penyesuaian hingga *dataset* dapat benar-benar digunakan. Beberapa penyesuaian tersebut, yaitu
- Menghapus kolom yang tidak digunakan dalam model, yaitu kolom `GeneralDiffuseFlows`, dan kolom `DiffuseFlows`.
  ```python
   epower.drop('GeneralDiffuseFlows', inplace=True, axis=1)
   epower.drop('DiffuseFlows',        inplace=True, axis=1)
   ```
- Mengubah format atau tipe data pada kolom `Datetime` dari format `string` menjadi `datetime`.
  ```python
  epower.Datetime = pd.to_datetime(epower.Datetime)
  ```
- Melakukan penguraian atau pemisahan kolom `Datetime` menjadi `Year`, `Month`, `Day`, `Hour`, dan `Minute`, lalu menghapus atau membuang (*drop*) kolom `Datetime`.
  ```python
  epower['Year']   = epower['Datetime'].apply(lambda date: date.year)
  epower['Month']  = epower['Datetime'].apply(lambda date: date.month)
  epower['Day']    = epower['Datetime'].apply(lambda date: date.day)
  epower['Hour']   = epower['Datetime'].apply(lambda date: date.hour)
  epower['Minute'] = epower['Datetime'].apply(lambda date: date.minute)
  ```

Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) sebagai investigasi awal untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data dengan menggunakan teknik statistik dan representasi grafis atau visualisasi.

1. **Deskripsi Variabel**  
   Berikut adalah informasi mengenai variabel-variabel yang terdapat pada *dataset* *Electric Power Consumption* adalah sebagai berikut,
   
   <img src="https://user-images.githubusercontent.com/64983961/188505396-dda2d93c-9266-4c80-bb67-6f7ae4e6e8aa.png" alt="Deskripsi Variabel" title="Deskripsi Variabel">
   
   Dari gambar di atas dapat dilihat bahwa terdapat 52.416 baris data dan 10 kolom atribut atau fitur. Di antaranya adalah enam (6) atribut/variabel dengan tipe data `float64 non-null` dan lima (5) atribut/variabel dengan tipe data `int64 non-null` yang merupakan hasil penguraian dari variabel `Datetime` yang sebelumnya memiliki tipe data `datetime64[ns]`. Berikut adalah keterangan untuk masing-masing variabel,
   - `Temperature` : Temperatur
   - `Humidity`    : Kelembaban
   - `WindSpeed`   : Kecepatan angin
   - `PowerConsumption_Zone1` : Konsumsi daya listrik di stasiun Quads, Tétouan, Maroko
   - `PowerConsumption_Zone2` : Konsumsi daya listrik di stasiun Smir, Tétouan, Maroko
   - `PowerConsumption_Zone3` : Konsumsi daya listrik di stasiun Boussafou, Tétouan, Maroko
   - `Year`   : Tahun
   - `Month`  : Bulan
   - `Day`    : Tanggal
   - `Hour`   : Jam
   - `Minute` : Menit
   
2. **Deskripsi Statistik**  
   
   <img src="https://user-images.githubusercontent.com/64983961/188506144-7b2f5f52-be07-47ef-96a5-c65dbba6452a.png" alt="Deskripsi Statistik" title="Deskripsi Statistik">
   
3. **Menangani Missing Value**  
   
   <img src="https://user-images.githubusercontent.com/64983961/188506196-0c2457b4-123c-4e13-8954-5edb04c0ed17.png" alt="Menangani Missing Value" title="Menangani Missing Value">
   
   Berdasarkan gambar tersebut, tidak terdapat *missing value*.
   
4. **Menangani Outliers**  
   *Outliers* merupakan sampel data yang nilainya berada sangat jauh dari cakupan umum data utama yang dapat merusak hasil analisis data. Berikut adalah visualisasi *boxplot* untuk melakukan pengecekan keberadaan *outliers*.
   
   <img src="https://user-images.githubusercontent.com/64983961/188506260-f27e7d3d-e16e-42e7-a31e-8812f2aca7ea.png" alt="Menangani Outliers - Sebelum" title="Menangani Outliers - Sebelum">
     
   Berdasarkan gambar tersebut, terdapat *outliers* pada fitur `Temperature`, `Humidity`, `PowerConsumption_Zone2`, dan `PowerConsumption_Zone3`. Sehingga dilakukan proses pembersihan *outliers* dengan metode IQR (*Inter Quartile Range*).
   
   $$IQR=Q_3-Q_1$$
   
   Kemudian membuat batas bawah dan batas atas untuk mencakup *outliers* dengan menggunakan,
   
   $BatasBawah=Q_1-1.5*IQR$
   
   $BatasAtas=Q_3-1.5*IQR$
   
   
   Setelah dilakukan pembersihan *outliers*, dilakukan kembali visualisasi *outliers* untuk melakukan pengecekan kembali sebagai berikut,
   
   <img src="https://user-images.githubusercontent.com/64983961/188506280-e40fe70d-804c-457e-a6f3-7a89d425950d.png" alt="Menangani Outliers - Sesudah" title="Menangani Outliers - Sesudah">
   
   Dari gambar di atas dapat dilihat bahwa *outliers* telah berkurang. Meskipun *outliers* masih terdapat pada fitur `Temperatur`, `Humidity`, `PowerConsumption_Zone2`, dan `PowerConsumption_Zone3`, tetapi masih dalam batas aman.
   
5. **Univariate Analysis**  
   Melakukan proses analisis data *univariate* pada fitur-fitur numerik. Proses analisis ini menggunakan bantuan visualisasi histogram untuk masing-masing fitur numerik.
   
   <img src="https://user-images.githubusercontent.com/64983961/188506395-dae2772e-f61a-4ce2-b6ad-26acaa99c319.png" alt="Univariate Analysis" title="Univariate Analysis">
   
   Dari data histogram di atas diperoleh informasi, yaitu:
   - Temperatur menunjukkan *zero-skewed* atau histogram simetris/normal.
   - Lebih dari 50% data kecepatan angin mendekati nilai 0, dan sisanya berada pada nilai 5.
   - Konsumsi daya pada stasiun Quads (Zona 1) sebagian besar berada pada rentang daya 21.000 hingga 40.000, dan paling banyak berada pada daya sekitar 22.500.
   - Konsumsi daya pada stasiun Smir (Zona 2) sebagian besar berada pada rentang daya 12.500 hingga 27.500, dan paling banyak berada pada daya sekitar 16.500.
   - Konsumsi daya pada stasiun Boussafou (Zona 3) sebagian besar berada pada rentang daya 9.000 hingga 17.500, dan rentang 24.000 hingga 26.000, serta paling banyak berada pada daya sekitar 14.000.
   - Data diambil pada tahun 2017.
   
6. **Multivariate Analysis**  
   Melakukan visualisasi distribusi data pada fitur-fitur numerik dari *dataframe* `epower`. Visualisasi dilakukan dengan bantuan *library* `seaborn` `pairplot` menggunakan parameter `diag_kind`, yaitu `kde`, untuk melihat perkiraan distribusi probabilitas antar fitur numerik.
   
   <img src="https://user-images.githubusercontent.com/64983961/188507899-65cd3a60-d19c-47d6-8d7d-c7b1a57364ea.png" alt="Multivariate Analysis" title="Multivariate Analysis">
   
7. **Correlation Matrix with Heatmap**  
   Melakukan pengecekan korelasi antar fitur numerik dengan menggunakan visualisasi diagram *heatmap* *correlation matrix*.
   
   <img src="https://user-images.githubusercontent.com/64983961/188507977-c0120633-e8c2-44f6-9bc6-1b59347ebf86.png" alt="Correlation Matrix with Heatmap" title="Correlation Matrix with Heatmap">
   
   Dapat dilihat pada diagram *heatmap* di atas memiliki *range* atau rentang angka dari 1.0 hingga 0.4 dengan keterangan sebagai berikut,
   - Jika semakin mendekati 1, maka korelasi antar fitur numerik semakin kuat bernilai positif.
   - Jika semakin mendekati 0, maka korelasi antar fitur numerik semakin rendah.
   - Jika semakin mendekati -1, maka korelasi antar fitur numerik semakin kuat bernilai negatif.
   
   Jika korelasi bernilai positif, berarti nilai kedua fitur numerik cenderung meningkat bersama-sama.  
   
   Jika korelasi bernilai negatif, berarti nilai salah satu fitur numerik cenderung meningkat ketika nilai fitur numerik yang lain menurun.

8. **Analisis Korelasi Antar Fitur**  
   - Fitur `PowerConsumption_Zone1` memiliki korelasi yang cukup kuat dengan fitur `Temperature`, `Humidity`, dan `Hour`.
   - Fitur `PowerConsumption_Zone2` memiliki korelasi yang cukup kuat dengan fitur `Temperature`, `Humidity`, `Month`, dan `Hour`.
   - Fitur `PowerConsumption_Zone3` memiliki korelasi yang cukup kuat dengan fitur `Temperature`, `Humidity`, `Month`, dan `Hour`.
   
   Sehingga, fitur `WindSpeed`, `Year`, `Day`, dan `Minute` memiliki korelasi yang paling rendah dengan fitur `PowerConsumption_Zone1`, `PowerConsumption_Zone2`, dan `PowerConsumption_Zone3`. Dengan begitu, dapat dilakukan *drop* (menghapus) fitur-fitur tersebut.
   
   <img src="https://user-images.githubusercontent.com/64983961/188507983-6b44443c-d576-4ab3-8dcf-f7b9cf22ad99.png" alt="Analisis Korelasi Antar Fitur" title="Analisis Korelasi Antar Fitur">

## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan berdasarkan penjelasan yang sudah dipaparkan pada bagian [Solution Statements](#solution-statements "Solution Statements"). Tahap ini penting dilakukan untuk mempersiapkan data sehingga dapat digunakan untuk melatih model *machine learning* dengan baik. Berikut adalah dua tahapan data preparation yang dilakukan, yaitu,

1. **Split Data**  
   Pembagian data dilakukan untuk memisahkan data keseluruhan menjadi dua (2) bagian, yaitu data latih (*training data*) dan data uji (*testing data*) dengan perbandingan rasio sebesar 90 : 10 menggunakan `train_test_split`.
   
    ```python
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, random_state=123)
    ```
    
   Kemudian diperoleh hasil pembagian data masing-masing, yaitu sebagai berikut,
   
    ```python
    Total seluruh sampel : 50931
    Total data train     : 45837
    Total data test      : 5094
    ```

2. **Standarisasi pada Fitur Numerik**  
   Standarisasi fitur numerik menggunakan `StandardScaler` untuk mencegah terjadinya penyimpangan nilai data yang cukup besar. Proses standarisasi tersebut dilakukan dengan mengurangkan nilai rata-rata, lalu membaginya dengan standar deviasi atau simpangan baku untuk menggeser distribusi. Proses standarisasi akan menghasilkan distribusi dengan nilai rata-rata menjadi 0, dan nilai standar deviasi menjadi 1.
   
    ```python
    scaler = StandardScaler()
    scaler.fit(xTrain[numericalFeatures])
    xTrain[numericalFeatures]  = scaler.transform(xTrain.loc[:, numericalFeatures])
    ```
   
   <img src="https://user-images.githubusercontent.com/64983961/188508047-08b6a450-aa39-4b2f-8b40-ef86e5adc216.png" alt="Standarisasi pada Fitur Numerik" title="Standarisasi pada Fitur Numerik">

    ```python
    xTrain[numericalFeatures].describe().round(4)
    ```
   
   <img src="https://user-images.githubusercontent.com/64983961/188508061-75a22910-be6c-485a-a2da-e5364d75e311.png" alt="Deskripsi Statistik setelah Standarisasi" title="Deskripsi Statistik setelah Standarisasi">

## Modelling

Setelah dilakukannya tahap *data preparation*, selanjutnya adalah melakukan tahap persiapan model terlebih dahulu sebelum mengembangkan model menggunakan algoritma yang telah ditentukan.

Tahap persiapan *dataframe* untuk analisis model menggunakan parameter `index`, yaitu train_mse dan test_mse, serta parameter `columns` yang merupakan algoritma yang akan digunakan untuk melakukan prediksi, yaitu algoritma K-Nearest Neighbor (KNN), Random Forest, dan Adaptive Boosting (AdaBoost).

```python
models = pd.DataFrame(
    index   = ['train_mse', 'test_mse'],
    columns = ['KNN', 'RandomForest', 'Boosting']
)
```

Kemudian terapkan ketiga algoritma ke dalam model tersebut.

1. **K-Nearest Neighbor (KNN) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_neighbors` dengan nilai k = 10 tetangga dan `metric` bawaan, yaitu Euclidean.
   
   ```python
   knn = KNeighborsRegressor(n_neighbors=10)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
2. **Random Forest Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_estimator` dengan jumlah 50 *trees* (pohon), `max_depth` dengan nilai kedalaman atau panjang pohon 16, `random_state` dengan nilai 55, dan `n_jobs` yang bernilai -1 (pekerjaan dilakukan secara paralel).
   
   ```python
   rf = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
3. **Adaptive Boosting (AdaBoost) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `learning_rate` dengan nilai bobot setiap *regressor* adalah 0.05, dan `random_state` dengan nilai 55.
   
   ```python
   boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)

Ketiga model yang telah dibangun di atas, akan dilakukan pengujian kinerja untuk masing-masing model yang menggunakan algoritma K-Nearest Neighbor, algoritma Random Forest, dan algoritma Adaptive Boosting. Dari ketiga model tersebut akan diperoleh satu (1) model dengan hasil prediksi yang paling baik dan tingkat *error* yang paling rendah.

## Evaluation

Pada tahap evaluasi model, akan dilakukan pengujian untuk melihat algoritma mana yang memberikan hasil prediksi paling baik dan dengan tingkat *error* yang paling rendah. Sebelumnya, akan dilakukan proses standarisasi atau *scaling* pada fitur numerik data uji (*testing data*) agar nilai rata-rata (*mean*) bernilai 0, dan varians bernilai 1.

```python
xTest.loc[:, numericalFeatures] = scaler.transform(xTest[numericalFeatures])
```

Kemudian evaluasi dari ketiga model, yaitu algoritma K-Nearest Neighbor, Random Forest, dan Adaptive Boosting (AdaBoost) untuk masing-masing data latih (*training data*) dan data uji (*testing data*) dengan melihat tingkat *error*-nya menggunakan *Mean Squared Error* (MSE),

$$MSE=\frac{1}{N}\sum_{i=1}^{N} (y_i-y\\_pred_i)^2$$

di mana, nilai $N$ adalah jumlah *dataset*, nilai $y_i$ merupakan nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya.

Penggunaan metode metrik *Mean Squared Error* (MSE) memiliki kelebihan, yaitu cukup sederhana dan mudah dipahami dalam melakukan perhitungan. Meskipun begitu, terdapat kelemahan pada metrik ini, yaitu hasil akurasi prediksi yang kecil karena tidak dapat membandingan hasil peramalan tersebut dengan kenyataannya. []

```python
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'RF', 'Boosting'])
modelDict = {'KNN': knn, 'RF': rf, 'Boosting': boosting}
for name, model in modelDict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=yTrain, y_pred=model.predict(xTrain))/1e3
    mse.loc[name, 'test']  = mean_squared_error(y_true=yTest,  y_pred=model.predict(xTest))/1e3
```

<img src="https://user-images.githubusercontent.com/64983961/188511052-986610cd-7ef4-4f79-a7c1-eef777d3a4f8.png" alt="Evaluation" title="Evaluation">

Dari data tabel tersebut dapat divisualisasikan pada grafik batang berikut.

<img src="https://user-images.githubusercontent.com/64983961/188511209-7f53ee96-f76b-4252-b87c-5e27b0fed0fb.png" alt="Evaluation Graph" title="Evaluation Graph">

Dari visualisasi diagram di atas dapat disimpulkan bahwa,
1. Model dengan algoritma Random Forest memberikan nilai *error* yang paling kecil, yaitu sebesar 583.1 pada *training error*, dan 1542.6 pada *testing error*.
2. Model dengan algoritma K-Nearest Neighbor memiliki tingkat *error* yang sedang di antara dua algoritma lainnya.
3. Model dengan algoritma Adaptive Boosting mengalami *error* yang paling beser dengan nilai *training error* sebesar 7602.37, dan nilai *testing error* sebesar 7436.21.

Selanjutnya adalah pengujian prediksi model dengan menggunakan beberapa nilai konsumsi daya (*power consumption*) dari data uji (*testing data*)

<img src="https://user-images.githubusercontent.com/64983961/188511397-7664a384-d933-4962-9569-f42cdbdbcf69.png" alt="Testing Model" title="Testing Model">

Dapat dilihat prediksi pada model dengan algoritma Random Forest memberikan hasi yang paling mendekati dengan nilai `y_true` jika dibandingkan dengan algoritma model yang lainnya.

Nilai `y_true` sebesar **28507** dan nilai prediksi `Random Forest` sebesar **28308**.

Kesimpulannya adalah model yang digunakan untuk melakukan prediksi penggunaan daya listrik (*electric power consumption*) menghasilkan **tingkat *error* yang paling rendah** dengan menggunakan **algoritma Random Forest** pada model yang telah dibangun.

---

## Referensi

[1] "Tétouan", Retrieved from: https://en.wikipedia.org/wiki/T%C3%A9touan

[2] Enerdata, "Morocco Energy Information", Retrieved from: https://www.enerdata.net/estore/energy-market/morocco

[3] S. Hussein, "Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning", *GEOSPASIALIS*, 2021, Retrieved from: https://geospasialis.com/k-nearest-neighbor

[4] scikit-learn, "sklearn.neighbors.KNeighborsRegressor", Retrieved from: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

[5] A. Yanuar, "Random Forest", *Universitas Gadjah Mada Menara Ilmu Machine Learning*, 2018, Retrieved from: https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest

[6] "Random Forest", Retrieved from: https://en.wikipedia.org/wiki/Random_forest#Bagging

[7] "AdaBoost", Retrieved from: https://en.wikipedia.org/wiki/AdaBoost#Training

[8] S. R. P. Nur Hidayatika, and S. N. W.P, "USULAN PENGGUNAAN METODE FORECASTING UNTUK PERMINTAAN KOPI ROBUSTA PADA PT. XYZ," *Industrial Engineering Online Journal*, vol. 4, no. 3, 2016, Retrieved from: https://ejournal3.undip.ac.id/index.php/ieoj/article/view/9002

[9] A. Salam and A. E. Hibaoui, "Comparison of Machine Learning Algorithms for the Power Consumption Prediction : - Case Study of Tetouan city –," *2018 6th International Renewable and Sustainable Energy Conference (IRSEC)*, 2018, pp. 1-5, doi: 10.1109/IRSEC.2018.8703007.
