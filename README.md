# Bekal Praktikum — Klasifikasi Teks

**IF5153 Advanced NLP** · slide `02a-Klasifikasi-Teks` & `02b-Word-Representation`

Isi folder ini: **4 cheatsheet siap pakai per jenis task**, **2 cheatsheet referensi teori**,
**2 template lengkap**, dan tiga catatan `.md`.

```
lat-prak1/
├── CATATAN.md                       <- BACA INI DULU besok (5 menit)
├── README.md                        <- file ini: peta seluruh folder
├── TAHAPAN.md                       <- penjelasan 7 tahap + skrip tiap tahap
│
│   === PILIH SESUAI JENIS TASK: ubah CONFIG -> Run All ===
├── cheatsheet-biner.ipynb           2 kelas (spam/ham, positif/negatif)
├── cheatsheet-multiclass.ipynb      >2 kelas, satu label per dokumen
├── cheatsheet-multilabel.ipynb      satu dokumen bisa banyak label ("a|b")
├── cheatsheet-imbalanced.ipynb      kelas timpang (rasio >5:1)
│
│   === VERSI PYTORCH dari empat case di atas ===
├── pytorch-biner.ipynb              meanpool / cnn / lstm, eval tetap sklearn
├── pytorch-multiclass.ipynb
├── pytorch-multilabel.ipynb         BCEWithLogitsLoss + pos_weight
├── pytorch-imbalanced.ipynb         bobot kelas pada loss
│
│   === TEMPLATE LENGKAP: semua opsi, semua bentuk input ===
├── latihan_sklearn.ipynb            termasuk kasus train.csv + test.csv terpisah
├── latihan_pytorch.ipynb            CNN / LSTM / GRU / word2vec
│
│   === REFERENSI TEORI ===
├── cheatsheet_klasifikasi_teks.ipynb   slide 02a: preprocessing, BoW/TF-IDF, classical ML
├── cheatsheet_pytorch.ipynb            slide 02b: word representation, CNN, LSTM
│
└── data/
    ├── sms_spam.csv                 80 SMS EN, header, koma, biner
    ├── ulasan_produk_id.csv         40 ulasan ID, header, koma, biner
    ├── berita_topik.tsv             88 judul, TAB, 4 kelas (multi-class)
    ├── berita_multilabel.csv        88 judul, label ganda dipisah "|"
    ├── keluhan_imbalanced.csv       1.777 baris, rasio 9:1
    ├── sentiment140_sample.csv      3.200 tweet, TANPA header, latin-1, label 0/4
    ├── reviews_messy.csv            18 baris kotor: NaN, duplikat, label tak konsisten
    └── split/                       train.csv + test.csv + test_unlabeled.csv
```

---

## Yang mana yang saya buka besok?

```
Datanya seperti apa?
│
├── 2 kelas seimbang         ->  cheatsheet-biner.ipynb
├── >2 kelas, 1 label        ->  cheatsheet-multiclass.ipynb
├── 1 dokumen banyak label   ->  cheatsheet-multilabel.ipynb
├── kelas timpang >5:1       ->  cheatsheet-imbalanced.ipynb
│
├── train & test dua file    ->  latihan_sklearn.ipynb ("sumber": "train_test")
├── perlu tabel perbandingan ->  latihan_sklearn.ipynb §9
├── diminta neural network   ->  pytorch-<case>.ipynb (atau latihan_pytorch.ipynb)
└── ditanya teori            ->  cheatsheet_klasifikasi_teks.ipynb / cheatsheet_pytorch.ipynb
```

**Kemungkinan besar: salah satu `cheatsheet-<case>.ipynb`.** Slide berhenti di TF-IDF + classical ML, dan
Homework #2 meminta Naive Bayes atau Decision Tree. Versi PyTorch adalah asuransi.

---

## Alur pakai template (kedua-duanya sama)

1. Buka notebook **dari folder `lat-prak1`** (path data ditulis relatif, `data/...`).
2. Baca **§2 Menu** sebentar — daftar semua nilai yang sah.
3. Ubah **§1 SEL CONFIG** sesuai soal praktikum.
4. **Run All.**
5. Ambil angkanya dari §8 (sklearn) / §7 (pytorch), tabel perbandingan dari §9 / §8.

Yang perlu diubah cuma sel `CFG`. Sel-sel lain adalah mesin yang membaca `CFG` dan menyesuaikan
diri sendiri — tidak perlu disentuh.

### Contoh: dosen memberi dua file `train.csv` dan `test.csv`

```python
CFG = {
    "sumber":     "train_test",
    "path_train": "data/split/train.csv",
    "path_test":  "data/split/test.csv",
    "case":       "biner",
    "model":      "logreg",
    ...
}
```

### Contoh: data bahasa Indonesia, 4 kelas topik

```python
CFG = {
    "sumber":            "satu_file",
    "path":              "data/berita_topik.tsv",
    "case":              "multiclass",
    "bahasa":            "id",
    "normalisasi_slang": True,
    "stemming":          True,
    ...
}
```

---

## Isi tiap file

### `latihan_sklearn.ipynb` — template utama

Satu sel config mengatur enam kelompok keputusan:

| Kelompok | Contoh isi |
|---|---|
| **Input** | `sumber` (satu file / dua file / folder txt / dataframe), path, kolom, encoding, header, `label_map` |
| **Case** | `biner`, `multiclass`, `multilabel`, `imbalanced` · `bahasa` en/id |
| **Preprocessing** | 9 sakelar True/False: lowercase, masking, stopword, jaga negasi, slang, stemming, lemmatization, koreksi ejaan, filter POS |
| **Feature extraction** | `boolean` / `tf` / `tfidf` / `hashing`, n-gram, min_df, max_df, LSA (`svd`), seleksi (`chi2` / `mutual_info`) |
| **Model** | `nb`, `bernoulli_nb`, `logreg`, `svm`, `tree`, `rf`, `mlp`, `xgboost` |
| **Evaluasi** | test_size, cross-validation, tuning GridSearchCV |

Fitur lain: validasi config yang **memperbaiki kombinasi mustahil sendiri** (mis. Naive Bayes +
SVD → SVD dimatikan, karena NB tidak menerima nilai negatif), tabel perbandingan banyak config
sekaligus, analisis fitur penting, daftar dokumen salah klasifikasi, dan simpan/muat model.

### `latihan_pytorch.ipynb` — template neural

Bentuk config sengaja dibuat mirip. Bedanya di kelompok model:

| Kelompok | Contoh isi |
|---|---|
| **Arsitektur** | `meanpool`, `cnn` (Kim 2014), `lstm`, `gru` |
| **Embedding** | `acak` atau `word2vec` (dilatih gensim di data latih), `freeze_emb` |
| **Ukuran** | `dim`, `hidden`, `n_filter`, `kernel`, `bidirectional`, `dropout` |
| **Training** | `epochs`, `batch`, `lr`, `weight_decay`, `seimbangkan`, `val_size` |

Otomatis pakai GPU kalau ada, menyimpan bobot epoch terbaik menurut validasi, dan **selalu
menghitung pembanding TF-IDF** di §9 supaya angkanya jujur.

### `cheatsheet_klasifikasi_teks.ipynb` — referensi klasik

Teori lengkap dari slide 02a + kode yang bisa dijalankan: pipeline, preprocessing (termasuk
sentence splitter, maximum matching, morphological analyzer, filter POS, koreksi ejaan),
BoW/TF/TF-IDF, seleksi fitur, classifier, evaluasi, tuning, interpretasi, tabel error umum,
dan lampiran LSA/SVD. Setiap bagian ditandai **WAJIB** atau **OPSIONAL**.

### `cheatsheet_pytorch.ipynb` — referensi neural

Teori jalur neural: sklearn vs PyTorch, teks→tensor, anatomi training loop, CNN, LSTM,
word2vec, kode referensi fine-tune IndoBERT, dan tabel error khas PyTorch.

---

## Wajib vs opsional

Hanya empat hal yang benar-benar wajib:

```
baca data  ->  ubah teks jadi angka  ->  latih model  ->  ukur
```

| Langkah | Status | Kalau dilewati |
|---|---|---|
| Baca & bersihkan data | **WAJIB** | tanpa dibersihkan, skornya palsu |
| Pisah train/test | **WAJIB** | tidak bisa mengukur generalisasi |
| Preprocessing | opsional | tetap jalan — `TfidfVectorizer` sudah lowercase + tokenisasi sendiri |
| Vectorization | **WAJIB** | sklearn menolak string mentah |
| Seleksi fitur / LSA | opsional | hanya lebih lambat |
| Latih + evaluasi | **WAJIB** | — |
| Tuning, interpretasi | opsional | tambahan nilai untuk laporan |

Urutan kerja yang disarankan: **jalankan pipeline minimum sampai keluar angka dulu**, baru
tambahkan langkah opsional satu per satu sambil mencatat perubahan skornya. Kalau semua
ditumpuk di awal dan skornya jelek, kamu tidak akan tahu penyebabnya yang mana.

---

## Tiga jebakan yang paling sering menjatuhkan nilai

1. **Data leakage.** Vectorizer atau vocabulary di-`fit` pada seluruh data (termasuk test), atau
   ada dokumen duplikat yang masuk ke train *dan* test. Skornya jadi terlalu bagus. Selalu pakai
   `Pipeline` dan `drop_duplicates()` sebelum split — template sudah melakukan keduanya.
2. **Menyetel parameter sambil melihat skor test.** Itu overfit ke test set. Pakai validation
   set atau cross-validation dari train; test disentuh sekali di akhir.
3. **Membuang kata negasi.** Stopword list standar membuang "not" / "tidak", padahal itu
   membalik makna kalimat sentimen. Sakelar `jaga_negasi` ada untuk ini.

---

## Menjalankan

Semua notebook dijalankan **dari folder `lat-prak1`**:

```bash
jupyter lab
```

Atau buka di VS Code lalu **Run All**. Untuk memastikan semuanya jalan tanpa membuka UI:

```bash
python -m nbconvert --to notebook --execute --inplace latihan_sklearn.ipynb
```

Perintah itu berhenti dengan pesan error kalau ada satu sel pun yang gagal. Keempat notebook
sudah lolos perintah tersebut.

### Kebutuhan

| Paket | Status | Dipakai oleh |
|---|---|---|
| scikit-learn, pandas, numpy | terpasang | semua |
| nltk | terpasang | preprocessing bahasa Inggris (ada fallback kalau tak ada) |
| torch | terpasang | dua notebook PyTorch |
| gensim | terpasang | word2vec |
| xgboost | terpasang | `model="xgboost"` |
| **Sastrawi** | **belum terpasang** | stemming bahasa Indonesia |

Satu-satunya yang perlu dipasang, dan sebaiknya malam ini selagi ada internet:

```bash
pip install Sastrawi
```

Tanpa Sastrawi, opsi `stemming` untuk bahasa Indonesia dilewati otomatis — notebook tetap jalan.

### Catatan versi

Laptop ini memakai **pandas 3.0.1**, yang menyimpan kolom teks sebagai dtype `str`
(pandas 2.x memakai `object`). Semua notebook di sini sudah dibuat tahan dua-duanya, tapi kalau
kamu menyalin kode ke Colab, ingat perbedaan itu — deteksi kolom berbasis `dtype == object`
akan gagal di pandas 3.

---

## Peta ke slide

| Materi slide | Ada di |
|---|---|
| 02a hal. 4–6 · pipeline, rule vs statistical | cheatsheet klasik §1 |
| 02a hal. 7–13 · seluruh langkah preprocessing | cheatsheet klasik §3, §3d · template `CFG` bagian 3 |
| 02a hal. 15–16 · BoW, VSM, Boolean/TF/TF-IDF | cheatsheet klasik §4 · template `CFG["fitur"]` |
| 02a hal. 17 · reduksi fitur, TF-IDF, Mutual Information | cheatsheet klasik §5 · template `CFG["seleksi"]` |
| 02a hal. 18–19 · SVM, NN, XGBoost, NB, Decision Tree | cheatsheet klasik §6 · template `CFG["model"]` |
| 02b hal. 4–5 · one-hot, distributed representation | cheatsheet pytorch §3, §8 |
| 02b hal. 7–9 · SVD / LSI (ada contoh numeriknya) | cheatsheet klasik §11 · template `CFG["svd"]` |
| 02b hal. 10 · CNN for text (Kim 2014) | cheatsheet pytorch §6 · template `arsitektur="cnn"` |
| 02b hal. 11 · RNN / LSTM | cheatsheet pytorch §7 · template `arsitektur="lstm"` |
| 02b hal. 12 · word2vec CBOW & skip-gram | cheatsheet pytorch §8 · template `embedding="word2vec"` |
