# Catatan Praktikum — dibaca 5 menit sebelum mulai

## Alur

```
1. Lihat data      -> tentukan case-nya
2. Buka file       -> cheatsheet-<case>.ipynb
3. Run §2          -> lihat struktur file, salin nama kolom yang dicetaknya
4. Isi §1 CONFIG   -> PATH, kolom, BAHASA, preprocessing, MODEL
5. Run All
6. Ambil §8 (skor) dan §10 (file output)
```

## Menentukan case

Lihat kolom label. Output §2 sudah mencetak jumlah nilai unik per kolom.

| Yang terlihat di kolom label | Case | File sklearn | Kalau diminta PyTorch |
|---|---|---|---|
| 2 nilai unik (`ham` / `spam`) | biner | `cheatsheet-biner.ipynb` | `pytorch-biner.ipynb` |
| > 2 nilai, tiap sel satu nilai | multiclass | `cheatsheet-multiclass.ipynb` | `pytorch-multiclass.ipynb` |
| Ada sel berisi beberapa nilai (`ekonomi\|politik`) | multilabel | `cheatsheet-multilabel.ipynb` | `pytorch-multilabel.ipynb` |
| 2 nilai tapi timpang (rasio > 5:1) | imbalanced | `cheatsheet-imbalanced.ipynb` | `pytorch-imbalanced.ipynb` |

Versi PyTorch memakai CONFIG dengan bentuk sama, ditambah bagian arsitektur
(`ARSITEKTUR` = `meanpool`/`cnn`/`lstm`/`gru`) dan training (`EPOCHS`, `LR`, `BATCH`).
Evaluasinya tetap `classification_report` sklearn, dan tiap file otomatis mencetak
pembanding TF-IDF.

## Yang diisi di CONFIG

```python
# struktur file -- lihat hasil §2 dulu, jangan menebak
PATH   = "data/xxx.csv"
SEP    = None        # None = tebak (.tsv -> tab). Isi ";" kalau CSV gaya Excel ID
HEADER = "infer"     # None kalau baris pertama sudah berupa data
ENC    = None        # None = coba utf-8, latin-1, cp1252

# kolom -- salin dari baris "saran untuk CONFIG" yang dicetak §2
TEXT_COL   = None    # nama kolom "message", atau indeks 5 kalau HEADER=None
LABEL_COL  = None
ID_COL     = None
LABEL_MAP  = None    # kalau label angka: {0: "negatif", 4: "positif"}

BAHASA = "en"        # "id" kalau datanya bahasa Indonesia

# preprocessing -- semuanya opsional, pipeline tetap jalan kalau semua False
LOWERCASE   = True
MASK        = True
STOPWORD    = True
JAGA_NEGASI = True   # JANGAN dimatikan untuk task sentimen
STEMMING    = False  # bahasa id butuh: pip install Sastrawi

# TF-IDF (dikunci, tidak perlu diganti)
NGRAM  = (1, 2)
MIN_DF = 2           # turunkan ke 1 kalau datanya kecil (< 200 dokumen)

# model
MODEL       = "logreg"   # nb | logreg | svm | tree | rf
SEIMBANGKAN = False      # True untuk case imbalanced & multilabel
```

## Tiga situasi yang butuh file lain

| Situasi | Pakai |
|---|---|
| Dosen memberi **dua file** `train.csv` + `test.csv` | `latihan_sklearn.ipynb` dengan `"sumber": "train_test"` |
| Diminta **tabel perbandingan** beberapa setelan/model | `latihan_sklearn.ipynb` §9 (jalankan banyak varian sekaligus) |
| Diminta **neural network** | `pytorch-<case>.ipynb` — atau `MODEL = "mlp"` di `latihan_sklearn.ipynb` kalau cukup NN sederhana |

## Kalau error

| Gejala | Penyebab | Solusi |
|---|---|---|
| Baris pertama data hilang / nama kolom aneh | `HEADER` salah | `"infer"` ↔ `None` |
| Semua isi jadi satu kolom | `SEP` salah | isi `"\t"` atau `";"` |
| `UnicodeDecodeError` | encoding | `ENC = "latin-1"` |
| `empty vocabulary` | semua kata terbuang | `MIN_DF = 1`, matikan `STOPWORD` |
| Kelas jadi banyak padahal cuma 2 | label tidak konsisten kapital/spasi | sudah ditangani `.strip().lower()` di §3 |
| Akurasi 0.99 mencurigakan | duplikat / leakage | sudah ada `drop_duplicates`; cek lagi apakah test dipakai untuk menyetel |
| Multilabel prediksi kosong semua | model tanpa `class_weight` | `SEIMBANGKAN = True` |
| Stemming ID tidak jalan | Sastrawi belum ada | `pip install Sastrawi` (lakukan malam ini) |

## Yang dilaporkan

- `classification_report` **lengkap**, bukan cuma accuracy
- confusion matrix
- beberapa contoh salah klasifikasi + alasannya
- argumentasi tiap langkah preprocessing yang dipilih (ini permintaan Homework #2)

Untuk case **imbalanced**: laporkan f1/recall kelas minoritas, bukan accuracy. §8 mencetak
angka pembanding "model yang selalu menebak kelas mayoritas" — bandingkan dengan itu.

Untuk case **multilabel**: laporkan f1 micro **dan** macro, plus subset accuracy.

## Jawaban singkat kalau ditanya lisan

**Kenapa TF-IDF, bukan TF biasa?**
TF menghitung frekuensi mentah, jadi kata umum seperti "the" mendominasi. IDF menekan kata yang
muncul di banyak dokumen (`idf = log(N/df)`), sehingga bobot terbesar jatuh ke kata yang khas
untuk dokumen itu.

**Kenapa IDF sklearn beda dengan rumus di slide?**
sklearn memakai `ln((1+N)/(1+df)) + 1` (`smooth_idf`) supaya term yang muncul di semua dokumen
tidak jadi nol, lalu tiap baris di-L2-normalize supaya dokumen panjang tidak otomatis menang.
Kalau mau persis slide: `TfidfVectorizer(smooth_idf=False, norm=None)`.

**Kenapa negasi tidak dibuang padahal stopword?**
"the food is **not** recommended" → kalau "not" dibuang jadi "food recommended", maknanya
terbalik. Untuk sentimen, negasi justru penanda kelas.

**Kenapa pakai bigram?**
Bag of words membuang urutan kata. Bigram mengembalikan sebagian: "not good" jadi satu fitur
sendiri, berbeda dari "good".

**Beda multiclass dan multilabel?**
Multiclass: >2 kelas, tiap dokumen tepat satu label, kelas saling meniadakan. Multilabel: satu
dokumen boleh punya beberapa label sekaligus; targetnya matriks 0/1 dan classifier-nya dibungkus
`OneVsRestClassifier` (satu model biner per label).

**Kenapa `stratify` saat split?**
Menjaga proporsi kelas sama di train dan test. Tanpa itu, pada data timpang kelas minoritas bisa
nyaris tidak masuk test set dan skornya jadi tidak bermakna.

**Kenapa `Pipeline`?**
Supaya vectorizer hanya di-`fit` pada data latih. Kalau di-`fit` pada seluruh data, vocabulary
dan nilai IDF ikut "melihat" data uji → data leakage → skor terlalu bagus dan salah.

**Kenapa Naive Bayes disebut naive?**
Karena mengasumsikan tiap kata independen bila kelasnya diketahui. Asumsi itu jelas salah untuk
bahasa, tapi praktiknya tetap bagus dan sangat cepat.

**Kenapa accuracy tidak dipakai pada data timpang?**
Pada data 90:10, model yang selalu menebak kelas mayoritas dapat akurasi 0,90 dengan recall kelas
minoritas nol. Pakai f1 macro atau f1 kelas minoritas.

## Perintah

```bash
jupyter lab
```

Jalankan dari folder `lat-prak1`. Cek cepat tanpa buka UI:

```bash
python -m nbconvert --to notebook --execute --inplace cheatsheet-biner.ipynb
```

## Status

12 notebook, 172 sel kode, terakhir dijalankan ulang seluruhnya dengan kernel bersih:
**0 error, 0 warning**. Waktu jalan 9–34 detik per notebook.

Satu-satunya paket yang belum terpasang: **Sastrawi** (stemming bahasa Indonesia).
`pip install Sastrawi` — lakukan selagi ada internet. Tanpa itu, opsi `STEMMING` untuk bahasa
Indonesia dilewati otomatis dan notebook tetap jalan.
