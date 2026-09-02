# Panduan Tahap-per-Tahap — Klasifikasi Teks

Dokumen ini membedah pipeline klasifikasi teks jadi **7 tahap**, masing-masing dengan penjelasan,
skrip yang bisa langsung dijalankan, contoh keluarannya, dan jebakan yang biasa muncul.

Semua skrip di sini berurutan — kalau ditempel dari atas ke bawah jadi satu file, ia jalan utuh.
Dijalankan dari folder `lat-prak1`.

```
┌─────────┐   ┌──────────┐   ┌───────────────┐   ┌─────────┐   ┌───────────┐   ┌────────┐
│ 1 INPUT │ → │ 2 SPLIT  │ → │ 3 PREPROCESS  │ → │ 4 FITUR │ → │ 5 MODEL   │ → │ 6 EVAL │
└─────────┘   └──────────┘   └───────────────┘   └─────────┘   └───────────┘   └────────┘
  baca file    train/test      teks → token       token →        pilih &         angka +
  + bersihkan  (+validation)   (opsional)         angka          latih           analisis
                                                                                     ↓
                                                                              ┌──────────┐
                                                                              │ 7 OUTPUT │
                                                                              └──────────┘
                                                                               simpan model,
                                                                               prediksi baru
```

**Yang wajib:** 1, 2, 4, 5, 6. **Yang opsional:** 3 (preprocessing) dan sebagian 7.
Pipeline tanpa tahap 3 tetap jalan, karena `TfidfVectorizer` sudah melakukan lowercase dan
tokenisasi sendiri.

---

## Persiapan

```python
import re
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
SEED = 42
```

---

# TAHAP 1 · INPUT — membaca dan membersihkan data

## Apa yang dikerjakan

Mengubah apa pun yang diberikan dosen (CSV, TSV, folder `.txt`, JSON) menjadi satu DataFrame
dengan **tepat dua kolom**: `text` dan `label`. Semua tahap berikutnya mengasumsikan bentuk ini.

## Kenapa penting

Kesalahan di tahap ini tidak menimbulkan error — ia menimbulkan **skor palsu**. Dokumen duplikat
yang tersebar ke train dan test membuat akurasi melonjak tanpa alasan. Label `"Positif"`,
`"positif "`, dan `"POSITIF"` dihitung sebagai tiga kelas berbeda.

## Skrip

```python
def baca(path, text_col=None, label_col=None, sep=None, header="infer",
         encoding=None, label_map=None):
    path = Path(path)
    sep = sep or ("\t" if path.suffix.lower() in {".tsv", ".tab"} else ",")

    # encoding: coba berurutan, jangan asal utf-8
    df = None
    for enc in ([encoding] if encoding else ["utf-8", "latin-1", "cp1252"]):
        try:
            df = pd.read_csv(path, sep=sep, header=header, encoding=enc,
                             engine="python", on_bad_lines="skip")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    # deteksi kolom kalau tidak disebutkan
    if text_col is None:
        kand = [c for c in df.columns if df[c].map(lambda v: isinstance(v, str)).mean() > 0.5]
        text_col = max(kand, key=lambda c: df[c].astype(str).str.len().mean())
    if label_col is None:
        kand = [(df[c].nunique(), c) for c in df.columns
                if c != text_col and 2 <= df[c].nunique() <= 20]
        label_col = min(kand)[1]

    df = df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]

    # pembersihan
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].apply(lambda v: v.strip().lower() if isinstance(v, str) else v)
    if label_map:
        df["label"] = df["label"].map(label_map).fillna(df["label"])
    df = df[df["text"].str.len() >= 3]
    df = df[~df["text"].str.lower().isin({"nan", "none", "na", "-"})]
    df = df.dropna().drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


df = baca(DATA / "sentiment140_sample.csv", text_col=5, label_col=0,
          header=None, label_map={0: "negatif", 4: "positif"})

print("jumlah baris :", len(df))
print("distribusi   :", df["label"].value_counts().to_dict())
print("panjang teks : rata-rata", int(df["text"].str.split().str.len().mean()), "kata")
print("duplikat     :", int(df["text"].duplicated().sum()))
df.head(3)
```

## Keluaran

```
jumlah baris : 3200
distribusi   : {'negatif': 1600, 'positif': 1600}
panjang teks : rata-rata 13 kata
duplikat     : 0
```

## Varian bentuk input

Fungsi `baca` yang sama menangani ketiga bentuk ini:

```python
# dua file terpisah dari dosen
train = baca(DATA / "split" / "train.csv", label_map={0: "negatif", 4: "positif"})
test  = baca(DATA / "split" / "test.csv",  label_map={0: "negatif", 4: "positif"})
print("train:", train.shape, "| test:", test.shape)

# TSV (tab) multi-class -- separator ditebak dari ekstensi
berita = baca(DATA / "berita_topik.tsv")
print("berita:", berita.shape, berita["label"].unique())

# multi-label: satu dokumen banyak label, dipisah "|"
multi = baca(DATA / "berita_multilabel.csv", text_col="text", label_col="labels")
print("multilabel contoh:", multi["label"].iloc[0])
```

Untuk bentuk yang tidak berupa tabel (contoh referensi, path-nya disesuaikan sendiri):

```python
# satu folder per kelas berisi .txt  ->  data_folder/positif/*.txt, data_folder/negatif/*.txt
from sklearn.datasets import load_files
b = load_files("data_folder", encoding="utf-8", decode_error="replace")
df = pd.DataFrame({"text": b.data, "label": [b.target_names[i] for i in b.target]})

# Excel / JSON Lines
df = pd.read_excel("data.xlsx")
df = pd.read_json("data.jsonl", lines=True)
```

## Jebakan

| Gejala | Penyebab | Solusi |
|---|---|---|
| `UnicodeDecodeError` | file bukan UTF-8 (dataset Kaggle sering latin-1) | coba `encoding="latin-1"` |
| Baris pertama data hilang | file tanpa header tapi dibaca dengan header | `header=None` |
| Akurasi mencurigakan 99% | duplikat tersebar ke train & test | `drop_duplicates()` **sebelum** split |
| Kelas jadi 7 padahal cuma 2 | label tidak konsisten kapital/spasi | `.str.strip().str.lower()` |
| Semua kolom terbaca satu | separator salah | `sep="\t"` untuk TSV |

## Kendali di template

`CFG["sumber"]`, `path`, `text_col`, `label_col`, `sep`, `header`, `encoding`, `label_map`, `sample`

---

# TAHAP 2 · SPLIT — memisahkan data latih dan uji

## Apa yang dikerjakan

Memotong data jadi bagian yang dipakai **melatih** dan bagian yang dipakai **mengukur**.

## Kenapa penting

Model dinilai dari kemampuannya pada data yang **belum pernah dilihat**. Mengukur pada data latih
sama dengan ujian dengan soal yang sudah dibocorkan — nilainya tinggi tapi tidak berarti apa-apa.

`stratify=y` menjaga proporsi kelas tetap sama di kedua bagian. Tanpa itu, pada data timpang bisa
saja kelas minoritas nyaris tidak masuk test set.

## Skrip

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,          # 20% untuk uji
    stratify=df["label"],   # proporsi kelas terjaga
    random_state=SEED,      # hasil bisa diulang
)
print("train:", len(X_train), "| test:", len(X_test))
print("proporsi train:", y_train.value_counts(normalize=True).round(3).to_dict())
print("proporsi test :", y_test.value_counts(normalize=True).round(3).to_dict())
```

## Keluaran

```
train: 2560 | test: 640
proporsi train: {'negatif': 0.5, 'positif': 0.5}
proporsi test : {'negatif': 0.5, 'positif': 0.5}
```

## Kalau dosen sudah memisahkan (dua file)

**Jangan** panggil `train_test_split` lagi. Cukup periksa kesehatan split-nya:

```python
train = baca(DATA / "split" / "train.csv", label_map={0: "negatif", 4: "positif"})
test  = baca(DATA / "split" / "test.csv",  label_map={0: "negatif", 4: "positif"})

bocor = set(train["text"]) & set(test["text"])
print("dokumen bocor train<->test:", len(bocor))
if bocor:
    test = test[~test["text"].isin(bocor)]          # buang dari TEST, bukan train

print("label asing di test:", set(test["label"]) - set(train["label"]))
```

Untuk bereksperimen (memilih preprocessing, model, parameter), potong **validation set dari
train** — jangan pernah menyentuh test sebelum laporan akhir:

```python
tr2, val = train_test_split(train, test_size=0.2, stratify=train["label"], random_state=SEED)
```

## Jebakan

- Menyetel parameter sambil melihat skor test → overfit ke test set. Pakai validation/CV.
- Lupa `stratify` pada data timpang → kelas minoritas bisa nyaris hilang dari test.
- Lupa `random_state` → hasil berubah tiap dijalankan, tidak bisa dibandingkan.

## Kendali di template

`CFG["sumber"]` (`satu_file` vs `train_test`), `test_size`, `random_state`

---

# TAHAP 3 · PREPROCESSING — teks jadi token bersih  ·  OPSIONAL

## Apa yang dikerjakan

Menormalkan teks: huruf kecil, buang/ganti bagian yang tidak informatif, satukan varian kata.

## Kenapa opsional

`TfidfVectorizer` sudah melakukan lowercase dan tokenisasi sendiri. Pipeline **tanpa satu pun**
langkah di tahap ini tetap jalan dan tetap memberi skor wajar. Preprocessing itu penyetelan,
bukan syarat. Yang biasanya paling berdampak: masking URL/angka, stopword yang menjaga negasi,
dan stemming untuk bahasa Indonesia.

## Daftar langkah (slide 02a hal. 7–13)

| Langkah | Contoh | Wajib? |
|---|---|---|
| Lowercase | `FREE` → `free` | opsional (otomatis di vectorizer) |
| Tokenization | `"not enak"` → `["not","enak"]` | otomatis di vectorizer |
| Entity masking | `bit.ly/xx` → `urltoken` | opsional, sering membantu |
| Stopword elimination | buang `the`, `yang` | opsional |
| Stemming | `writing` → `writ` (Porter) | opsional |
| Lemmatization | `writing` → `write` | opsional |
| Morphological analyzer | `writing` → `writ` + `ing` | opsional, jarang untuk klasifikasi |
| Normalisasi slang | `gk` → `tidak` | opsional, penting untuk UGC Indonesia |
| Koreksi ejaan | `tommorow` → `tomorrow` | opsional, berisiko merusak |
| Sentence splitter | pisah kalimat | opsional, untuk dokumen panjang |
| Filter POS | sisakan Noun/Verb/Adj | opsional |

## Skrip

```python
STOP_EN = {"i","me","my","we","you","your","it","they","this","that","is","are","was","were",
           "be","the","a","an","and","but","or","of","at","by","for","with","to","from","in",
           "on","so","than","too","very","just","now","have","has","had","do","does","did"}
NEGASI  = {"no","not","never","nor","cannot"}
STOP_AMAN = STOP_EN - NEGASI          # negasi DIPERTAHANKAN


def mask_entities(t):
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", " emailtoken ", t)
    t = re.sub(r"http\S+|www\.\S+|\b\S+\.(?:com|org|net|ly|id|co)\S*", " urltoken ", t)
    t = re.sub(r"@\w+", " usertoken ", t)
    t = re.sub(r"\b\d{7,}\b", " phonetoken ", t)
    t = re.sub(r"[$£€]\s?\d[\d,.]*", " moneytoken ", t)
    t = re.sub(r"\b\d+\b", " numtoken ", t)
    return t


def preprocess(teks, buang_stopword=True, stemming=False):
    t = str(teks).lower()
    t = mask_entities(t)
    tokens = re.findall(r"[a-z]+", t)
    if buang_stopword:
        tokens = [w for w in tokens if w not in STOP_AMAN]
    if stemming:
        from nltk.stem import PorterStemmer
        tokens = [PorterStemmer().stem(w) for w in tokens]
    return " ".join(tokens) if tokens else "kosongtoken"


contoh = "URGENT!! Call 09066364349 now, claim your FREE £10,000 prize at bit.ly/win"
print("mentah :", contoh)
print("bersih :", preprocess(contoh))

X_train_bersih = X_train.map(preprocess)
X_test_bersih  = X_test.map(preprocess)
print("\ncontoh train:", X_train_bersih.iloc[0][:70])
```

## Keluaran

```
mentah : URGENT!! Call 09066364349 now, claim your FREE £10,000 prize at bit.ly/win
bersih : urgent call phonetoken claim free moneytoken prize urltoken
```

Perhatikan `bit.ly/win` ikut tertangkap walau tanpa `http://` — itu gunanya pola domain telanjang
`\b\S+\.(?:com|org|net|ly|id|co)\S*` di regex. Kalau hanya memakai `http\S+`, URL pendek
seperti ini lolos dan jadi tiga fitur sampah: `bit`, `ly`, `win`.

## Jebakan

1. **Membuang negasi.** Stopword list standar membuang `not` / `tidak`, padahal itu membalik
   makna. `"the food is not recommended"` → `"food recommended"`. Untuk sentimen, kecualikan
   kata negasi (seperti `STOP_AMAN` di atas) atau pakai bigram.
2. **Token mask jadi pecah.** Kalau memakai `nltk.word_tokenize`, placeholder `<URL>` dipecah
   jadi `["<", "URL", ">"]`. Pakai placeholder satu kata seperti `urltoken`.
3. **Preprocessing dua kali.** Kalau sudah membersihkan manual, set `lowercase=False` di
   vectorizer agar tidak membingungkan saat debug.
4. **Kosakata koreksi ejaan dari seluruh data** → leakage. Bangun hanya dari train.

## Kendali di template

`CFG["lowercase"]`, `mask_entity`, `buang_stopword`, `jaga_negasi`, `normalisasi_slang`,
`stemming`, `lemmatization`, `koreksi_ejaan`, `filter_pos`, `bahasa`

---

# TAHAP 4 · FEATURE EXTRACTION — teks jadi angka  ·  WAJIB

## Apa yang dikerjakan

Mengubah daftar token jadi **matriks term × document**. Satu baris = satu dokumen, satu kolom =
satu term. Model sklearn tidak menerima string mentah — tanpa tahap ini langsung `ValueError`.

## Tiga skema bobot (slide 02a hal. 16)

| Skema | Nilai sel | Cocok untuk |
|---|---|---|
| **Boolean** | 1 kalau term ada, 0 kalau tidak | teks pendek (SMS, judul) |
| **TF** | frekuensi term di dokumen | dokumen panjang |
| **TF-IDF** | tf × idf | default paling aman |

Rumus IDF versi slide: `idf = log(N / df)`. Term yang muncul di banyak dokumen → idf kecil →
bobot kecil. Term langka tapi ada di dokumen ini → bobot besar → paling membedakan.

## Skrip

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- pilih salah satu ---
# vect = CountVectorizer(binary=True)                    # Boolean
# vect = CountVectorizer()                               # TF
vect = TfidfVectorizer(                                  # TF-IDF
    ngram_range=(1, 2),      # unigram + bigram
    min_df=2,                # buang term yang muncul di < 2 dokumen
    max_df=0.9,              # buang term yang muncul di > 90% dokumen
    sublinear_tf=True,       # pakai 1+log(tf)
    lowercase=False,         # sudah dilakukan di tahap 3
)

Xtr = vect.fit_transform(X_train_bersih)   # fit HANYA di train
Xte = vect.transform(X_test_bersih)        # test cukup transform

print("matriks train:", Xtr.shape, "| test:", Xte.shape)
print("contoh fitur :", list(vect.get_feature_names_out()[:8]))
print("kepadatan    : %.4f%% sel terisi" % (100 * Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1])))
```

## Keluaran

```
matriks train: (2560, 4831) | test: (640, 4831)
contoh fitur : ['able', 'about', 'account', 'actually', 'add', 'afternoon', 'again', 'ago']
kepadatan    : 0.1489% sel terisi
```

Perhatikan `test` punya jumlah kolom **sama persis** dengan train. Itu tandanya vectorizer
dipakai benar: vocabulary ditetapkan dari train saja.

## IDF sklearn ≠ IDF slide

```python
N = Xtr.shape[0]
istilah = "good"
if istilah in vect.vocabulary_:
    dfreq = int((Xtr[:, vect.vocabulary_[istilah]] > 0).sum())
    print(f"df('{istilah}') = {dfreq}")
    print("  idf slide   log(N/df)             =", round(np.log(N / dfreq), 3))
    print("  idf sklearn ln((1+N)/(1+df)) + 1  =", round(np.log((1 + N) / (1 + dfreq)) + 1, 3))
    print("  tfidf.idf_ sklearn                =", round(vect.idf_[vect.vocabulary_[istilah]], 3))
```

Keluaran:

```
df('good') = 141
  idf slide   log(N/df)             = 2.899
  idf sklearn ln((1+N)/(1+df)) + 1  = 3.892
  tfidf.idf_ sklearn                = 3.892
```

Baris kedua dan ketiga cocok — itu bukti rumus sklearn sudah dipahami benar. Baris pertama
(versi slide) sengaja beda.

Bedanya dua: `smooth_idf` (menambah 1 di pembilang, penyebut, dan di luar log, supaya term yang
muncul di semua dokumen tidak hilang total) dan `norm="l2"` (tiap baris dinormalisasi jadi
panjang 1, supaya dokumen panjang tidak otomatis menang). Kalau ingin persis rumus slide:
`TfidfVectorizer(smooth_idf=False, norm=None)`.

## Reduksi & seleksi fitur (opsional, slide 02a hal. 17)

```python
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import TruncatedSVD

# ambil 1000 term paling diskriminatif
pilih = SelectKBest(chi2, k=1000).fit(Xtr, y_train)
print("setelah chi2:", pilih.transform(Xtr).shape)

# LSA / SVD (slide 02b hal. 7-9) -- hasilnya bisa NEGATIF, jangan disambung ke Naive Bayes
svd = TruncatedSVD(n_components=200, random_state=SEED).fit(Xtr)
print("setelah LSA :", svd.transform(Xtr).shape,
      "| variance dijelaskan:", round(svd.explained_variance_ratio_.sum(), 3))
```

## Jebakan

| Gejala | Penyebab | Solusi |
|---|---|---|
| `X has n features, but expects m` | vectorizer di-`fit` ulang di test | test cukup `transform` |
| `empty vocabulary` | semua kata terbuang stopword/`min_df` | turunkan `min_df` |
| Skor terlalu bagus | vectorizer di-`fit` di seluruh data | pakai `Pipeline` |
| `Negative values in MultinomialNB` | fitur dari SVD | ganti model, atau lepas SVD |

## Kendali di template

`CFG["fitur"]`, `ngram`, `min_df`, `max_df`, `max_features`, `sublinear_tf`, `svd`, `seleksi`, `k_fitur`

---

# TAHAP 5 · MODEL SELECTION — pilih dan latih  ·  WAJIB

## Apa yang dikerjakan

Memilih algoritma, melatihnya pada data latih, dan membandingkan beberapa kandidat secara adil.

## Kandidat (slide 02a hal. 18)

| Model | Sifat | Catatan |
|---|---|---|
| `MultinomialNB` | probabilistik, `P(c|d) ∝ P(c)·Π P(w|c)` | baseline wajib, sangat cepat |
| `BernoulliNB` | versi fitur biner | pasangan alami Boolean |
| `LogisticRegression` | linear diskriminatif | default solid, koefisien bisa dibaca |
| `LinearSVC` | margin terbesar | sering terbaik untuk teks |
| `DecisionTree` | aturan bercabang | bisa digambar, gampang overfit |
| `RandomForest` | ansambel pohon | lebih kuat, lebih lambat |
| `MLPClassifier` | neural network | inilah "NN" di slide, tanpa PyTorch |
| `XGBClassifier` | gradient boosting | disebut di slide |

**Naive Bayes** disebut "naive" karena mengasumsikan tiap kata independen bila kelasnya diketahui.
Jelas tidak benar, tapi praktiknya bagus. `alpha` adalah Laplace smoothing: mencegah kata yang tak
pernah muncul di suatu kelas membuat seluruh perkalian jadi nol.

## Skrip — bandingkan beberapa model sekaligus

```python
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

kandidat = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
baris = []
for nama, clf in kandidat.items():
    pipe = Pipeline([("vect", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                              sublinear_tf=True, lowercase=False)),
                     ("clf", clf)])
    skor = cross_val_score(pipe, X_train_bersih, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    baris.append((nama, round(skor.mean(), 3), round(skor.std(), 3)))

print(pd.DataFrame(baris, columns=["model", "cv f1_macro", "std"]).to_string(index=False))
```

## Keluaran

```
             model  cv f1_macro   std
     MultinomialNB        0.680 0.017
LogisticRegression        0.693 0.016
         LinearSVC        0.667 0.013
      DecisionTree        0.624 0.008
```

## Kenapa pakai `Pipeline`

Vectorizer di-`fit` **hanya** pada bagian latih tiap fold, lalu dipakai `transform` pada fold uji.
Kalau vectorizer di-`fit` di luar `cross_val_score`, nilai IDF sudah "melihat" data validasi →
data leakage → skor terlalu bagus.

## Latih model terpilih

```python
model = Pipeline([("vect", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                           sublinear_tf=True, lowercase=False)),
                  ("clf", LogisticRegression(max_iter=1000))])
model.fit(X_train_bersih, y_train)
print("model terlatih:", model.named_steps["clf"].__class__.__name__)
```

## Tuning (opsional)

```python
from sklearn.model_selection import GridSearchCV

grid = {"vect__ngram_range": [(1, 1), (1, 2)],
        "vect__min_df": [1, 2],
        "clf__C": [0.5, 1.0, 5.0]}
gs = GridSearchCV(model, grid, cv=3, scoring="f1_macro", n_jobs=-1)
gs.fit(X_train_bersih, y_train)
print("param terbaik:", gs.best_params_, "| cv f1:", round(gs.best_score_, 3))
model = gs.best_estimator_
```

## Kalau kelasnya timpang

```python
# LogisticRegression, LinearSVC, DecisionTree, RandomForest mendukung ini:
LogisticRegression(max_iter=1000, class_weight="balanced")
# MultinomialNB, MLP, XGBoost TIDAK punya class_weight
```

## Kendali di template

`CFG["model"]`, `seimbangkan`, `tuning`, `cv`

---

# TAHAP 6 · EVALUATION — mengukur dan menganalisis  ·  WAJIB

## Apa yang dikerjakan

Menghitung metrik pada data uji, lalu **menjelaskan** kesalahannya. Bagian kedua yang biasanya
dinilai lebih tinggi di laporan.

## Metrik

| Metrik | Rumus | Baca sebagai |
|---|---|---|
| Accuracy | (TP+TN)/total | proporsi benar — **menyesatkan bila kelas timpang** |
| Precision | TP/(TP+FP) | dari yang diprediksi spam, berapa yang benar spam |
| Recall | TP/(TP+FN) | dari semua spam sebenarnya, berapa yang tertangkap |
| F1 | 2·P·R/(P+R) | rata-rata harmonik keduanya |

- **macro avg** = rata-rata antar kelas, tiap kelas berbobot sama → pakai kalau timpang.
- **weighted avg** = dibobot jumlah sampel tiap kelas.
- Spam filtering: precision kelas spam lebih penting (email penting masuk spam lebih merugikan).

## Skrip

```python
from sklearn.metrics import classification_report, confusion_matrix

pred = model.predict(X_test_bersih)

print("akurasi :", round(accuracy_score(y_test, pred), 3))
print("f1 macro:", round(f1_score(y_test, pred, average="macro"), 3), "\n")
print(classification_report(y_test, pred, zero_division=0))

kelas = sorted(y_train.unique())
cm = confusion_matrix(y_test, pred, labels=kelas)
print("Confusion matrix (baris = aktual, kolom = prediksi):")
print(pd.DataFrame(cm, index=["true_" + k for k in kelas],
                   columns=["pred_" + k for k in kelas]).to_string())
```

## Keluaran

```
akurasi : 0.711
f1 macro: 0.711

              precision    recall  f1-score   support
     negatif       0.70      0.73      0.72       320
     positif       0.72      0.69      0.70       320
    accuracy                           0.71       640

Confusion matrix (baris = aktual, kolom = prediksi):
              pred_negatif  pred_positif
true_negatif           234            86
true_positif            99           221
```

## Analisis kesalahan — jangan dilewati

```python
salah = pd.DataFrame({"text": X_test.values, "aktual": y_test.values, "prediksi": pred})
salah = salah[salah.aktual != salah.prediksi]
print(f"salah klasifikasi: {len(salah)} dari {len(y_test)}\n")
for _, r in salah.head(5).iterrows():
    print(f"  [aktual={r.aktual} prediksi={r.prediksi}] {r.text[:75]}")
```

Contoh temuan yang layak ditulis di laporan: sarkasme, kalimat tanpa kata bermuatan sentimen,
negasi yang tidak tertangkap unigram, atau teks terlalu pendek.

## Jebakan

- Melaporkan accuracy saja pada data timpang. Model yang selalu menebak kelas mayoritas bisa
  dapat 90% accuracy dengan recall kelas minoritas 0.
- Mengukur di data latih. Angkanya selalu bagus dan tidak berarti.
- Membandingkan model dengan split berbeda. Kunci `random_state`.

## Kendali di template

Otomatis, mengikuti `CFG["case"]` (biner/multiclass memakai macro, multilabel memakai micro).

---

# TAHAP 7 · OUTPUT — interpretasi, simpan, prediksi baru

## Apa yang dikerjakan

Mengubah model terlatih jadi sesuatu yang bisa dipakai dan dijelaskan.

## 7a · Kata apa yang menentukan keputusan model

Ini versi otomatis dari "spam word list" manual di slide 02a hal. 4–5.

```python
nama = np.array(model.named_steps["vect"].get_feature_names_out())
clf = model.named_steps["clf"]

if hasattr(clf, "coef_"):
    skor = clf.coef_[0]
    print(f"[{clf.classes_[0]}] <-- ", ", ".join(nama[np.argsort(skor)[:10]]))
    print(f"[{clf.classes_[1]}] --> ", ", ".join(nama[np.argsort(skor)[-10:][::-1]]))
elif hasattr(clf, "feature_log_prob_"):
    for i, c in enumerate(clf.classes_):
        khas = clf.feature_log_prob_[i] - clf.feature_log_prob_.mean(axis=0)
        print(f"[{c}] ", ", ".join(nama[np.argsort(khas)[-10:][::-1]]))
```

Keluaran:

```
[negatif] <--  not, sorry, work, sad, miss, no, wish, hate, sucks, sick
[positif] -->  usertoken, thanks, thank, happy, good, love, great, re, awesome, wait
```

## 7b · Simpan model

```python
import joblib

joblib.dump(model, "model_latihan.joblib")
model_dimuat = joblib.load("model_latihan.joblib")
print("model berhasil dimuat ulang:", type(model_dimuat).__name__)
```

Simpan **seluruh pipeline**, bukan classifier saja — vectorizer-nya ikut, jadi model bisa langsung
menerima teks mentah.

## 7c · Prediksi teks baru

Teks baru **harus melewati preprocessing yang sama persis** dengan data latih.

```python
teks_baru = ["absolutely love this, best day ever thanks",
             "worst service ever, i hate it, so disappointed"]

bersih = [preprocess(t) for t in teks_baru]          # preprocessing SAMA seperti tahap 3
hasil = model_dimuat.predict(bersih)

for t, h in zip(teks_baru, hasil):
    print(f"  {h:8s} <- {t}")

# kalau butuh probabilitas (LinearSVC tidak punya predict_proba)
if hasattr(model_dimuat.named_steps["clf"], "predict_proba"):
    prob = model_dimuat.predict_proba(bersih)
    print("\nkelas:", list(model_dimuat.classes_))
    print("probabilitas:", prob.round(3))
```

Keluaran:

```
  positif  <- absolutely love this, best day ever thanks
  negatif  <- worst service ever, i hate it, so disappointed
```

## 7d · File submission (kalau test tanpa label)

```python
test_nolabel = pd.read_csv(DATA / "split" / "test_unlabeled.csv")
pred_baru = model.predict([preprocess(t) for t in test_nolabel["text"].astype(str)])
pd.DataFrame({"id": test_nolabel["id"], "label": pred_baru}).to_csv("submission.csv", index=False)
print("submission.csv tersimpan:", len(pred_baru), "baris")
```

## Kendali di template

`latihan_sklearn.ipynb` §10 (analisis) dan §11 (simpan + prediksi)

---

# Skrip lengkap — 7 tahap dalam satu halaman

Versi minimal yang sudah sah dilaporkan:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# 1. INPUT
df = pd.read_csv("data/sentiment140_sample.csv", header=None, encoding="latin-1")
df = df[[5, 0]].rename(columns={5: "text", 0: "label"})
df["label"] = df["label"].map({0: "negatif", 4: "positif"})
df = df.dropna().drop_duplicates(subset=["text"])

# 2. SPLIT
X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=0.2,
                                          stratify=df["label"], random_state=42)

# 3. PREPROCESSING  (opsional -- dilewati di versi minimal ini)

# 4+5. FEATURE EXTRACTION + MODEL, disatukan dalam Pipeline
model = Pipeline([("vect", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                  ("clf",  LogisticRegression(max_iter=1000))])
model.fit(X_tr, y_tr)

# 6. EVALUATION
pred = model.predict(X_te)
print(classification_report(y_te, pred, zero_division=0))
print(confusion_matrix(y_te, pred))

# 7. OUTPUT
print(model.predict(["absolutely love this, best day ever thanks",
                     "worst service ever, i hate it, so disappointed"]))
```

---

# Ringkasan: tahap → kendali → referensi

| Tahap | Wajib | `CFG` yang mengatur | Bagian di notebook | Slide |
|---|---|---|---|---|
| 1 Input | ✔ | `sumber`, `path`, `text_col`, `label_col`, `header`, `encoding`, `label_map` | cheatsheet §2, §2b | — |
| 2 Split | ✔ | `test_size`, `random_state`, `sumber="train_test"` | cheatsheet §2b | — |
| 3 Preprocessing | opsional | 9 sakelar `lowercase`…`filter_pos`, `bahasa` | cheatsheet §3, §3d | 02a hal. 7–13 |
| 4 Feature extraction | ✔ | `fitur`, `ngram`, `min_df`, `max_df`, `svd`, `seleksi` | cheatsheet §4, §5 | 02a hal. 15–17 |
| 5 Model selection | ✔ | `model`, `seimbangkan`, `tuning`, `cv` | cheatsheet §6 | 02a hal. 18–19 |
| 6 Evaluation | ✔ | otomatis mengikuti `case` | cheatsheet §7 | — |
| 7 Output | sebagian | — | cheatsheet §9, template §10–11 | 02a hal. 4–5 |

Untuk jalur neural (PyTorch), tahap 1, 2, 3, 6, 7 sama persis. Yang berbeda hanya tahap 4
(teks → **indeks + padding**, bukan TF-IDF) dan tahap 5 (arsitektur + training loop manual).
Detailnya di `cheatsheet_pytorch.ipynb` dan `latihan_pytorch.ipynb`.
