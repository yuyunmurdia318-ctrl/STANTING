# app.py — Multi-page Streamlit: IMT/U (WHO 2007, 5-19y) + BB/U (WHO 2006, 0-60m)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Status Gizi Anak — Multi Halaman", layout="wide")

# -------------------------
# WHO URLs (IMT/U 5-19y)
# -------------------------
WHO_IMTU_BOYS_XLSX = "https://cdn.who.int/media/docs/default-source/child-growth/growth-reference-5-19-years/bmi-for-age-%285-19-years%29/bmi-boys-z-who-2007-exp.xlsx?sfvrsn=a84bca93_2"
WHO_IMTU_GIRLS_XLSX = "https://cdn.who.int/media/docs/default-source/child-growth/growth-reference-5-19-years/bmi-for-age-%285-19-years%29/bmi-girls-z-who-2007-exp.xlsx?sfvrsn=79222875_2"

# Example CSV links for WFA (0-60m) — may require adjustment or upload fallback
WHO_WFA_BOYS_CSV = "https://www.who.int/childgrowth/standards/wfa_boys_0_60.csv"
WHO_WFA_GIRLS_CSV = "https://www.who.int/childgrowth/standards/wfa_girls_0_60.csv"

# -------------------------
# Helpers (shared)
# -------------------------
@st.cache_data(show_spinner=False)
def download_and_read_excel(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    x = BytesIO(r.content)
    xls = pd.read_excel(x, sheet_name=None, engine="openpyxl")
    # try to pick sheet with L/M/S & age
    for name, sheet in xls.items():
        cols = [c.lower().replace(" ", "") for c in sheet.columns.astype(str)]
        if any(k in cols for k in ("l", "m", "s")) and any(k in cols for k in ("age", "month", "agemonths")):
            return sheet.copy()
    return next(iter(xls.values())).copy()

@st.cache_data(show_spinner=False)
def download_csv(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))

def normalize_lms(df_lms):
    cols = list(df_lms.columns)
    mapping = {}
    for c in cols:
        cl = c.strip().lower().replace(" ", "")
        if cl in ("age","month","agemonths","ageinmonths","age(months)","months"):
            mapping[c] = "Umur"
        if cl == "l": mapping[c] = "L"
        if cl == "m": mapping[c] = "M"
        if cl == "s": mapping[c] = "S"
        if cl in ("sex","gender"): mapping[c] = "Jenis_Kelamin"
    df = df_lms.rename(columns=mapping)
    if "Umur" in df.columns:
        df["Umur"] = pd.to_numeric(df["Umur"], errors="coerce").round().astype("Int64")
    if "Jenis_Kelamin" in df.columns:
        df["Jenis_Kelamin"] = df["Jenis_Kelamin"].astype(str).str.strip().str.upper().replace({"1":"L","2":"P","M":"L","F":"P","B":"L","G":"P"})
    return df

def map_gender(v):
    if pd.isna(v): return "U"
    s = str(v).strip().upper()
    if s in ("1","L","M","BOY","MALE","B"): return "L"
    if s in ("2","P","F","GIRL","FEMALE","G"): return "P"
    return "U"

def zscore_who(BMI, L, M, S):
    try:
        L = float(L); M = float(M); S = float(S); BMI = float(BMI)
        if abs(L) < 1e-8:
            return np.log(BMI / M) / S
        return ((BMI / M) ** L - 1) / (L * S)
    except:
        return np.nan

def get_lms_row(age_month, sex, table_boys, table_girls):
    if sex == "L":
        table = table_boys
    elif sex == "P":
        table = table_girls
    else:
        table = table_boys
    if "Umur" not in table.columns:
        possible = [c for c in table.columns if "age" in c.lower()]
        if possible:
            table2 = table.rename(columns={possible[0]: "Umur"})
        else:
            return None
    else:
        table2 = table
    exact = table2[table2["Umur"] == age_month]
    if not exact.empty:
        return exact.iloc[0]
    t = table2.dropna(subset=["Umur"])
    if t.empty:
        return None
    idx = (t["Umur"] - age_month).abs().idxmin()
    return t.loc[idx]

def compute_z_for_row(r, table_boys, table_girls, value_col="IMT"):
    lms_row = get_lms_row(int(r["Umur"]), r["Jenis_Kelamin"], table_boys, table_girls)
    if lms_row is None:
        return np.nan
    L = M = S = None
    for col in lms_row.index:
        cl = str(col).lower().replace(" ", "")
        if cl == "l": L = lms_row[col]
        if cl == "m": M = lms_row[col]
        if cl == "s": S = lms_row[col]
    return zscore_who(r[value_col], L, M, S)

# =========================
# Sidebar menu (two pages)
# =========================
menu = st.sidebar.selectbox("Pilih Halaman", ["IMT/U (WHO 2007, 5–19 tahun)", "BB/U (WHO 2006, 0–60 bulan)"])
st.title("📊 Status Gizi Anak")

# ======================================================
# 1) HALAMAN IMT/U — gunakan kode IMT/U kamu persis
# ======================================================
if menu == "IMT/U (WHO 2007, 5–19 tahun)":

    st.header("IMT/U (5–19 tahun) — Naive Bayes (L/P)")
    st.write("Perhitungan IMT/U memakai LMS WHO 2007 (boys & girls).")

    # Load LMS WHO 2007
    st.sidebar.header("Data LMS WHO 2007 (IMT/U)")
    st.sidebar.write("Mengambil LMS WHO 2007 (boys & girls) — butuh koneksi & openpyxl.")

    try:
        lms_boys_raw = download_and_read_excel(WHO_IMTU_BOYS_XLSX)
        lms_girls_raw = download_and_read_excel(WHO_IMTU_GIRLS_XLSX)
        lms_boys = normalize_lms(lms_boys_raw)
        lms_girls = normalize_lms(lms_girls_raw)
        st.sidebar.success("LMS WHO 2007 dimuat.")
    except Exception as e:
        st.sidebar.error(f"Gagal muat LMS WHO 2007: {e}")
        st.stop()

    # -------------------------
    # Upload children dataset
    # -------------------------
    st.header("1) Upload dataset anak (CSV / Excel)")
    uploaded = st.file_uploader("Pilih file CSV atau Excel (kolom: NAMA ANAK, Berat, Tinggi, Umur (bulan), Jenis Kelamin)", type=["csv","xlsx","xls"], key="imtu_upload")
    if not uploaded:
        st.info("Upload dataset anak untuk melanjutkan (IMT/U).")
        st.stop()

    # robust read
    try:
        if str(uploaded.name).lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded, engine="openpyxl")
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file data: {e}")
        st.stop()

    st.subheader("Preview (acak 30% dari data)")
    preview_count = max(1, int(len(df) * 0.3))
    st.dataframe(df.sample(n=preview_count, random_state=42))

    # Auto-detect & let user confirm columns
    cols = df.columns.tolist()
    def find_col(cols, keywords):
        for k in keywords:
            for c in cols:
                if k in str(c).lower().replace(" ", ""):
                    return c
        return None

    col_name = find_col(cols, ["nama","name"])
    col_weight = find_col(cols, ["berat","bb","weight"])
    col_height = find_col(cols, ["tinggi","tb","height"])
    col_age = find_col(cols, ["umur","usia","age","month"])
    col_sex = find_col(cols, ["jenis","kelamin","jk","sex","gender"])

    st.subheader("2) Konfirmasi kolom (auto-detect)")
    c_name = st.selectbox("Kolom Nama Anak (opsional)", ["-"] + cols, index=(0 if col_name is None else cols.index(col_name)+1))
    c_weight = st.selectbox("Kolom Berat (kg)", cols, index=cols.index(col_weight) if col_weight in cols else 0)
    c_height = st.selectbox("Kolom Tinggi (cm)", cols, index=cols.index(col_height) if col_height in cols else 0)
    c_age = st.selectbox("Kolom Umur (bulan)", cols, index=cols.index(col_age) if col_age in cols else 0)
    c_sex = st.selectbox("Kolom Jenis Kelamin", cols, index=cols.index(col_sex) if col_sex in cols else 0)

    if c_weight is None or c_height is None or c_age is None or c_sex is None:
        st.error("Pilih semua kolom wajib.")
        st.stop()

    # prepare working copy and cleanup
    dfw = df.copy()
    name_col = None if c_name == "-" else c_name
    if name_col is None:
        dfw["NAMA_ANAK"] = [f"Anak {i+1}" for i in range(len(dfw))]
        name_internal = "NAMA_ANAK"
    else:
        name_internal = c_name

    # clean numeric columns
    for c in [c_weight, c_height, c_age]:
        if dfw[c].dtype == object:
            dfw[c] = dfw[c].astype(str).str.replace(",", ".").str.strip()
        dfw[c] = pd.to_numeric(dfw[c], errors="coerce")

    # drop rows with missing essential data or zero-sentinel (0,0,0)
    before = len(dfw)
    dfw = dfw.dropna(subset=[c_weight, c_height, c_age])
    dfw = dfw[~((dfw[c_weight] == 0) & (dfw[c_height] == 0) & (dfw[c_age] == 0))]
    dropped = before - len(dfw)
    if dropped:
        st.warning(f"{dropped} baris dihapus karena nilai kosong/0 untuk umur/berat/tinggi.")

    # rename canonical
    dfw = dfw.rename(columns={c_weight: "Berat", c_height: "Tinggi", c_age: "Umur", c_sex: "Jenis_Kelamin", name_internal: "NAMA_ANAK"})

    # normalize gender
    dfw["Jenis_Kelamin"] = dfw["Jenis_Kelamin"].apply(map_gender)

    # compute IMT
    dfw["Tinggi_meter"] = dfw["Tinggi"] / 100.0
    dfw["IMT"] = dfw["Berat"] / (dfw["Tinggi_meter"] ** 2)

    # Ensure Umur integer months
    dfw["Umur"] = pd.to_numeric(dfw["Umur"], errors="coerce").round().astype("Int64")

    # Compute Z-score using sex-specific LMS (nearest month fallback)
    with st.spinner("Menghitung Z-score dan klasifikasi..."):
        dfw["Z_IMTU"] = dfw.apply(lambda r: compute_z_for_row(r, lms_boys, lms_girls, value_col="IMT"), axis=1)
        def class_by_z_imtu(z):
            if pd.isna(z):
                return "LMS tidak tersedia"
            if z < -3: return "Sangat Kurus"
            if -3 <= z < -2: return "Kurus"
            if -2 <= z <= 1: return "Normal"
            return "Gemuk"
        dfw["Status_Gizi"] = dfw["Z_IMTU"].apply(class_by_z_imtu)

    # Summary & charts
    st.header("Ringkasan & Visualisasi")
    col1, col2, col3, col4 = st.columns(4)
    total = len(dfw)
    col1.metric("Total anak diproses", total)
    counts = dfw["Status_Gizi"].value_counts()
    col2.metric("Sangat Kurus", int(counts.get("Sangat Kurus",0)))
    col3.metric("Kurus", int(counts.get("Kurus",0)))
    col4.metric("Gemuk", int(counts.get("Gemuk",0)))

    # pie chart
    fig1, ax1 = plt.subplots()
    counts.plot.pie(ax=ax1, autopct="%1.1f%%", ylabel="")
    ax1.set_title("Distribusi Status Gizi")
    st.pyplot(fig1)

    # histogram of Z-scores
    fig2, ax2 = plt.subplots()
    dfw["Z_IMTU"].dropna().plot.hist(ax=ax2, bins=30)
    ax2.set_xlabel("Z-score IMT/U")
    ax2.set_title("Distribusi Z-score IMT/U")
    st.pyplot(fig2)

    # Show results + ML
    st.header("Hasil lengkap")
    st.dataframe(dfw[["NAMA_ANAK","Umur","Jenis_Kelamin","Berat","Tinggi","IMT","Z_IMTU","Status_Gizi"]].reset_index(drop=True))

    st.header("Machine Learning — Gaussian Naive Bayes")
    possible_feats = ["Umur","Berat","Tinggi","IMT"]
    feats = st.multiselect("Pilih fitur untuk model", possible_feats, default=possible_feats)

    X = dfw[feats].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    y = dfw["Status_Gizi"].astype(str)
    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)

    if len(set(y_enc)) < 2:
        st.warning("Label tidak cukup beragam untuk training ML. Tambah lebih banyak data/kelas.")
    else:
        # handle stratify edge-case: if any class has only 1 sample, remove stratify
        counts_cls = pd.Series(y_enc).value_counts()
        use_stratify = counts_cls.min() >= 2
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42)

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Akurasi (test):", f"{accuracy_score(y_test,y_pred)*100:.2f}%")
        st.text(classification_report(y_test,y_pred, target_names=le_y.classes_))
        dfw["Prediksi_ML"] = le_y.inverse_transform(model.predict(X))
        st.subheader("Preview prediksi ML")
        st.dataframe(dfw[["NAMA_ANAK","Status_Gizi","Prediksi_ML"]].head(50))

        csv_bytes = dfw.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download hasil lengkap (CSV)", csv_bytes, "hasil_imtu_who2007.csv", mime="text/csv")
        payload = {"model": model, "features": feats, "label_classes": le_y.classes_.tolist()}
        st.download_button("📦 Download model (pickle)", pickle.dumps(payload), "model_imtu_nb.pkl", mime="application/octet-stream")

    st.success("Selesai — aplikasi memakai LMS WHO 2007 (boys & girls).")

# ======================================================
# 2) HALAMAN BB/U (0–60 bulan) — Weight-for-age (WHO 2006)
# ======================================================
if menu == "BB/U (WHO 2006, 0–60 bulan)":

    st.header("BB/U (WHO 2006 — 0–60 bulan)")
    st.write("Perhitungan BB/U memakai LMS WHO 2006 (Weight-for-Age).")

    # Try to download WFA tables (CSV). If fail, ask user to upload LMS CSV.
    st.sidebar.header("LMS WHO 2006 (BB/U)")
    auto_load = st.sidebar.checkbox("Coba unduh LMS WFA otomatis (internet required)", value=True)

    lms_boys_wfa = None
    lms_girls_wfa = None
    if auto_load:
        try:
            lms_boys_wfa = normalize_lms(download_csv(WHO_WFA_BOYS_CSV))
            lms_girls_wfa = normalize_lms(download_csv(WHO_WFA_GIRLS_CSV))
            st.sidebar.success("LMS WFA (WHO 2006) berhasil diunduh.")
        except Exception as e:
            st.sidebar.error(f"Gagal unduh LMS WFA otomatis: {e}")
            st.info("Silakan upload file LMS WFA (CSV) jika tersedia, atau ubah opsi sidebar.")

    upload_lms = st.file_uploader("Jika unduh otomatis gagal, upload LMS WFA CSV (zip/two files allowed) — upload boys lalu girls", type=["csv"], accept_multiple_files=True, key="upload_wfa")
    if upload_lms:
        try:
            if len(upload_lms) >= 1:
                lms_boys_wfa = normalize_lms(pd.read_csv(upload_lms[0]))
            if len(upload_lms) >= 2:
                lms_girls_wfa = normalize_lms(pd.read_csv(upload_lms[1]))
            st.success("LMS WFA diupload dan dimuat.")
        except Exception as e:
            st.error(f"Gagal memuat file LMS yang diupload: {e}")

    if lms_boys_wfa is None or lms_girls_wfa is None:
        st.warning("LMS WFA belum tersedia. Gunakan tombol unduh otomatis (sidebar) atau upload file LMS.")
        st.stop()

    # Upload children dataset for BB/U
    st.header("Upload dataset anak (CSV/Excel) untuk BB/U")
    uploaded_bbu = st.file_uploader("File anak (kolom: Nama, Berat(kg), Umur(bulan), Jenis Kelamin)", type=["csv","xlsx","xls"], key="bbu_upload")
    if not uploaded_bbu:
        st.info("Upload dataset anak untuk BB/U.")
        st.stop()

    try:
        if str(uploaded_bbu.name).lower().endswith((".xls", ".xlsx")):
            df_bbu = pd.read_excel(uploaded_bbu, engine="openpyxl")
        else:
            df_bbu = pd.read_csv(uploaded_bbu)
    except Exception as e:
        st.error(f"Gagal membaca file anak: {e}")
        st.stop()

    st.subheader("Preview (acak 30%)")
    st.dataframe(df_bbu.sample(max(1,int(len(df_bbu)*0.3)), random_state=42))

    # auto-detect columns for BB/U
    cols_b = df_bbu.columns.tolist()
    col_name_b = find_col(cols_b, ["nama","name"])
    col_weight_b = find_col(cols_b, ["berat","bb","weight"])
    col_age_b = find_col(cols_b, ["umur","usia","age","month"])
    col_sex_b = find_col(cols_b, ["jenis","kelamin","jk","sex","gender"])

    st.subheader("Konfirmasi kolom (BB/U)")
    c_name_b = st.selectbox("Kolom Nama Anak (opsional)", ["-"]+cols_b, index=(0 if col_name_b is None else cols_b.index(col_name_b)+1))
    c_weight_b = st.selectbox("Kolom Berat (kg)", cols_b, index=cols_b.index(col_weight_b) if col_weight_b in cols_b else 0)
    c_age_b = st.selectbox("Kolom Umur (bulan)", cols_b, index=cols_b.index(col_age_b) if col_age_b in cols_b else 0)
    c_sex_b = st.selectbox("Kolom Jenis Kelamin", cols_b, index=cols_b.index(col_sex_b) if col_sex_b in cols_b else 0)

    # prepare df
    dfw_b = df_bbu.copy()
    name_col_b = None if c_name_b == "-" else c_name_b
    if name_col_b is None:
        dfw_b["NAMA_ANAK"] = [f"Anak {i+1}" for i in range(len(dfw_b))]
    else:
        dfw_b["NAMA_ANAK"] = dfw_b[name_col_b]
    dfw_b["Berat"] = pd.to_numeric(dfw_b[c_weight_b], errors="coerce")
    dfw_b["Umur"] = pd.to_numeric(dfw_b[c_age_b], errors="coerce").round().astype("Int64")
    dfw_b["Jenis_Kelamin"] = dfw_b[c_sex_b].apply(map_gender)

    # compute Z using WFA LMS (we treat value_col as 'IMT' placeholder referencing Berat)
    with st.spinner("Menghitung Z-score BB/U dan klasifikasi..."):
        dfw_b["IMT_for_z"] = dfw_b["Berat"]  # use Berat as 'value' for WFA z formula
        dfw_b["Z_WFA"] = dfw_b.apply(lambda r: compute_z_for_row(r, lms_boys_wfa, lms_girls_wfa, value_col="IMT_for_z"), axis=1)
        def class_by_z_bbu(z):
            if pd.isna(z): return "LMS tidak tersedia"
            if z < -3: return "Berat badan sangat kurang"
            if -3 <= z < -2: return "Berat badan kurang"
            if -2 <= z <= 1: return "Berat badan normal"
            return "Berat badan lebih"
        dfw_b["Status_Gizi"] = dfw_b["Z_WFA"].apply(class_by_z_bbu)

    # Summary & charts
    st.header("Ringkasan & Visualisasi (BB/U)")
    col1, col2, col3, col4 = st.columns(4)
    total_b = len(dfw_b)
    counts_b = dfw_b["Status_Gizi"].value_counts()
    col1.metric("Total anak diproses", total_b)
    col2.metric("Sangat Kurus", int(counts_b.get("Berat badan sangat kurang",0)))
    col3.metric("Kurus", int(counts_b.get("Berat badan kurang",0)))
    col4.metric("Gemuk", int(counts_b.get("Berat badan lebih",0)))

    figb, axb = plt.subplots()
    counts_b.plot.pie(ax=axb, autopct="%1.1f%%", ylabel="")
    axb.set_title("Distribusi Status Gizi (BB/U)")
    st.pyplot(figb)

    st.header("Hasil lengkap (BB/U)")
    st.dataframe(dfw_b[["NAMA_ANAK","Umur","Jenis_Kelamin","Berat","Z_WFA","Status_Gizi"]].reset_index(drop=True))

    # ML for BB/U (optional)
    st.header("Machine Learning — Gaussian Naive Bayes (BB/U)")
    possible_feats_b = ["Umur","Berat"]
    feats_b = st.multiselect("Pilih fitur untuk model (BB/U)", possible_feats_b, default=possible_feats_b)

    Xb = dfw_b[feats_b].copy()
    for c in Xb.columns:
        Xb[c] = pd.to_numeric(Xb[c], errors="coerce").fillna(Xb[c].median())
    yb = dfw_b["Status_Gizi"].astype(str)
    le_yb = LabelEncoder()
    yb_enc = le_yb.fit_transform(yb)

    if len(set(yb_enc)) < 2:
        st.warning("Label tidak cukup beragam untuk training ML (BB/U).")
    else:
        counts_cls_b = pd.Series(yb_enc).value_counts()
        use_strat_b = counts_cls_b.min() >= 2
        if use_strat_b:
            Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb_enc, test_size=0.25, random_state=42, stratify=yb_enc)
        else:
            Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb_enc, test_size=0.25, random_state=42)

        model_b = GaussianNB()
        model_b.fit(Xb_train, yb_train)
        yb_pred = model_b.predict(Xb_test)
        st.write("Akurasi (test) BB/U:", f"{accuracy_score(yb_test,yb_pred)*100:.2f}%")
        st.text(classification_report(yb_test,yb_pred, target_names=le_yb.classes_))
        dfw_b["Prediksi_ML"] = le_yb.inverse_transform(model_b.predict(Xb))
        st.subheader("Preview prediksi ML (BB/U)")
        st.dataframe(dfw_b[["NAMA_ANAK","Status_Gizi","Prediksi_ML"]].head(50))

        csv_bytes_b = dfw_b.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download hasil BB/U (CSV)", csv_bytes_b, "hasil_bbu.csv", mime="text/csv")
        payload_b = {"model": model_b, "features": feats_b, "label_classes": le_yb.classes_.tolist()}
        st.download_button("📦 Download model BB/U (pickle)", pickle.dumps(payload_b), "model_bbu_nb.pkl", mime="application/octet-stream")

    st.success("Selesai — Mode BB/U diproses.")
   gini kodinganya
