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
import base64

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Status Gizi Anak — Lengkap", layout="wide")

# -------------------------
# External resources (WHO)
# -------------------------
WHO_IMTU_BOYS = "https://cdn.who.int/media/docs/default-source/child-growth/growth-reference-5-19-years/bmi-for-age-%285-19-years%29/bmi-boys-z-who-2007-exp.xlsx?sfvrsn=a84bca93_2"
WHO_IMTU_GIRLS = "https://cdn.who.int/media/docs/default-source/child-growth/growth-reference-5-19-years/bmi-for-age-%285-19-years%29/bmi-girls-z-who-2007-exp.xlsx?sfvrsn=79222875_2"
WHO_WFA_BOYS = "https://www.who.int/childgrowth/standards/wfa_boys_0_60.csv"
WHO_WFA_GIRLS = "https://www.who.int/childgrowth/standards/wfa_girls_0_60.csv"

# -------------------------
# Helpers and caching
# -------------------------
@st.cache_data(show_spinner=False)
def download_excel(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    x = BytesIO(r.content)
    xls = pd.read_excel(x, sheet_name=None, engine="openpyxl")
    for _, sheet in xls.items():
        cols = [c.lower() for c in sheet.columns.astype(str)]
        if any(k in cols for k in ("l","m","s")) and any(k in cols for k in ("age","month","agemonths")):
            return sheet.copy()
    return next(iter(xls.values())).copy()

@st.cache_data(show_spinner=False)
def download_csv(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))


def normalize_lms(df):
    mapping = {}
    for c in df.columns:
        cl = c.lower().strip().replace(' ', '')
        if 'age' in cl or 'month' in cl:
            mapping[c] = 'Umur'
        elif cl == 'l':
            mapping[c] = 'L'
        elif cl == 'm':
            mapping[c] = 'M'
        elif cl == 's':
            mapping[c] = 'S'
        elif cl in ('sex','gender'):
            mapping[c] = 'Jenis_Kelamin'
    df = df.rename(columns=mapping)
    if 'Umur' in df.columns:
        df['Umur'] = pd.to_numeric(df['Umur'], errors='coerce').round().astype('Int64')
    return df


def map_gender(v):
    if pd.isna(v): return 'U'
    s = str(v).strip().upper()
    if s in ('1','L','M','BOY','MALE','B'): return 'L'
    if s in ('2','P','F','GIRL','FEMALE','G'): return 'P'
    return 'U'


def zscore_who(value, L, M, S):
    try:
        L = float(L); M = float(M); S = float(S); value = float(value)
        if abs(L) < 1e-9:
            return np.log(value / M) / S
        return ((value / M) ** L - 1) / (L * S)
    except Exception:
        return np.nan


def get_lms_row(age_month, sex, table_boys, table_girls):
    table = table_boys if sex == 'L' else table_girls
    if 'Umur' not in table.columns:
        possible = [c for c in table.columns if 'age' in c.lower()]
        if possible:
            table2 = table.rename(columns={possible[0]: 'Umur'})
        else:
            return None
    else:
        table2 = table
    exact = table2[table2['Umur'] == age_month]
    if not exact.empty:
        return exact.iloc[0]
    t = table2.dropna(subset=['Umur'])
    if t.empty:
        return None
    idx = (t['Umur'] - age_month).abs().idxmin()
    return t.loc[idx]


# -------------------------
# UI Helpers
# -------------------------
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


def download_button_bytes(data_bytes, filename, label):
    st.download_button(label, data_bytes, file_name=filename, mime='text/csv')


# -------------------------
# App layout
# -------------------------
st.title('📊 Status Gizi Anak — Versi Lengkap')
st.markdown('Aplikasi menghitung Z-score IMT/U (WHO 2007) dan BB/U (WHO 2006), plus Naive Bayes untuk prediksi. Upload CSV/XLSX siswa.')

page = st.sidebar.radio('Pilih modul', ['Panduan & Contoh', 'IMT/U (5-19y)', 'BB/U (0-60m)', 'Model & Export'])

# ---------- Panduan & contoh ----------
if page == 'Panduan & Contoh':
    st.header('Panduan cepat')
    st.write('- Upload file CSV atau Excel yang berisi kolom: Nama(optional), Berat(kg), Tinggi(cm, untuk IMT), Umur(bulan), Jenis Kelamin')
    st.write('- Untuk IMT/U gunakan halaman IMT/U; untuk BB/U gunakan halaman BB/U')
    st.write('- Aplikasi otomatis mendeteksi nama kolom yang mirip. Jika salah, pilih manual.')
    st.write('- Setelah proses selesai, kamu bisa mendownload hasil dan model (pickle).')

    st.subheader('Contoh dataset kecil (download)')
    example = pd.DataFrame({
        'Nama':['Anak 1','Anak 2','Anak 3','Anak 4'],
        'Berat':[12.4, 14.0, 18.5, 9.1],
        'Tinggi':[90,95,110,78],
        'Umur':[24,30,60,12],
        'Jenis_Kelamin':['L','P','L','P']
    })
    download_button_bytes(example, 'contoh_anak.csv', '📥 Download contoh dataset (CSV)')
    st.info('Jika mau, gunakan contoh ini untuk tes cepat.')

# ---------- IMT/U ----------
if page == 'IMT/U (5-19y)':
    st.header('IMT/U — WHO 2007 (BMI-for-age, 5–19 tahun)')
    st.write('Modul ini menghitung IMT, Z-score IMT/U, klasifikasi, visualisasi, dan model Naive Bayes.')

    # load LMS
    with st.spinner('Memuat tabel LMS WHO 2007 (IMT/U) — koneksi internet diperlukan'):
        try:
            lms_boys_raw = download_excel(WHO_IMTU_BOYS)
            lms_girls_raw = download_excel(WHO_IMTU_GIRLS)
            lms_boys = normalize_lms(lms_boys_raw)
            lms_girls = normalize_lms(lms_girls_raw)
            st.success('LMS WHO IMT/U berhasil dimuat.')
        except Exception as e:
            st.error('Gagal memuat LMS IMT/U: ' + str(e))
            st.stop()

    uploaded = st.file_uploader('Upload dataset anak (CSV / XLSX) — kolom minimal: Berat, Tinggi, Umur, Jenis Kelamin', type=['csv','xlsx'], key='imtu_upload')
    if not uploaded:
        st.info('Silakan upload file untuk memulai proses.')
        st.stop()

    try:
        if uploaded.name.lower().endswith(('xlsx','xls')):
            df = pd.read_excel(uploaded, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error('Gagal membaca file: ' + str(e))
        st.stop()

    st.subheader('Preview & Deteksi kolom')
    st.dataframe(df.head(10))

    cols = df.columns.tolist()
    def find_col(cols, keywords):
        for k in keywords:
            for c in cols:
                if k in c.lower().replace(' ',''):
                    return c
        return None

    c_name = st.selectbox('Kolom Nama (opsional)', ['-']+cols, index=0)
    c_weight = st.selectbox('Kolom Berat (kg)', cols, index=cols.index(find_col(cols,['berat','bb','weight'])) if find_col(cols,['berat','bb','weight']) in cols else 0)
    c_height = st.selectbox('Kolom Tinggi (cm)', cols, index=cols.index(find_col(cols,['tinggi','tb','height'])) if find_col(cols,['tinggi','tb','height']) in cols else 0)
    c_age = st.selectbox('Kolom Umur (bulan)', cols, index=cols.index(find_col(cols,['umur','usia','age','month'])) if find_col(cols,['umur','usia','age','month']) in cols else 0)
    c_sex = st.selectbox('Kolom Jenis Kelamin', cols, index=cols.index(find_col(cols,['jenis','kelamin','jk','sex'])) if find_col(cols,['jenis','kelamin','jk','sex']) in cols else 0)

    # prepare df
    dfw = df.copy()
    if c_name == '-':
        dfw['NAMA_ANAK'] = [f'Anak {i+1}' for i in range(len(dfw))]
    else:
        dfw['NAMA_ANAK'] = dfw[c_name]

    for c in [c_weight, c_height, c_age]:
        dfw[c] = dfw[c].astype(str).str.replace(',','.')
        dfw[c] = pd.to_numeric(dfw[c], errors='coerce')

    dfw = dfw.dropna(subset=[c_weight, c_height, c_age])
    dfw = dfw[~((dfw[c_weight]==0) & (dfw[c_height]==0) & (dfw[c_age]==0))]

    dfw = dfw.rename(columns={c_weight:'Berat', c_height:'Tinggi', c_age:'Umur'})
    dfw['Jenis_Kelamin'] = dfw[c_sex].apply(map_gender)
    dfw['Tinggi_meter'] = dfw['Tinggi'] / 100.0
    dfw['IMT'] = dfw['Berat'] / (dfw['Tinggi_meter']**2)
    dfw['Umur'] = pd.to_numeric(dfw['Umur'], errors='coerce').round().astype('Int64')

    # compute z
    with st.spinner('Menghitung Z-score dan klasifikasi...'):
        dfw['Z_IMTU'] = dfw.apply(lambda r: (lambda row: zscore_who(r['IMT'], row['L'], row['M'], row['S']) if row is not None else np.nan)(get_lms_row(r['Umur'], r['Jenis_Kelamin'], lms_boys, lms_girls)), axis=1)

    def class_by_z(z):
        if pd.isna(z): return 'LMS tidak tersedia'
        if z < -3: return 'Sangat Kurus'
        if -3 <= z < -2: return 'Kurus'
        if -2 <= z <= 1: return 'Normal'
        return 'Gemuk'

    dfw['Status_Gizi'] = dfw['Z_IMTU'].apply(class_by_z)

    st.subheader('Ringkasan')
    total = len(dfw)
    st.metric('Total anak diproses', total)
    counts = dfw['Status_Gizi'].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Sangat Kurus', int(counts.get('Sangat Kurus',0)))
    col2.metric('Kurus', int(counts.get('Kurus',0)))
    col3.metric('Normal', int(counts.get('Normal',0)))
    col4.metric('Gemuk', int(counts.get('Gemuk',0)))

    # charts
    fig1, ax1 = plt.subplots()
    counts.plot.pie(ax=ax1, autopct='%1.1f%%', ylabel='')
    ax1.set_title('Distribusi Status Gizi')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    dfw['Z_IMTU'].dropna().plot.hist(ax=ax2, bins=30)
    ax2.set_xlabel('Z-score IMT/U')
    ax2.set_title('Distribusi Z-score IMT/U')
    st.pyplot(fig2)

    st.subheader('Tabel Hasil')
    st.dataframe(dfw[['NAMA_ANAK','Umur','Jenis_Kelamin','Berat','Tinggi','IMT','Z_IMTU','Status_Gizi']].reset_index(drop=True))

    # ML
    st.header('Machine Learning — Gaussian Naive Bayes')
    possible_feats = ['Umur','Berat','Tinggi','IMT']
    feats = st.multiselect('Pilih fitur untuk model', possible_feats, default=possible_feats)

    X = dfw[feats].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(X[c].median())
    y = dfw['Status_Gizi'].astype(str)
    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)

    if len(set(y_enc)) < 2:
        st.warning('Label tidak cukup beragam untuk training ML. Tambah lebih banyak data/kelas.')
    else:
        counts_cls = pd.Series(y_enc).value_counts()
        use_stratify = counts_cls.min() >= 2
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42)

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write('Akurasi (test):', f"{accuracy_score(y_test,y_pred)*100:.2f}%")
        st.text(classification_report(y_test,y_pred, target_names=le_y.classes_))

        dfw['Prediksi_ML'] = le_y.inverse_transform(model.predict(X))
        st.subheader('Preview prediksi ML')
        st.dataframe(dfw[['NAMA_ANAK','Status_Gizi','Prediksi_ML']].head(50))

        # download results
        st.download_button('📥 Download hasil lengkap (CSV)', df_to_csv_bytes(dfw), 'hasil_imtu.csv', mime='text/csv')
        payload = {'model': model, 'features': feats, 'label_classes': le_y.classes_.tolist()}
        st.download_button('📦 Download model (pickle)', pickle.dumps(payload), 'model_imtu_nb.pkl', mime='application/octet-stream')

# ---------- BB/U ----------
if page == 'BB/U (0-60m)':
    st.header('BB/U — WHO 2006 (Weight-for-age 0–60 months)')

    auto = st.checkbox('Coba unduh LMS WFA otomatis (internet diperlukan)', value=True)
    lms_boys_wfa = lms_girls_wfa = None
    if auto:
        with st.spinner('Mengunduh LMS WFA WHO...'):
            try:
                lms_boys_wfa = normalize_lms(download_csv(WHO_WFA_BOYS))
                lms_girls_wfa = normalize_lms(download_csv(WHO_WFA_GIRLS))
                st.success('LMS WFA berhasil dimuat.')
            except Exception as e:
                st.error('Gagal unduh LMS WFA: ' + str(e))

    uploaded_bbu = st.file_uploader('Upload dataset anak (CSV/XLSX) untuk BB/U', type=['csv','xlsx'], key='bbu_upload')
    if not uploaded_bbu:
        st.info('Silakan upload dataset untuk lanjut.')
        st.stop()

    try:
        if uploaded_bbu.name.lower().endswith(('xlsx','xls')):
            dfb = pd.read_excel(uploaded_bbu, engine='openpyxl')
        else:
            dfb = pd.read_csv(uploaded_bbu)
    except Exception as e:
        st.error('Gagal membaca file: ' + str(e))
        st.stop()

    st.subheader('Preview')
    st.dataframe(dfb.head(10))

    cols_b = dfb.columns.tolist()
    col_weight_b = st.selectbox('Kolom Berat (kg)', cols_b, index=cols_b.index(next((c for c in cols_b if 'berat' in c.lower() or 'bb' in c.lower() or 'weight' in c.lower()), cols_b[0])))
    col_age_b = st.selectbox('Kolom Umur (bulan)', cols_b, index=cols_b.index(next((c for c in cols_b if 'umur' in c.lower() or 'age' in c.lower()), cols_b[0])))
    col_sex_b = st.selectbox('Kolom Jenis Kelamin', cols_b, index=cols_b.index(next((c for c in cols_b if 'jenis' in c.lower() or 'jk' in c.lower() or 'sex' in c.lower()), cols_b[0])))

    if lms_boys_wfa is None or lms_girls_wfa is None:
        st.warning('LMS WFA belum tersedia. Jika internet bermasalah, unggah file LMS WFA (CSV) di sidebar.')

    dfb['Berat'] = pd.to_numeric(dfb[col_weight_b].astype(str).str.replace(',','.'), errors='coerce')
    dfb['Umur'] = pd.to_numeric(dfb[col_age_b], errors='coerce').round().astype('Int64')
    dfb['Jenis_Kelamin'] = dfb[col_sex_b].apply(map_gender)

    with st.spinner('Menghitung Z-score WFA...'):
        dfb['Z_WFA'] = dfb.apply(lambda r: (lambda row: zscore_who(r['Berat'], row['L'], row['M'], row['S']) if row is not None else np.nan)(get_lms_row(r['Umur'], r['Jenis_Kelamin'], lms_boys_wfa, lms_girls_wfa)), axis=1)

    def class_bbu(z):
        if pd.isna(z): return 'LMS tidak tersedia'
        if z < -3: return 'Berat badan sangat kurang'
        if z < -2: return 'Berat badan kurang'
        if z <= 1: return 'Berat badan normal'
        return 'Berat badan lebih'

    dfb['Status_Gizi'] = dfb['Z_WFA'].apply(class_bbu)

    st.subheader('Ringkasan BB/U')
    counts_b = dfb['Status_Gizi'].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Sangat Kurus', int(counts_b.get('Berat badan sangat kurang',0)))
    col2.metric('Kurus', int(counts_b.get('Berat badan kurang',0)))
    col3.metric('Normal', int(counts_b.get('Berat badan normal',0)))
    col4.metric('Lebih', int(counts_b.get('Berat badan lebih',0)))

    figb, axb = plt.subplots()
    counts_b.plot.pie(ax=axb, autopct='%1.1f%%', ylabel='')
    axb.set_title('Distribusi Status Gizi (BB/U)')
    st.pyplot(figb)

    st.subheader('Tabel Hasil BB/U')
    st.dataframe(dfb[['Umur','Jenis_Kelamin','Berat','Z_WFA','Status_Gizi']].reset_index(drop=True))

    # optional ML
    st.header('Machine Learning (opsional) — BB/U')
    possible_feats_b = ['Umur','Berat']
    feats_b = st.multiselect('Pilih fitur untuk model (BB/U)', possible_feats_b, default=possible_feats_b)
    Xb = dfb[feats_b].copy()
    for c in Xb.columns:
        Xb[c] = pd.to_numeric(Xb[c], errors='coerce').fillna(Xb[c].median())
    yb = dfb['Status_Gizi'].astype(str)
    le_yb = LabelEncoder()
    yb_enc = le_yb.fit_transform(yb)

    if len(set(yb_enc)) >= 2:
        Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb_enc, test_size=0.25, random_state=42)
        model_b = GaussianNB()
        model_b.fit(Xb_train, yb_train)
        yb_pred = model_b.predict(Xb_test)
        st.write('Akurasi (test) BB/U:', f"{accuracy_score(yb_test,yb_pred)*100:.2f}%")
        st.text(classification_report(yb_test,yb_pred, target_names=le_yb.classes_))
        dfb['Prediksi_ML'] = le_yb.inverse_transform(model_b.predict(Xb))
        st.download_button('📥 Download hasil BB/U (CSV)', df_to_csv_bytes(dfb), 'hasil_bbu.csv', mime='text/csv')
        payload_b = {'model': model_b, 'features': feats_b, 'label_classes': le_yb.classes_.tolist()}
        st.download_button('📦 Download model BB/U (pickle)', pickle.dumps(payload_b), 'model_bbu_nb.pkl', mime='application/octet-stream')
    else:
        st.info('Label tidak cukup beragam untuk training BB/U. Model tidak dibuat.')

# ---------- Model & Export ----------
if page == 'Model & Export':
    st.header('Tooling & Export')
    st.write('Seksi ini untuk menaruh/menyimpan model, melihat sample, dan panduan deployment.')
    st.subheader('Requirements yang disarankan (copy ke file requirements.txt)')
    st.code('\n'.join(['streamlit','pandas','numpy','scikit-learn','openpyxl','matplotlib','requests']))

    st.subheader('Panduan deploy singkat')
    st.write('1. Pastikan file `app.py` di root repository GitHub.\n2. Tambahkan `requirements.txt` berisi dependensi di atas.\n3. Deploy di Streamlit Cloud: pilih repo -> main branch -> main file `app.py`.')

    st.subheader('Catatan')
    st.write('- Jika kamu menggunakan dataset besar, pertimbangkan mengupload dataset ke Drive atau bucket, lalu akses dari sana.\n- LMS WHO diunduh saat runtime — membutuhkan koneksi internet di server deploy.')

    st.success('Selesai — buka halaman IMT/U atau BB/U untuk mulai pakai aplikasi.')
