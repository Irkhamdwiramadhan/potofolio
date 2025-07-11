# utils.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import xgboost as xgb
import calendar 
from sklearn.linear_model import LinearRegression # <<< TAMBAHKAN BARIS INI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # <<< TAMBAHKAN BARIS INI
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import shapiro, normaltest, pearsonr, spearmanr
from sklearn.tree import DecisionTreeClassifier, plot_tree


# --- ABOUT ME --- (Tetap sama)
MY_NAME = "M Irkham Dwi Ramadhan"
MY_TITLE_OR_SPECIALIZATION = "Data Analyst | Data Scientist | Data Engineer"
ABOUT_ME_TEXT = """
Halo! Saya **M Irkham Dwi Ramadhan**, seorang mahasiswa Teknik Informatika semester 5 yang berfokus mendalam pada bidang **Data Engineering, Data Analysis, dan Data Science**.
Saya memiliki latar belakang yang kuat dalam memanfaatkan data untuk menciptakan wawasan yang dapat ditindaklanjuti dan mendorong keputusan bisnis yang informatif. Saya berpengalaman dalam mengembangkan dan mengimplementasikan analisis statistik untuk mengungkap pola dan tren, melakukan pembersihan dan pengayaan data, serta memanfaatkan algoritma pembelajaran mesin.
"""

# Data untuk Pie Chart Bidang (Tetap sama)
BIDANG_PEMAHAMAN = {
    "Data Analyst": 50,
    "Data Science": 30,
    "Data Engineering": 20
}

# Data untuk Bar Chart Skill (Amatir, Beginner, Medium, Profesional) (Tetap sama)
SKILLS_LEVELS = {
    "Python": "Profesional",
    "HTML": "Profesional",
    "CSS": "Medium",
    "Laravel": "Amatir",
    "SQL": "Beginner",
    "Excel": "Medium"
}
LEVEL_MAPPING = {
    "Amatir": 1,
    "Beginner": 2,
    "Medium": 3,
    "Profesional": 4
}

# SKILLS_RATING lama (opsional, tidak dipakai di visualisasi baru) (Tetap sama)
SKILLS_RATING = {
    "Data Analysis": 4.5, "Data Engineering": 3.5, "Data Science": 3.5, "Python": 4.5,
    "SQL": 3.0, "Pandas & NumPy": 4.0, "Matplotlib & Seaborn": 3.5, "Plotly & Altair": 3.0,
    "Scikit-learn": 3.5, "Google Colab": 4.0, "Streamlit": 3.0, "Jupyter Notebook": 4.0,
}

TOOLS = [
    "Google Collab", "Streamlit", "Visual Studio Code", "Jupyter Notebook",
    "Git/GitHub", "Tableau", "Power BI"
]

CONTACT_INFO = {
    "Email": "irkhamdr19@gmail.com",
    "Phone": "0823-1392-7515",
    "LinkedIn": "https://www.linkedin.com/in/m-irkham-dwi-ramadhan-689aaa2a4/", # <<< GANTI DENGAN LINK LINKEDIN ANDA
    "GitHub": "https://github.com/Irkhamdwiramadhan", # <<< GANTI DENGAN LINK GITHUB ANDA
    "Medium": "https://medium.com/@irkhamdr19" # <<< GANTI DENGAN LINK MEDIUM ANDA
}

EDUCATION = [
    {"institution": "STT Terpadu NF", "degree": "S1 Teknik Informatika", "years": "2023 - Sekarang", "description": "Peminatan Data Engineering"},
    {"institution": "SMA N 01 Sirampog", "degree": "", "years": "2020-2023"},
    {"institution": "SMP N 02 Sirampog", "degree": "", "years": "2017-2019"},
    {"institution": "SDN 04 Dukuhbenda", "degree": "", "years": "2011-2016"},
]

# Perbarui PROJECT_SUMMARY_DATA untuk menyertakan proyek bot telegram
PROJECT_SUMMARY_DATA = pd.DataFrame({
    'Proyek': ['Prediksi Pemain Bola', 'Analisis E-commerce', 'Prediksi Kredivo', 'Analisis Perceraian', 'Analisis Kemiskinan', 'Bot AI Manage Keuangan'],
    'Jenis Proyek': ['Machine Learning', 'Data Analysis & Viz', 'Machine Learning', 'Data Analysis & Viz', 'Data Analysis & ML', 'AI Application'],
    'Skill Utama': ['Scikit-learn', 'Visualisasi Data', 'Random Forest', 'Pandas, Matplotlib', 'XGBoost, Pandas', 'Python, NLP, SQLite'],
    'Kompleksitas': ['Tinggi', 'Menengah', 'Tinggi', 'Menengah', 'Tinggi', 'Menengah'],
    'Akurasi Estimasi': [0.55, np.nan, 0.78, np.nan, 0.32, np.nan], # Tidak ada akurasi untuk bot
    'Tahun': [2024, 2023, 2024, 2024, 2024, 2025]
})

# --- Fungsi-fungsi Pembantu Streamlit ---

def set_page_config(title="Portofolio Data M Irkham"):
    """Mengatur konfigurasi dasar halaman Streamlit."""
    st.set_page_config(
        page_title=title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def display_section_header(title):
    """Menampilkan header bagian dengan garis pemisah."""
    st.markdown(f"## {title}")
    st.markdown("---")

def display_about_me_summary():
    """Menampilkan ringkasan profil diri (foto, nama, deskripsi singkat) di main content."""
    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    with col1:
        st.image("pp.png", width=250) # Ganti dengan path Anda & sesuaikan width
    with col2:
        st.title(MY_NAME)
        st.write(f"### {MY_TITLE_OR_SPECIALIZATION}")
        st.markdown(ABOUT_ME_TEXT)
        
    st.markdown("---")

def display_sidebar_profile():
    """Menampilkan profil singkat (foto dan nama) di sidebar."""
    st.image("pp_sidebar.png", width=100) # Sesuaikan width untuk sidebar
    st.markdown(f"### {MY_NAME}")
    st.markdown(f"*{MY_TITLE_OR_SPECIALIZATION}*")
    st.markdown("---")

def display_skills_visualization():
    """Menampilkan visualisasi interaktif dari profil skill teknis dan pemahaman bidang."""
    st.subheader("Profil Skill Teknis & Pemahaman Bidang")
    
    col_pie, col_bar = st.columns([1, 2])

    with col_pie:
        st.markdown("#### Pemahaman Bidang Utama")
        df_bidang = pd.DataFrame(BIDANG_PEMAHAMAN.items(), columns=['Bidang', 'Persentase'])
        fig_pie = px.pie(df_bidang, values='Persentase', names='Bidang', 
                         title='Pemahaman Terkait Bidang',
                         hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu,
                         template="plotly_dark")
        fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0, 0])
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.markdown("#### Tingkat Kemahiran Skill Spesifik")
        df_skills_level = pd.DataFrame(SKILLS_LEVELS.items(), columns=['Skill', 'Level'])
        df_skills_level['Level_Numeric'] = df_skills_level['Level'].map(LEVEL_MAPPING)
        
        df_skills_level_sorted = df_skills_level.sort_values(by='Level_Numeric', ascending=True)

        fig_bar = px.bar(df_skills_level_sorted, 
                         x='Level_Numeric',
                         y='Skill', 
                         orientation='h', 
                         color='Level', 
                         category_orders={"Level": ["Amatir", "Beginner", "Medium", "Profesional"]}, 
                         color_discrete_map={
                             "Amatir": "#FF6347",
                             "Beginner": "#FFD700",
                             "Medium": "#32CD32",
                             "Profesional": "#1E90FF"
                         }, 
                         template="plotly_dark",
                         title="Tingkat Kemahiran Skill")
        
        fig_bar.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(LEVEL_MAPPING.values()),
                ticktext=list(LEVEL_MAPPING.keys()),
                title="Tingkat Kemahiran"
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def display_tools():
    """Menampilkan daftar tools yang dikuasai dalam format kolom."""
    st.subheader("Tools yang Dikuasai")
    tools_cols = st.columns(3)
    for i, tool in enumerate(TOOLS):
        with tools_cols[i % 3]:
            st.markdown(f"- **{tool}**")

def display_project_summary_visualization():
    """Menampilkan visualisasi ringkasan proyek berdasarkan kategori, kompleksitas, dan akurasi."""
    st.subheader("Ringkasan Proyek Berdasarkan Kategori")

    fig_jenis = px.pie(PROJECT_SUMMARY_DATA, names='Jenis Proyek', title='Distribusi Jenis Proyek',
                       hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel,
                       template="plotly_dark")
    st.plotly_chart(fig_jenis, use_container_width=True)

    st.subheader("Kompleksitas Proyek")
    fig_kompleksitas = px.bar(PROJECT_SUMMARY_DATA, x='Proyek', y='Kompleksitas',
                              color='Kompleksitas', template="plotly_dark",
                              title='Kompleksitas Proyek Saya')
    st.plotly_chart(fig_kompleksitas, use_container_width=True)

    st.subheader("Akurasi Model Proyek (Jika Ada)")
    df_accuracy = PROJECT_SUMMARY_DATA.dropna(subset=['Akurasi Estimasi'])
    if not df_accuracy.empty:
        fig_acc = px.bar(df_accuracy, x='Proyek', y='Akurasi Estimasi',
                         color='Akurasi Estimasi', color_continuous_scale=px.colors.sequential.Plasma,
                         template="plotly_dark",
                         title='Akurasi Estimasi Model Proyek')
        st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("Tidak ada proyek dengan akurasi model untuk ditampilkan saat ini.")

# --- Fungsi-fungsi untuk Halaman Proyek ---

def page_prediksi_kredivo():
    display_section_header("💰 Analisis Determinan Keterlambatan Pembayaran Lebih dari Satu Tahun pada Nasabah Kredivo")
    st.write("""
    Proyek ini menganalisis faktor-faktor yang mempengaruhi keterlambatan pembayaran lebih dari satu tahun pada nasabah Kredivo dan menggunakan algoritma Random Forest untuk memprediksi risiko tersebut.
    """)
    st.subheader("Latar Belakang & Permasalahan")
    st.write("""
    Kredivo adalah contoh fintech di Indonesia dengan konsep pinjaman tanpa kartu kredit dengan proses pendaftaran serta pencairan dana yang cepat. Permasalahannya adalah mengidentifikasi variabel yang mempengaruhi keterlambatan pembayaran nasabah Kredivo lebih dari satu tahun, serta menguji apakah model prediktif seperti Random Forest dapat mengidentifikasi faktor utama penyebab keterlambatan.
    """)
    
    # --- DATA LOADING AND PREPROCESSING ---
    @st.cache_data
    def load_and_preprocess_kredivo_data():
        try:
            # Ganti dengan path lokal Anda yang benar.
            # Pastikan file 'Copy of DATA_KREDIVO_2024.xlsx' ada di direktori yang sama dengan utils.py
            df = pd.read_excel('Copy of DATA_KREDIVO_2024.xlsx')
            st.success("Dataset Kredivo berhasil dimuat!")
        except FileNotFoundError:
            st.error("Error: 'Copy of DATA_KREDIVO_2024.xlsx' tidak ditemukan. Pastikan file ada di direktori yang sama dengan app.py/utils.py.")
            return pd.DataFrame() # Mengembalikan DataFrame kosong jika file tidak ditemukan
        except Exception as e:
            st.error(f"Error saat memuat 'Copy of DATA_KREDIVO_2024.xlsx': {e}")
            return pd.DataFrame()
        
        # Preparing data - Cek duplikasi data
        if df.duplicated().sum() > 0:
            st.warning(f"Ditemukan {df.duplicated().sum()} duplikasi data. Duplikasi akan dihapus.")
            df.drop_duplicates(inplace=True)
        
        # Menghitung usia dari birth_date
        df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
        current_time = pd.Timestamp.now()
        df['age'] = (current_time - df['birth_date']).dt.days // 365

        # Menghapus kolom yang tidak digunakan
        columns_to_drop = [ 'debtor_id', 'legal_name', 'activation_date', 'va_number', 'address_legal',
                           'address_user', 'last payment date', 'last payment channel', 'assignment_date', 'expiry date',
                           'last call result', 'last visit result', 'month_in_agency','transaction date(max dpd)',
                           'religion', 'employer_name', 'employer_number', 'emergency_contact_name', 'emergency_contact_number',
                           'residence_address', 'residence_city', 'residence_kecamatan', 'residence_kelurahan', 'residence_postal_code',
                           'employer_address', 'user_type', 'sdp', 'main_address', 'submit_report', 'area_level_3', 'assignment_name',
                           'bar', 'user_collector_name', 'AGENT']
        # Pastikan kolom-kolom yang akan dihapus ada di DataFrame
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

        # Mengelompokkan umur berdasarkan kategori Kemkes
        bins = [0, 10, 20, 40, 60, 120]
        labels = ['Anak', 'Remaja', 'Dewasa Muda', 'Dewasa', 'Lansia']
        df['age_group_kemkes'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        # Encoding variabel kategorikal untuk model
        if 'gender' in df.columns:
            df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
        if 'employment_type' in df.columns:
            df['employment_type_encoded'] = LabelEncoder().fit_transform(df['employment_type'])
        if 'loan_type' in df.columns:
            df['loan_type_encoded'] = LabelEncoder().fit_transform(df['loan_type'])
        if 'marital_status' in df.columns:
            df['marital_status_encoded'] = LabelEncoder().fit_transform(df['marital_status'])
        
        # Membuat target (binary classification untuk keterlambatan)
        df['late_payment'] = (df['dpd_in'] > 365).astype(int)

        return df

    df_kredivo = load_and_preprocess_kredivo_data()

    if df_kredivo.empty:
        st.stop()

    st.subheader("Contoh Data Setelah Preprocessing")
    st.dataframe(df_kredivo.head())

    st.subheader("Tahapan Analisis & Model")
    st.write("""
    Proyek ini melalui beberapa tahapan utama:
    1.  **Pengumpulan Data**: Data diperoleh dari perusahaan PT Kredivo Finance Indonesia.
    2.  **Preprocessing Data**: Pembersihan data (nilai hilang, duplikasi), penghapusan kolom tidak perlu, *rename* kolom, dan pembuatan kolom umur.
    3.  **Exploratory Data Analysis (EDA)**: Melakukan analisis deskriptif dan visualisasi untuk memahami karakteristik nasabah dan keterlambatan pembayaran.
    4.  **Pembangunan Model Random Forest**: Membangun model untuk memprediksi atau mengklasifikasikan keterlambatan.
    5.  **Evaluasi Model**: Menilai performa model.
    """)
    st.subheader("Statistik Deskriptif")
    st.dataframe(df_kredivo.describe())

    # --- EDA VISUALIZATIONS ---
    st.subheader("Distribusi Keterlambatan (dpd_in)")
    st.write("Distribusi DPD menunjukkan sebagian besar nasabah yang terlambat berada di sekitar 365 hari (1 tahun).")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_kredivo['dpd_in'], bins=30, kde=True, color='blue', ax=ax)
    ax.set_title('Distribusi Keterlambatan (dpd_in)')
    ax.set_xlabel('Keterlambatan (hari)')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    st.subheader("Distribusi Outstanding Amount")
    st.write("""
    Sebagian besar nasabah yang terlambat membayar lebih dari 1 tahun memiliki nilai *outstanding* rendah hingga sedang. Hal ini mungkin disebabkan oleh kemudahan akses pinjaman dengan limit kecil.
    """)
    st.write("""
    *Walaupun jumlahnya sedikit*, nasabah dengan *outstanding* yang sangat tinggi (*outlier*) memiliki dampak signifikan terhadap total kerugian. Analisis lebih mendalam diperlukan untuk memahami profil nasabah ini.
    """)
    fig_oa, ax_oa = plt.subplots(figsize=(12, 6))
    sns.histplot(df_kredivo['outstanding_amount_in'], kde=True, bins=30, color='skyblue', edgecolor='black', ax=ax_oa)
    
    mean_outstanding = df_kredivo['outstanding_amount_in'].mean()
    median_outstanding = df_kredivo['outstanding_amount_in'].median()
    ax_oa.axvline(mean_outstanding, color='red', linestyle='--', linewidth=2, label=f'Rata-rata: Rp {mean_outstanding:,.0f}')
    ax_oa.axvline(median_outstanding, color='green', linestyle='--', linewidth=2, label=f'Median: Rp {median_outstanding:,.0f}')
    
    import matplotlib.ticker as mtick
    ax_oa.xaxis.set_major_formatter(mtick.StrMethodFormatter('Rp {x:,.0f}'))
    ax_oa.set_title('Distribusi Outstanding Amount Jumlah total tagihan Nasabah yang Telat > 1 Tahun', fontsize=14)
    ax_oa.set_xlabel('Outstanding Amount (dalam Rupiah)', fontsize=12)
    ax_oa.set_ylabel('Frekuensi', fontsize=12)
    ax_oa.legend(loc='upper right', fontsize=10)
    ax_oa.grid(alpha=0.3)
    fig_oa.tight_layout()
    st.pyplot(fig_oa)

    st.subheader("Hubungan Usia dan Outstanding Amount")
    st.write("Visualisasi hubungan antara usia nasabah dan jumlah *outstanding* mereka, dibedakan berdasarkan jenis pekerjaan.")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_kredivo, x='age', y='outstanding_amount_in', hue='employment_type', ax=ax_scatter)
    ax_scatter.set_title('Hubungan Usia dan Outstanding Amount')
    ax_scatter.set_xlabel('Usia')
    ax_scatter.set_ylabel('Outstanding Amount (Rp)')
    st.pyplot(fig_scatter)

    st.subheader("Korelasi Antar Fitur")
    st.write("""
    Heatmap korelasi menunjukkan hubungan linear antara variabel-variabel numerik dalam dataset.
    """)
    
    correlation_cols = [
        'user_id', 'dpd_in', 'outstanding_amount_in',
        'outstanding_principal_in', 'last payment amount', 'tenure(max dpd)', 'age'
    ]
    # Filter df_kredivo to only include columns present in correlation_cols and are numeric
    numeric_df_kredivo = df_kredivo[[col for col in correlation_cols if col in df_kredivo.columns and pd.api.types.is_numeric_dtype(df_kredivo[col])]]
    corr_matrix = numeric_df_kredivo.corr(numeric_only=True)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    ax_corr.set_title('Correlation Heatmap')
    st.pyplot(fig_corr)

    st.subheader("Analisis Korelasi (Pearson/Spearman)")
    st.write("Analisis ini memilih uji korelasi Pearson (jika data normal) atau Spearman (jika tidak normal) untuk menguji hubungan antara DPD dan variabel independen lainnya.")
    var1 = "dpd_in"
    independent_vars = ["outstanding_amount_in", "outstanding_principal_in", "last payment amount", "tenure(max dpd)", "age"]
    
    correlation_results = []
    for var in independent_vars:
        # Check if column exists and is numeric
        if var in df_kredivo.columns and pd.api.types.is_numeric_dtype(df_kredivo[var]):
            try:
                if len(df_kredivo[var].unique()) > 1: # Check if there's variance
                    _, p_var1 = shapiro(df_kredivo[var1])[1]
                    _, p_var = shapiro(df_kredivo[var])[1]
                else: # Data is constant, not normal
                    p_var1 = 0
                    p_var = 0

            except Exception:
                p_var1 = 0
                p_var = 0

            if p_var1 > 0.05 and p_var > 0.05:
                corr, p_value = pearsonr(df_kredivo[var1], df_kredivo[var])
                corr_type = "Pearson"
            else:
                corr, p_value = spearmanr(df_kredivo[var1], df_kredivo[var])
                corr_type = "Spearman"
            
            correlation_results.append({
                "Variabel Independen": var,
                "Jenis Korelasi": corr_type,
                "Koefisien Korelasi": f"{corr:.2f}",
                "P-value": f"{p_value:.4f}"
            })
        else:
            correlation_results.append({
                "Variabel Independen": var,
                "Jenis Korelasi": "N/A",
                "Koefisien Korelasi": "N/A",
                "P-value": "N/A"
            })
    st.dataframe(pd.DataFrame(correlation_results))

    st.subheader("Jumlah Nasabah Berdasarkan Jenis Pinjaman")
    st.write("""
    Mengelompokkan nasabah berdasarkan jenis pinjaman yang mereka ambil, menunjukkan distribusi nasabah yang terlambat.
    * **KREDIFAZZ**: Biasanya merupakan layanan pinjaman cepat dengan proses sederhana, di mana nasabah mendapatkan uang tunai yang langsung ditransfer ke rekening mereka.
    * **KREDIVO BILLER**: Digunakan oleh nasabah untuk membayar tagihan seperti listrik, air, internet, atau layanan lain yang didukung oleh Kredivo.
    * **SHOPEE VA TRANSFER**: Merupakan pinjaman yang digunakan oleh nasabah untuk belanja di platform Shopee dengan metode pembayaran menggunakan Kredivo.
    * **TOKOPEDIA, LAZADA**: Fasilitas pinjaman yang digunakan untuk pembelian barang atau layanan di platformnya masing-masing, difasilitasi oleh Kredivo.
    """)
    jenis_pinjaman = df_kredivo.groupby('loan_type').agg({'user_id': 'count'}).rename(columns={'user_id': 'jumlah nasabah'}).reset_index()
    top_10_loan_type = jenis_pinjaman.sort_values(by='jumlah nasabah', ascending=False).head(10)
    fig_loan_type, ax_loan_type = plt.subplots(figsize=(10, 7))
    sns.barplot(x='jumlah nasabah', y='loan_type', data=top_10_loan_type, palette='viridis', ax=ax_loan_type)
    ax_loan_type.set_title('Top 10 Jenis Pinjaman dengan Jumlah Nasabah Terlambat Terbanyak')
    ax_loan_type.set_xlabel('Jumlah Nasabah')
    ax_loan_type.set_ylabel('Jenis Pinjaman')
    st.pyplot(fig_loan_type)

    st.subheader("Jumlah Nasabah Berdasarkan Gender")
    fig_gender, ax_gender = plt.subplots(figsize=(8, 5))
    sns.countplot(x='gender', data=df_kredivo, ax=ax_gender, palette='Blues_d')
    ax_gender.set_xlabel('Jenis Kelamin')
    ax_gender.set_ylabel('Jumlah Nasabah')
    ax_gender.set_title('Jumlah Nasabah yang Telat Membayar Berdasarkan Gender')
    st.pyplot(fig_gender)

    st.subheader("Jumlah Nasabah Berdasarkan Jenis Pekerjaan")
    fig_emp, ax_emp = plt.subplots(figsize=(8, 5))
    sns.countplot(x='employment_type', data=df_kredivo, ax=ax_emp, palette='Greens_d')
    ax_emp.set_xlabel('Jenis Pekerjaan')
    ax_emp.set_ylabel('Jumlah Nasabah')
    ax_emp.set_title('Jumlah Nasabah yang Telat Membayar Berdasarkan Jenis Pekerjaan')
    st.pyplot(fig_emp)

    st.subheader("Jumlah Nasabah Berdasarkan Status Perkawinan")
    fig_marital, ax_marital = plt.subplots(figsize=(8, 5))
    sns.countplot(x='marital_status', data=df_kredivo, ax=ax_marital, palette='Oranges_d')
    ax_marital.set_xlabel('Status Perkawinan')
    ax_marital.set_ylabel('Jumlah Nasabah')
    ax_marital.set_title('Jumlah Nasabah Berdasarkan Status Perkawinan')
    st.pyplot(fig_marital)

    st.subheader("Jumlah Nasabah Berdasarkan Kategori Umur (Kemkes)")
    st.write("Mengelompokkan nasabah berdasarkan kategori umur sesuai standar Kemenkes.")
    fig_age_kemkes, ax_age_kemkes = plt.subplots(figsize=(10, 6))
    sns.countplot(x='age_group_kemkes', data=df_kredivo, ax=ax_age_kemkes, palette='cubehelix')
    ax_age_kemkes.set_title('Jumlah Nasabah Berdasarkan Kategori Umur (Kemkes)')
    ax_age_kemkes.set_xlabel('Kategori Umur (Kemkes)')
    ax_age_kemkes.set_ylabel('Jumlah Nasabah')
    st.pyplot(fig_age_kemkes)


    # --- RANDOM FOREST MODEL ---
    st.subheader("Model Prediktif: Random Forest Classifier")
    st.write("Model Random Forest Classifier digunakan untuk memprediksi apakah seorang nasabah akan terlambat membayar lebih dari satu tahun (`late_payment`).")
    
    # Memisahkan fitur dan target
    model_features = [
        'outstanding_amount_in', 'outstanding_principal_in', 'tenure(max dpd)',
        'employment_type_encoded', 'loan_type_encoded', 'gender_encoded'
    ]
    X = df_kredivo[[col for col in model_features if col in df_kredivo.columns and pd.api.types.is_numeric_dtype(df_kredivo[col])]]
    y = df_kredivo['late_payment']

    if X.empty or y.empty or y.nunique() < 2:
        st.error("Fitur atau target untuk model tidak valid setelah preprocessing. Tidak dapat melatih model.")
        st.stop()


    # Split data untuk training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Random Forest Classifier
    @st.cache_resource
    def train_random_forest_model(X_train_data, y_train_data):
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_data, y_train_data)
        return rf
    
    rf_model = train_random_forest_model(X_train, y_train)
    st.success("Model Random Forest berhasil dilatih!")

    st.subheader("Feature Importance")
    st.write("""
    *Feature importance* adalah metode dalam Machine Learning yang digunakan untuk menentukan fitur atau atribut mana yang paling berpengaruh dalam mempengaruhi target variabel (`late_payment`).
    """)
    st.write("""
    * **outstanding_amount_in**: Memberikan kontribusi terbesar, menunjukkan bahwa jumlah tunggakan memiliki hubungan yang sangat kuat terhadap keterlambatan pembayaran.
    * **outstanding_principal_in**: Berada di peringkat kedua, menunjukkan bahwa besar pinjaman awal juga sangat memengaruhi kemungkinan nasabah untuk terlambat.
    * **loan_type**: Jenis pinjaman memiliki pengaruh signifikan, mencerminkan bahwa tipe produk keuangan tertentu mungkin lebih berisiko dibanding yang lain.
    * Fitur lain (e.g., employment_type, gender, dan tenure(max dpd)): Meskipun memiliki pengaruh, dampaknya relatif lebih kecil dibandingkan *outstanding_amount_in* dan *outstanding_principal_in*.
    """)
    
    importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis', ax=ax_fi)
    ax_fi.set_title('Feature Importance')
    ax_fi.set_xlabel('Importance')
    ax_fi.set_ylabel('Feature')
    st.pyplot(fig_fi)


    st.subheader("Evaluasi Model: Confusion Matrix")
    st.write("""
    Confusion Matrix menunjukkan performa model dalam mengklasifikasikan nasabah yang terlambat dan tidak terlambat.
    """)
    y_pred_m = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_m)

    fig_cm_kredivo, ax_cm_kredivo = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Terlambat (0)', 'Terlambat (1)'],
                yticklabels=['Tidak Terlambat (0)', 'Terlambat (1)'],
                ax=ax_cm_kredivo)
    ax_cm_kredivo.set_xlabel('Prediksi')
    ax_cm_kredivo.set_ylabel('Aktual')
    ax_cm_kredivo.set_title('Confusion Matrix')
    st.pyplot(fig_cm_kredivo)

    st.write("#### Classification Report")
    report = classification_report(y_test, y_pred_m, target_names=['Tidak Terlambat', 'Terlambat'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())


    st.subheader("Kesimpulan & Saran")
    st.write("""
    * **Mayoritas Nasabah Berada di Outstanding Rendah:** Sebagian besar nasabah yang terlambat membayar lebih dari 1 tahun memiliki nilai *outstanding* rendah hingga sedang. Hal ini mungkin disebabkan oleh kemudahan akses pinjaman dengan limit kecil.
    * **Perlu Perhatian untuk Nasabah dengan Outstanding Tinggi:** Walaupun jumlahnya sedikit, nasabah dengan *outstanding* yang sangat tinggi (outlier) memiliki dampak signifikan terhadap total kerugian. Analisis lebih mendalam diperlukan untuk memahami profil nasabah ini.
    """)
    st.write("""
    Penelitian ini memberikan wawasan penting bagi PT Kredivo untuk memahami faktor-faktor utama yang memengaruhi keterlambatan pembayaran dan dapat digunakan untuk merancang strategi mitigasi risiko yang lebih efektif, seperti kebijakan kredit yang lebih selektif atau intervensi dini terhadap nasabah dengan profil berisiko tinggi.
    """)
    st.write("""
    Untuk pengembangan model prediktif, penelitian selanjutnya bisa menguji berbagai algoritma *machine learning* lain, selain Decision Tree Regression dan Random Forest, untuk membandingkan performanya. Misalnya, model-model seperti XGBoost, Gradient Boosting, atau Neural Networks bisa dieksplorasi untuk melihat apakah ada peningkatan akurasi atau kemampuan prediksi.
    """)
    st.write("---")
    st.markdown("[Lihat Kode Proyek di GitHub](https://github.com/Irkhamdwiramadhan)")

def page_analisis_ecommerce():
    display_section_header("📈 Analisis Data Transaksi E-commerce Toko Online")
    st.write("""
    Proyek ini berfokus pada analisis data transaksi e-commerce untuk mengungkap pola dan tren penting yang dapat mendorong keputusan bisnis.
    """)
    st.subheader("Tujuan Analisis")
    st.write("""
    Tujuan utama proyek ini meliputi:
    * Melihat tren transaksi dari bulan dan tahunnya.
    * Menganalisis pola transaksi harian dan bulanan.
    * Mengidentifikasi produk yang paling laris dan paling menguntungkan.
    * Menentukan produk/kategori mana yang memiliki tingkat pembatalan dan pengembalian paling mendominasi.
    * Menganalisis distribusi pelanggan berdasarkan sumber trafik dan demografi.
    """)
    st.subheader("Dataset")
    st.write("""
    Dataset yang digunakan adalah data transaksi E-commerce yang telah digabungkan dan dibersihkan, siap untuk analisis lebih lanjut.
    """)

    @st.cache_data
    def load_and_preprocess_ecommerce_data():
        try:
            st.write("--- Memuat Dataset Gabungan yang Sudah Dibersihkan ---")
            # Memuat file data_final.csv yang sudah digabung dan dibersihkan
            df_merged = pd.read_csv('data_final (1).csv')
            st.success(f"Dataset gabungan 'data_final.csv' berhasil dimuat! ({len(df_merged)} baris)")

            if df_merged.empty:
                st.error("Dataset 'data_final.csv' kosong setelah dimuat. Tidak dapat melanjutkan.")
                return pd.DataFrame()

            st.write("--- Menyiapkan Kolom Tanggal dan Profit ---")
            # Konversi kolom tanggal ke datetime (errors='coerce' akan mengubah yang tidak valid jadi NaT)
            df_merged['created_at_order'] = pd.to_datetime(df_merged['created_at_order'], errors='coerce')
            df_merged['returned_at'] = pd.to_datetime(df_merged['returned_at'], errors='coerce')
            df_merged['created_at_user'] = pd.to_datetime(df_merged['created_at_user'], errors='coerce')

            # Hitung profit
            df_merged['profit'] = df_merged['sale_price'] - df_merged['cost']

            # Buat kolom tahun, bulan, hari, hari dalam seminggu, minggu, awal/akhir bulan
            df_merged['year'] = df_merged['created_at_order'].dt.year
            df_merged['month'] = df_merged['created_at_order'].dt.month
            df_merged['day'] = df_merged['created_at_order'].dt.day
            df_merged['dayofweek'] = df_merged['created_at_order'].dt.dayofweek # 0 = Senin, 6 = Minggu
            df_merged['week'] = df_merged['created_at_order'].dt.isocalendar().week
            df_merged['is_start_month'] = df_merged['day'] <= 10
            df_merged['is_end_month'] = df_merged['day'] >= 21

            # Hapus baris dengan created_at_order yang null setelah konversi (karena tidak bisa dianalisis berdasarkan waktu)
            df_merged = df_merged.dropna(subset=['created_at_order']).copy()
            
            st.success(f"Dataset siap untuk analisis! ({len(df_merged)} baris akhir)")
            return df_merged

        except FileNotFoundError as e:
            st.error(f"Error: File 'data_final.csv' tidak ditemukan. Pastikan nama file dan ekstensi sudah benar serta file ada di direktori proyek Anda. Detail: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Terjadi error saat memuat atau menyiapkan data e-commerce: {e}")
            return pd.DataFrame()

    df_merged = load_and_preprocess_ecommerce_data()

    if df_merged.empty:
        st.stop()

    st.subheader("Data Gabungan (Preview)")
    st.dataframe(df_merged.head())
    st.write("Tipe Data Gabungan:")
    st.dataframe(df_merged.dtypes.reset_index().rename(columns={'index': 'Kolom', 0: 'Tipe Data'}))
    st.write("Jumlah nilai null per kolom:")
    st.dataframe(df_merged.isnull().sum())


    # --- ANALISIS UTAMA E-COMMERCE ---

    st.subheader("Tren Transaksi per Bulan dan Tahun")
    st.write("Heatmap ini menunjukkan jumlah transaksi (item pesanan) per bulan dan tahun, memberikan gambaran tentang pola penjualan musiman atau tren pertumbuhan.")
    
    # Agregasi data per bulan-tahun
    monthly_summary = df_merged.groupby(['year', 'month']).size().reset_index(name='jumlah_transaksi')
    
    # Pivot agar bisa divisualisasikan sebagai heatmap
    pivot_table = monthly_summary.pivot(index='month', columns='year', values='jumlah_transaksi')

    plt.figure(figsize=(10,6))
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=.5, linecolor='black')
    plt.title('Jumlah Transaksi per Bulan per Tahun')
    plt.ylabel('Bulan')
    plt.xlabel('Tahun')
    plt.yticks(ticks=np.arange(0, 12), labels=['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'], rotation=0)
    plt.tight_layout()
    st.pyplot(plt) # Menampilkan plot di Streamlit #


    st.subheader("Analisis Pola Transaksi Harian dan Bulanan")
    st.write("Menganalisis pola transaksi berdasarkan hari dalam sebulan dan lonjakan di awal/akhir bulan.")

    # Pola bulanan (Jumlah Pesanan per Bulan)
    orders_per_month = df_merged.groupby('month').size()
    fig_month, ax_month = plt.subplots(figsize=(10, 6))
    ax_month.bar(orders_per_month.index, orders_per_month.values, color='skyblue')
    ax_month.set_title('Jumlah Pesanan per Bulan')
    ax_month.set_xlabel('Bulan')
    ax_month.set_ylabel('Jumlah Pesanan')
    plt.xticks(orders_per_month.index, ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'], rotation=45)
    plt.tight_layout()
    st.pyplot(fig_month)

    # Pola harian (Jumlah Pesanan per Tanggal dalam Sebulan)
    orders_per_day = df_merged.groupby('day').size()
    fig_day, ax_day = plt.subplots(figsize=(10, 6))
    ax_day.plot(orders_per_day.index, orders_per_day.values, marker='o', color='blue')
    ax_day.set_title('Jumlah Pesanan per Tanggal dalam Sebulan')
    ax_day.set_xlabel('Tanggal')
    ax_day.set_ylabel('Jumlah Pesanan')
    ax_day.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_day)

    # Lonjakan awal/akhir bulan
    start_month_orders = df_merged[df_merged['is_start_month']].shape[0]
    end_month_orders = df_merged[df_merged['is_end_month']].shape[0]
    if start_month_orders > 0:
        percentage_change = ((end_month_orders - start_month_orders) / start_month_orders) * 100
        st.write(f"Lonjakan transaksi di akhir bulan dibandingkan awal bulan: **{percentage_change:.2f}%**")
    else:
        st.write("Data awal bulan tidak tersedia atau nol untuk menghitung lonjakan.")

    # Hari paling banyak transaksi
    orders_per_day_clean = orders_per_day.dropna()
    if not orders_per_day_clean.empty:
        peak_day = orders_per_day_clean.idxmax()
        peak_count = orders_per_day_clean.max()
        average_orders = orders_per_day_clean.mean()
        if average_orders > 0:
            percentage_spike = ((peak_count - average_orders) / average_orders) * 100
            st.write(f"Hari paling banyak transaksi: **Tanggal {int(peak_day)}** dengan **{int(peak_count)}** transaksi (lonjakan **{percentage_spike:.2f}%** dibanding rata-rata).")
        else:
            st.write("Rata-rata pesanan harian nol, tidak dapat menghitung lonjakan.")
    else:
        st.write("Data pesanan harian tidak tersedia.")


    st.subheader("Produk Terlaris vs Keuntungan Tertinggi")
    st.write("""
    Membandingkan kategori produk berdasarkan jumlah transaksi dan total profit untuk mengoptimalkan portofolio produk. Kesimpulan strategisnya adalah perlunya fokus pada profitabilitas di samping volume penjualan untuk mengoptimalkan portofolio produk dan alokasi sumber daya.
    """)
    
    # Filter hanya transaksi berhasil (Shipped atau Complete)
    df_delivered = df_merged[df_merged['status'].isin(['Shipped', 'Complete'])]

    col_sales, col_profit = st.columns(2)
    with col_sales:
        st.write("#### Kategori Produk Terlaris (Jumlah Transaksi)")
        # Hitung jumlah transaksi sukses per kategori
        category_sales = df_delivered['category'].value_counts().head(10)

        fig_sales, ax_sales = plt.subplots(figsize=(10, 7))
        # Pastikan data tidak kosong sebelum plotting
        if not category_sales.empty:
            bars_sales = sns.barplot(x=category_sales.values, y=category_sales.index, palette="Blues_d", hue=category_sales.index, legend=False, ax=ax_sales)
            ax_sales.set_title("Kategori Produk Terlaris (Jumlah Transaksi)")
            ax_sales.set_xlabel("Jumlah Transaksi")
            ax_sales.set_ylabel("Kategori Produk")
            for bar in bars_sales.containers:
                # KOREKSI: fmt untuk angka integer dengan koma ribuan
                ax_sales.bar_label(bar, fmt='{:,}', label_type='edge', padding=3, color='white', fontsize=10) # Dulu: '%.0f'
            plt.tight_layout()
            st.pyplot(fig_sales)
        else:
            st.info("Tidak ada data penjualan untuk kategori produk terlaris.")

    with col_profit:
        st.write("#### Top 10 Kategori Produk dengan Keuntungan Tertinggi")
        # Hitung total profit per kategori
        category_profit = df_delivered.groupby('category')['profit'].sum().sort_values(ascending=False).head(10)

        fig_profit, ax_profit = plt.subplots(figsize=(10, 7))
        if not category_profit.empty:
            bars_profit = sns.barplot(x=category_profit.values, y=category_profit.index, palette="Greens_d", hue=category_profit.index, legend=False, ax=ax_profit)
            ax_profit.set_title("Kategori Produk dengan Profit Tertinggi")
            ax_profit.set_xlabel("Total Profit")
            ax_profit.set_ylabel("Kategori Produk")
            for bar in bars_profit.containers:
                # KOREKSI: fmt untuk float dengan koma ribuan dan 0 desimal
                ax_profit.bar_label(bar, fmt='Rp {:,.0f}', label_type='edge', padding=3, color='white', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_profit)
        else:
            st.info("Tidak ada data profit untuk kategori produk terlaris.")


    st.subheader("Analisis Pembatalan dan Pengembalian")
    st.write("Menganalisis tren persentase transaksi yang dibatalkan dan item yang dikembalikan per bulan.")
    
    # Ekstrak Tahun dan Bulan untuk analisis tren
    df_analysis_cr = df_merged.dropna(subset=['created_at_order']).copy()
    df_analysis_cr['order_year_month'] = df_analysis_cr['created_at_order'].dt.to_period('M')

    # Agregasi Data per Bulan
    monthly_trends_cr = df_analysis_cr.groupby('order_year_month').agg(
        total_transactions=('order_id', 'count'), # Menghitung jumlah order item
        cancelled_transactions=('status', lambda x: (x == 'Cancelled').sum()),
        returned_items=('returned_at', lambda x: x.notna().sum())
    ).reset_index()

    # Konversi order_year_month kembali ke datetime untuk plotting yang lebih baik
    monthly_trends_cr['order_year_month'] = monthly_trends_cr['order_year_month'].dt.to_timestamp()

    # Hitung persentase
    monthly_trends_cr['percentage_cancelled'] = (monthly_trends_cr['cancelled_transactions'] / monthly_trends_cr['total_transactions']) * 100
    monthly_trends_cr['percentage_returned'] = (monthly_trends_cr['returned_items'] / monthly_trends_cr['total_transactions']) * 100

    # Urutkan berdasarkan waktu
    monthly_trends_cr = monthly_trends_cr.sort_values(by='order_year_month')

    # Visualisasi Tren
    fig_trends, ax_trends = plt.subplots(figsize=(14, 7))
    if not monthly_trends_cr.empty:
        sns.lineplot(x='order_year_month', y='percentage_cancelled', data=monthly_trends_cr, marker='o', label='Persentase Dibatalkan', ax=ax_trends)
        sns.lineplot(x='order_year_month', y='percentage_returned', data=monthly_trends_cr, marker='o', label='Persentase Dikembalikan', ax=ax_trends)

        ax_trends.set_title('Tren Persentase Transaksi Dibatalkan dan Item Dikembalikan per Bulan')
        ax_trends.set_xlabel('Bulan')
        ax_trends.set_ylabel('Persentase (%)')
        ax_trends.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        ax_trends.legend()
        plt.tight_layout()
        st.pyplot(fig_trends)
    else:
        st.info("Tidak ada data tren untuk analisis pembatalan dan pengembalian.")

    st.subheader("Analisis Demografi Pengguna")
    st.write("Analisis demografi pengguna berdasarkan usia, jenis kelamin, dan negara.")

    # Usia pelanggan termuda dan tertua
    if not df_merged.empty and 'age' in df_merged.columns:
        min_age = df_merged['age'].min()
        max_age = df_merged['age'].max()
        st.write(f"Usia pelanggan termuda: **{min_age} tahun**.")
        st.write(f"Usia pelanggan tertua: **{max_age} tahun**.")

        youngest_customers = df_merged[df_merged['age'] == min_age]
        youngest_countries = youngest_customers['country'].unique()
        youngest_users_count = youngest_customers['user_id'].nunique()
        st.write(f"Jumlah pengguna dengan usia {min_age} tahun: **{youngest_users_count} orang**.")
        st.write(f"Mereka berasal dari negara: **{', '.join(youngest_countries)}**.")

        oldest_customers = df_merged[df_merged['age'] == max_age]
        oldest_countries = oldest_customers['country'].unique()
        oldest_users_count = oldest_customers['user_id'].nunique()
        st.write(f"Jumlah pengguna dengan usia {max_age} tahun: **{oldest_users_count} orang**.")
        st.write(f"Mereka berasal dari negara: **{', '.join(oldest_countries)}**.")
    else:
        st.info("Data usia pengguna tidak tersedia untuk analisis demografi.")

    st.write("---")
    st.markdown("[Lihat Kode Proyek di GitHub](https://github.com/yourgithubusername/nama_repo_ecommerce_analysis)")

def page_prediksi_pemain_bola():
    display_section_header("⚽ Prediksi Nilai Pasar Pemain Bola Berdasarkan Data Transfer")
    st.write("""
    Proyek ini bertujuan untuk memprediksi nilai pasar pemain sepak bola berdasarkan data transfer menggunakan algoritma klasifikasi Decision Tree.
    """)
    st.subheader("Permasalahan & Tujuan")
    st.write("""
    Menganalisis dan memprediksi nilai pasar pemain bola, yang merupakan estimasi nilai pasar seorang pemain berdasarkan kinerja, potensi, umur, dan faktor-faktor lain.
    """)
    st.subheader("Dataset")
    st.write("""
    Dataset transfer pemain sepak bola dari Kaggle, mencakup 13.290 baris data dan 10 kolom awal yang berisi tanggal transfer, klub tujuan dan asal, nilai transfer, dan nilai pasar pemain. Disebutkan juga dataset ini mencakup 76 ribu baris data.
    """)

    # --- DATA LOADING AND PREPROCESSING ---
    @st.cache_data
    def load_and_preprocess_player_data():
        df = pd.DataFrame()
        le_market_value = LabelEncoder()

        try:
            df = pd.read_csv('transfers.csv')
            st.success(f"Dataset Pemain Bola berhasil dimuat! ({len(df)} baris)")
        except FileNotFoundError:
            st.error("Error: 'transfers.csv' tidak ditemukan. Pastikan file ada di direktori yang sama dengan app.py/utils.py.")
            return pd.DataFrame(), LabelEncoder()
        except Exception as e:
            st.error(f"Error saat memuat 'transfers.csv': {e}")
            return pd.DataFrame(), LabelEncoder()
        
        valid_seasons = ['23/24', '24/25', '25/26']
        df = df[df['transfer_season'].isin(valid_seasons)].copy()
        
        if df.empty:
            st.warning("Dataset kosong setelah filtering berdasarkan transfer_season. Tidak dapat melanjutkan.")
            return pd.DataFrame(), LabelEncoder()

        # --- Fungsi Kustom untuk Mengkonversi String Angka dengan 'K' atau 'M' ---
        def convert_currency_string_to_numeric(s):
            if pd.isna(s): # Handle NaN values
                return np.nan
            s = str(s).replace('€', '').strip()
            if 'K' in s:
                return float(s.replace('K', '')) * 1_000
            elif 'M' in s:
                return float(s.replace('M', '')) * 1_000_000
            else:
                return pd.to_numeric(s, errors='coerce') # Handle regular numbers or other unparseable strings

        # --- TERUBAH: Menggunakan fungsi kustom untuk transfer_fee dan market_value_in_eur ---
        df['transfer_fee'] = df['transfer_fee'].apply(convert_currency_string_to_numeric).fillna(0)
        df['market_value_in_eur'] = df['market_value_in_eur'].apply(convert_currency_string_to_numeric)
        
        df = df.dropna(subset=['market_value_in_eur'])

        if df.empty:
            st.warning("Dataset kosong setelah menghapus baris dengan NaN di market_value_in_eur. Tidak dapat melanjutkan.")
            return pd.DataFrame(), LabelEncoder()

        big_clubs = [
            "Real Madrid", "Barcelona", "Manchester United", "Liverpool", "Bayern Munich",
            "Paris Saint-Germain", "Juventus", "Chelsea", "Arsenal", "Inter Milan",
            "AC Milan", "Borussia Dortmund", "Manchester City", "Atletico Madrid",
            "Tottenham Hotspur", "RB Leipzig", "Roma", "Sevilla FC", "Ajax Amsterdam",
            "Olympique Lyon", "Napoli", "Porto", "Benfica", "Villarreal", "Southampton",
            "Everton", "Leeds United", "West Ham United", "Wolves", "Celtic", "Galatasaray",
            "Fenerbahçe", "Beşiktaş", "Valencia", "Marseille", "Boca Juniors", "River Plate",
            "Corinthians", "Sao Paulo FC", "Flamengo", "Fluminense", "Santos FC", "LA Galaxy",
            "Toronto FC", "New York City FC", "Vancouver Whitecaps", "Dynamo Kyiv", "Shakhtar Donetsk",
            "Zenit Saint Petersburg", "Club América", "Chivas Guadalajara", "Monterrey", "America de Cali",
            "Independiente", "Atlético Nacional", "Rangers FC", "Wanderers", "Olympique de Marseille","Lazio"
        ]
        medium_clubs = [
            "Real Sociedad", "Fiorentina", "Schalke 04", "Leicester City", "Brighton & Hove Albion",
            "West Bromwich Albion", "Cagliari", "Crystal Palace", "Betis", "Atalanta", "Sampdoria",
            "Brondby IF", "Club Brugge", "Anderlecht", "Red Bull Salzburg", "Club Tijuana", "Rayo Vallecano",
            "Eintracht Frankfurt", "Sampdoria", "Maritimo", "Akhisar Belediyespor", "Legia Warsaw",
            "Spartak Moscow", "Hellas Verona", "Bayer Leverkusen", "Real Betis", "Vitesse", "Swansea City",
            "Sheffield United", "Alaves", "Espanyol", "Osasuna", "Almeria", "Granada CF", "Eibar", "Tenerife",
            "Bordeaux", "Lille OSC", "Aston Villa", "Fiorentina", "Espanyol", "Torino", "Krasnodar", "Dinamo Zagreb",
            "Zorya Luhansk", "Lech Poznan", "Girondins Bordeaux", "SC Freiburg", "Villarreal", "Vélez Sarsfield",
            "Pumas UNAM", "Atlético Mineiro", "Real Valladolid", "Vitoria Guimaraes", "Brescia", "Osasuna",
            "Getafe", "Levante", "Omonia Nicosia", "Nacional", "Tondela", "Famalicão", "Rayo Vallecano",
            "Deportivo La Coruña", "Huesca"
        ]
        def categorize_club(club_name):
            if club_name in big_clubs:
                return 'Big Club'
            elif club_name in medium_clubs:
                return 'Medium Club'
            else:
                return 'Small Club'
        
        df['from_club_category'] = df['from_club_name'].apply(categorize_club)
        df['to_club_category'] = df['to_club_name'].apply(categorize_club)
        
        df['from_club_category'] = df['from_club_category'].astype('category')
        df['to_club_category'] = df['to_club_category'].astype('category')

        # Perbaikan pd.qcut: Pastikan ada nilai non-NaN yang cukup dan unik
        if df['market_value_in_eur'].dropna().nunique() < 3:
            st.warning("Kolom 'market_value_in_eur' memiliki kurang dari 3 nilai unik non-NaN. Tidak dapat membuat kategori nilai pasar yang meaningful. Menggunakan kategori default (low).")
            df.loc[:, 'market_value_category'] = 'low'
        else:
            try:
                df.loc[:, 'market_value_category'] = pd.qcut(df['market_value_in_eur'], q=3, labels=['low', 'medium', 'high'])
            except Exception as e:
                st.warning(f"Tidak dapat membuat kategori nilai pasar menggunakan qcut: {e}. Menggunakan kategori default (low).")
                df.loc[:, 'market_value_category'] = 'low'


        df['transfer_date'] = pd.to_datetime(df['transfer_date'], errors='coerce')

        le_from = LabelEncoder()
        df.loc[:, 'from_club_category'] = le_from.fit_transform(df['from_club_category'])
        le_to = LabelEncoder()
        df.loc[:, 'to_club_category'] = le_to.fit_transform(df['to_club_category'])
        
        le_market_value = LabelEncoder()
        if df['market_value_category'].nunique() > 1:
            df.loc[:, 'market_value_category'] = le_market_value.fit_transform(df['market_value_category'])
        else:
            df.loc[:, 'market_value_category'] = 0
            le_market_value.fit(df['market_value_category'].unique())


        return df, le_market_value

    df_pemain_bola, le_market_value_player = load_and_preprocess_player_data()

    if df_pemain_bola.empty:
        st.error("Gagal memuat atau memproses data pemain bola. Halaman ini mungkin tidak berfungsi dengan benar.")
        st.stop()

    st.subheader("Contoh Data Setelah Preprocessing")
    st.dataframe(df_pemain_bola.head())

    st.subheader("Metodologi Penelitian")
    st.write("""
    Proyek ini mengikuti tahapan standar dalam Data Mining:
    1.  **Pengumpulan Data**: Data diperoleh dari platform Kaggle, yang menyediakan dataset terkait transfer pemain sepak bola.
    2.  **Preprocessing Data**: Dilakukan pembersihan nilai kosong dan transformasi data (mengkategorikan kolom, merubah tipe data) untuk memastikan dataset siap untuk analisis dan membangun model klasifikasi.
    3.  **Pembangunan Model Decision Tree**: Menggunakan algoritma Decision Tree untuk memprediksi kelas tertentu.
    4.  **Evaluasi Model**: Menilai performa model berdasarkan tingkat ketepatan dan laporan klasifikasi.
    """)

    st.subheader("Heatmap Korelasi")
    st.write("Heatmap korelasi menunjukkan hubungan antara variabel-variabel numerik dalam dataset.")
    
    df_numeric_player = df_pemain_bola.select_dtypes(include=['number'])
    corr_matrix_player = df_numeric_player.corr()

    fig_corr_player, ax_corr_player = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix_player, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr_player)
    ax_corr_player.set_title('Correlation Heatmap')
    st.pyplot(fig_corr_player)

    st.subheader("Visualisasi Pohon Keputusan")
    st.write("""
    Gambar pohon keputusan ini menunjukkan bagaimana model mengambil keputusan berdasarkan data yang dimilikinya untuk memprediksi kelas tertentu. Pohon dimulai dari bagian atas (root node), di mana model memeriksa fitur `from_club_category` untuk memutuskan apakah data masuk ke cabang kiri atau kanan. Fitur ini dipilih karena paling membantu dalam memisahkan data. Di setiap titik, model mencoba memisahkan data menjadi kelompok-kelompok yang lebih kecil dan lebih seragam, hingga sampai ke ujung pohon (leaf nodes). Ujung pohon ini menunjukkan prediksi akhir model, berdasarkan kelompok data yang tersisa.
    """)
    
    model_features_player = ['transfer_fee', 'from_club_category', 'to_club_category']
    X_player = df_pemain_bola[[col for col in model_features_player if col in df_pemain_bola.columns and pd.api.types.is_numeric_dtype(df_pemain_bola[col])]]
    y_player = df_pemain_bola['market_value_category']

    if X_player.empty or y_player.empty or y_player.nunique() < 2:
        st.error("Fitur atau target untuk model tidak valid setelah preprocessing. Tidak dapat melatih model Decision Tree.")
        st.stop()


    X_train_player, X_test_player, y_train_player, y_test_player = train_test_split(X_player, y_player, test_size=0.2, random_state=42)
    
    @st.cache_resource
    def train_decision_tree_model(X_train_data, y_train_data):
        clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42, ccp_alpha=0.001)
        clf.fit(X_train_data, y_train_data)
        return clf

    clf_player = train_decision_tree_model(X_train_player, y_train_player)
    st.success("Model Decision Tree berhasil dilatih!")
    
    fig_dt, ax_dt = plt.subplots(figsize=(15, 10))
    class_names_dt = [str(label) for label in sorted(y_player.unique())]
    plot_tree(clf_player, feature_names=X_player.columns, class_names=class_names_dt, filled=True, ax=ax_dt)
    ax_dt.set_title("Decision Tree Visualization")
    st.pyplot(fig_dt)

    st.subheader("Evaluasi Model")
    st.write("""
    Hasil evaluasi model menunjukkan bahwa tingkat ketepatan keseluruhan model adalah 55%. Ini berarti model mampu membuat prediksi yang benar sebanyak 55% dari semua data yang diuji, sementara 45% sisanya merupakan prediksi yang salah. Dalam laporan klasifikasi, terlihat bahwa kemampuan model berbeda-beda untuk setiap kelas. Misalnya, untuk kelas 0, model cukup baik dalam mengidentifikasi data yang benar-benar termasuk ke kelas tersebut, dengan tingkat akurasi prediksi mendekati 70%. Namun, untuk kelas 2, model memiliki performa yang buruk dengan nilai pengukuran rendah, menunjukkan bahwa model sering salah dalam memprediksi data untuk kelas ini.
    """)
    
    y_pred_player = clf_player.predict(X_test_player)
    
    st.write("#### Accuracy:")
    st.write(f"{accuracy_score(y_test_player, y_pred_player):.2f}")

    st.write("#### Classification Report:")
    target_names_cr = [str(label) for label in sorted(y_player.unique())]
    st.dataframe(pd.DataFrame(classification_report(y_test_player, y_pred_player, target_names=target_names_cr, output_dict=True)).transpose())

    st.write("#### Confusion Matrix:")
    cm_player = confusion_matrix(y_test_player, y_pred_player)
    fig_cm_player, ax_cm_player = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_player, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names_cr,
                yticklabels=target_names_cr,
                ax=ax_cm_player)
    ax_cm_player.set_xlabel('Prediksi')
    ax_cm_player.set_ylabel('Aktual')
    ax_cm_player.set_title('Confusion Matrix')
    st.pyplot(fig_cm_player)

    st.write("---")
    st.markdown("[Lihat Kode Proyek di GitHub](https://github.com/Irkhamdwiramadhan)")
# utils.py
# ... (Pastikan semua import di bagian atas file sudah ada) ...

def page_analisis_perceraian():
    display_section_header("📊 Analisis Tren Perceraian di Indonesia (2018 - 2024)")
    st.write("""
    Proyek ini bertujuan untuk menganalisis tren perceraian di Indonesia dari tahun 2018 hingga 2024 berdasarkan data dari Badan Pusat Statistik (BPS). Analisis ini mencakup perbandingan tren pernikahan dan perceraian, serta perbandingan faktor perceraian (Cerai Talak vs Cerai Gugat) secara nasional dan per provinsi.
    """)
    st.subheader("Dataset")
    st.write("""
    Dataset diperoleh dari Badan Pusat Statistik (BPS) Indonesia, yang merupakan gabungan dari data jumlah perceraian berdasarkan provinsi dan faktor penyebabnya, serta data jumlah pernikahan dan perceraian menurut provinsi per tahun. Data mencakup periode 2018 hingga 2024.
    """)

    @st.cache_data
    def load_and_preprocess_divorce_data():
        jumlah = pd.DataFrame()
        nikah_cerai = pd.DataFrame()
        
        try:
            for tahun in range(2018, 2025):
                file_jumlah = f'Jumlah Perceraian Menurut Provinsi dan Faktor, {tahun}.csv'
                try:
                    df_jumlah = pd.read_csv(file_jumlah)
                except Exception:
                    df_jumlah = pd.read_csv(file_jumlah, sep=';')
                
                df_jumlah['Tahun'] = tahun
                jumlah = pd.concat([jumlah, df_jumlah], ignore_index=True)
            st.success("Dataset 'Jumlah Perceraian Menurut Provinsi dan Faktor' berhasil dimuat.")
        except FileNotFoundError:
            st.error(f"Error: Salah satu file 'Jumlah Perceraian Menurut Provinsi dan Faktor, <tahun>.csv' tidak ditemukan. Pastikan semua file 2018-2024 ada.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error saat memuat atau menggabungkan data 'Jumlah Perceraian Menurut Provinsi dan Faktor': {e}")
            return pd.DataFrame()
        

        try:
            for tahun in range(2018, 2025):
                file_nikah_cerai = f'Nikah dan Cerai Menurut Provinsi, {tahun}.csv'
                try:
                    df_nikah_cerai = pd.read_csv(file_nikah_cerai)
                except Exception:
                    df_nikah_cerai = pd.read_csv(file_nikah_cerai, sep=';')
                
                df_nikah_cerai['Tahun'] = tahun
                nikah_cerai = pd.concat([nikah_cerai, df_nikah_cerai], ignore_index=True)
            st.success("Dataset 'Nikah dan Cerai Menurut Provinsi' berhasil dimuat.")
        except FileNotFoundError:
            st.error(f"Error: Salah satu file 'Nikah dan Cerai Menurut Provinsi, <tahun>.csv' tidak ditemukan. Pastikan semua file 2018-2024 ada.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error saat memuat atau menggabungkan data 'Nikah dan Cerai Menurut Provinsi': {e}")
            return pd.DataFrame()


        # --- Data Cleansing (jumlah DataFrame) ---
        jumlah.columns = jumlah.columns.str.replace('Fakor Perceraian - ', '', regex=False)

        factor_columns = [col for col in jumlah.columns if col not in ['Provinsi', 'Tahun']]

        for col in factor_columns:
            if col in jumlah.columns:
                jumlah[col] = jumlah[col].astype(str).str.replace('...', '0', regex=False)
                jumlah[col] = pd.to_numeric(jumlah[col], errors='coerce').fillna(0).astype('Int64')
        
        jumlah = jumlah[jumlah['Provinsi'] != 'Indonesia'].copy()

        # --- Data Cleansing (nikah_cerai DataFrame) ---
        nikah_cerai_numeric_cols = ['Nikah', 'Cerai Talak', 'Cerai Gugat', 'Jumlah Cerai']
        for col in nikah_cerai_numeric_cols:
            if col in nikah_cerai.columns:
                nikah_cerai[col] = nikah_cerai[col].astype(str).str.replace('...', '0', regex=False)
                nikah_cerai[col] = pd.to_numeric(nikah_cerai[col], errors='coerce').fillna(0).astype('Int64')
        
        nikah_cerai = nikah_cerai[nikah_cerai['Provinsi'] != 'Indonesia'].copy()

        # --- Menggabungkan Dataframe ---
        jumlah = jumlah.reset_index(drop=True)
        nikah_cerai = nikah_cerai.reset_index(drop=True)
        
        df_merged = pd.merge(jumlah, nikah_cerai, on=['Provinsi', 'Tahun'], how='outer')
        
        for col in df_merged.columns:
            if pd.api.types.is_numeric_dtype(df_merged[col]):
                df_merged[col] = df_merged[col].fillna(0)
            
        df_merged.drop(columns=['level_0_x', 'index_x', 'level_0_y', 'index_y'], errors='ignore', inplace=True)

        return df_merged

    df_perceraian = load_and_preprocess_divorce_data()

    if df_perceraian.empty:
        st.stop()

    st.subheader("Contoh Data Setelah Preprocessing")
    st.dataframe(df_perceraian.head())
    st.dataframe(df_perceraian.isnull().sum())
    st.write(f"Jumlah duplikasi: {df_perceraian.duplicated().sum()}")
    st.dataframe(df_perceraian.describe())
    st.write(df_perceraian.info())


    # --- EDA VISUALIZATIONS ---
    st.subheader("Tren Jumlah Perceraian Nasional per Tahun")
    st.write("Tren jumlah perceraian (Jumlah Cerai) dari tahun ke tahun secara nasional.")
    
    trend_total_cerai = df_perceraian.groupby('Tahun')['Jumlah Cerai'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=trend_total_cerai, x='Tahun', y='Jumlah Cerai', marker='o', color='steelblue', alpha=0.8, ax=ax)
    ax.set_title('Tren Jumlah Perceraian Secara Nasional per Tahun')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Jumlah Perceraian')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Tren & Persentase Jumlah Perceraian Nasional per Tahun (Bar Chart)")
    st.write("Visualisasi *bar chart* tren jumlah perceraian nasional per tahun, dilengkapi dengan persentase pertumbuhan tahunan.")

    trend_total_cerai_bar = df_perceraian.groupby('Tahun')['Jumlah Cerai'].sum().reset_index()
    trend_total_cerai_bar['Growth %'] = trend_total_cerai_bar['Jumlah Cerai'].pct_change() * 100

    fig_bar_trend, ax_bar_trend = plt.subplots(figsize=(8, 5))
    barplot = sns.barplot(data=trend_total_cerai_bar, x='Tahun', y='Jumlah Cerai', color='steelblue', ax=ax_bar_trend)
    
    ax_bar_trend.spines['top'].set_visible(False)
    ax_bar_trend.spines['right'].set_visible(False)
    ax_bar_trend.spines['left'].set_visible(True)
    ax_bar_trend.spines['bottom'].set_visible(True)

    for index, row in trend_total_cerai_bar.iterrows():
        jumlah = row['Jumlah Cerai']
        growth = row['Growth %']
        label = f"{jumlah:,}"
        if index > 0 and pd.notnull(growth):
            growth_label = f"\n({growth:+.1f}%)"
            label += growth_label
        ax_bar_trend.text(index, jumlah + max(trend_total_cerai_bar['Jumlah Cerai']) * 0.01,
                     label, ha='center', va='bottom', fontsize=9, color='crimson')

    ax_bar_trend.set_title('Tren Jumlah Perceraian Nasional per Tahun', fontsize=14, weight='bold', pad=20)
    ax_bar_trend.set_xlabel('Tahun')
    ax_bar_trend.set_ylabel('Jumlah Perceraian')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bar_trend)

    st.subheader("Jumlah Perceraian per Provinsi per Tahun (Interaktif)")
    st.write("Gunakan dropdown di bawah untuk melihat tren jumlah perceraian di setiap provinsi.")

    provinsi_list = ['Semua Provinsi'] + sorted(df_perceraian['Provinsi'].unique().tolist())
    
    selected_provinsi = st.selectbox("Pilih Provinsi:", options=provinsi_list, key="perceraian_provinsi_dropdown")

    if selected_provinsi == 'Semua Provinsi':
        data_prov = df_perceraian.groupby('Tahun')['Jumlah Cerai'].sum().reset_index()
        title_prov = 'Tren Jumlah Perceraian di Semua Provinsi'
    else:
        data_prov = df_perceraian[df_perceraian['Provinsi'] == selected_provinsi].groupby('Tahun')['Jumlah Cerai'].sum().reset_index()
        title_prov = f'Tren Jumlah Perceraian di {selected_provinsi}'

    data_prov['Growth %'] = data_prov['Jumlah Cerai'].pct_change() * 100

    fig_prov, ax_prov = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=data_prov, x='Tahun', y='Jumlah Cerai', marker='o', color='steelblue', alpha=0.8, ax=ax_prov)
    
    ax_prov.spines['top'].set_visible(False)
    ax_prov.spines['right'].set_visible(False)
    ax_prov.spines['left'].set_visible(True)
    ax_prov.spines['bottom'].set_visible(True)

    for i, row in data_prov.iterrows():
        jumlah = row['Jumlah Cerai']
        growth = row['Growth %']
        label = f"{jumlah:,}"
        if i > 0 and pd.notnull(growth):
            growth_label = f"\n({growth:+.1f}%)"
            label += growth_label
        ax_prov.text(row['Tahun'], jumlah + max(data_prov['Jumlah Cerai']) * 0.01,
                     label, ha='center', va='bottom', fontsize=9, color='crimson')

    ax_prov.set_title(title_prov, fontsize=14, weight='bold', pad=20)
    ax_prov.set_xlabel('Tahun')
    ax_prov.set_ylabel('Jumlah Perceraian')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_prov)


    st.subheader("Perbandingan Cerai Talak vs Cerai Gugat per Tahun")
    st.write("""
    Sederhananya jika istri merupakan pihak yang mengajukan gugatan untuk menceraikan suami, prosesnya dikenal sebagai cerai gugat. Sebaliknya, jika suami merupakan pihak yang mengajukan gugatan untuk menceraikan istri, prosesnya dikenal sebagai cerai talak.
    """)

    cerai_per_jenis = df_perceraian.groupby('Tahun')[['Cerai Talak', 'Cerai Gugat']].sum().reset_index()
    cerai_per_jenis['Talak Growth'] = cerai_per_jenis['Cerai Talak'].pct_change() * 100
    cerai_per_jenis['Gugat Growth'] = cerai_per_jenis['Cerai Gugat'].pct_change() * 100


    fig_talak_gugat, ax_talak_gugat = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=cerai_per_jenis, x='Tahun', y='Cerai Talak', marker='o', label='Cerai Talak (oleh Suami)', color='steelblue', ax=ax_talak_gugat)
    sns.lineplot(data=cerai_per_jenis, x='Tahun', y='Cerai Gugat', marker='o', label='Cerai Gugat (oleh Istri)', color='darkorange', ax=ax_talak_gugat)

    ax_talak_gugat.spines['top'].set_visible(False)
    ax_talak_gugat.spines['right'].set_visible(False)
    ax_talak_gugat.spines['left'].set_visible(True)
    ax_talak_gugat.spines['bottom'].set_visible(True)

    for i in range(len(cerai_per_jenis)):
        tahun = cerai_per_jenis['Tahun'][i]
        talak = cerai_per_jenis['Cerai Talak'][i]
        gugat = cerai_per_jenis['Cerai Gugat'][i]
        talak_growth = cerai_per_jenis['Talak Growth'][i]
        gugat_growth = cerai_per_jenis['Gugat Growth'][i]

        label_talak = f"{talak:,}"
        if not pd.isna(talak_growth):
            label_talak += f"\n({talak_growth:+.1f}%)"
        ax_talak_gugat.text(tahun, talak + max(cerai_per_jenis['Cerai Talak']) * 0.02,
                             label_talak, ha='center', va='bottom', fontsize=8, color='steelblue')

        label_gugat = f"{gugat:,}"
        if not pd.isna(gugat_growth):
            label_gugat += f"\n({gugat_growth:+.1f}%)"
        ax_talak_gugat.text(tahun, gugat - max(cerai_per_jenis['Cerai Gugat']) * 0.08, # Disesuaikan posisi vertikal
                             label_gugat, ha='center', va='top', fontsize=8, color='darkorange')

    ax_talak_gugat.set_title('Tren & Persentase Cerai Talak vs Cerai Gugat per Tahun', fontsize=14, weight='bold', pad=20)
    ax_talak_gugat.set_xlabel('Tahun')
    ax_talak_gugat.set_ylabel('Jumlah Kasus')
    plt.xticks(rotation=45)
    ax_talak_gugat.legend()
    plt.tight_layout()
    st.pyplot(fig_talak_gugat)

    st.subheader("Perbandingan Jumlah Pernikahan dan Perceraian per Tahun")
    st.write("Visualisasi ini menunjukkan perbandingan jumlah pernikahan dan perceraian secara nasional dari tahun 2018 hingga 2024.")

    # Agregasi nasional per tahun
    # Asumsikan df_perceraian sudah dimuat dan memiliki kolom: 'Tahun', 'Nikah', 'Jumlah Cerai'
    perbandingan_df = df_perceraian.groupby('Tahun')[['Nikah', 'Jumlah Cerai']].sum().reset_index()

    fig_nikah_cerai_tahunan, ax_nikah_cerai_tahunan = plt.subplots(figsize=(12, 6)) #
    sns.set(style='white') #

    x = np.arange(len(perbandingan_df['Tahun'])) #
    bar_width = 0.4 #

    # Buat bar chart
    ax_nikah_cerai_tahunan.bar(x - bar_width/2, perbandingan_df['Nikah'], width=bar_width, label='Jumlah Pernikahan', color='green')
    ax_nikah_cerai_tahunan.bar(x + bar_width/2, perbandingan_df['Jumlah Cerai'], width=bar_width, label='Jumlah Perceraian', color='gold')

    # Tambahkan label pada setiap bar
    for i in range(len(x)):
        nikah_val = perbandingan_df['Nikah'][i]
        cerai_val = perbandingan_df['Jumlah Cerai'][i]

        # Pastikan nilai numerik, jika bukan, lewati atau konversi
        if pd.isna(nikah_val): nikah_val = 0
        if pd.isna(cerai_val): cerai_val = 0
        
        ax_nikah_cerai_tahunan.text(x[i] - bar_width/2, nikah_val + max(perbandingan_df['Nikah'].max(),0)*0.01,
                                 f"{int(nikah_val):,}", ha='center', va='bottom', fontsize=8, color='green')
        ax_nikah_cerai_tahunan.text(x[i] + bar_width/2, cerai_val + max(perbandingan_df['Jumlah Cerai'].max(),0)*0.01,
                                 f"{int(cerai_val):,}", ha='center', va='bottom', fontsize=8, color='gold')

    # Judul dan tampilan
    ax_nikah_cerai_tahunan.set_xticks(x)
    ax_nikah_cerai_tahunan.set_xticklabels(perbandingan_df['Tahun'].astype(str), rotation=45)
    ax_nikah_cerai_tahunan.set_title('Perbandingan Jumlah Pernikahan dan Perceraian per Tahun', fontsize=14, weight='bold')
    ax_nikah_cerai_tahunan.set_xlabel('Tahun')
    ax_nikah_cerai_tahunan.set_ylabel('Jumlah Kasus')
    ax_nikah_cerai_tahunan.legend()
    plt.tight_layout()
    st.pyplot(fig_nikah_cerai_tahunan) # Menampilkan plot di Streamlit

    st.subheader("Perbandingan Pernikahan vs. Perceraian per Provinsi (Interaktif)")
    st.write("Gunakan dropdown di bawah untuk melihat perbandingan jumlah pernikahan dan perceraian di setiap provinsi dari tahun 2018 hingga 2024.")

    # Daftar unik provinsi (dari df_perceraian yang sudah dimuat)
    daftar_provinsi = sorted(df_perceraian['Provinsi'].unique().tolist()) #

    # Dropdown interaktif Streamlit
    selected_provinsi_nikah_cerai = st.selectbox(
        "Pilih Provinsi untuk Perbandingan Pernikahan vs. Perceraian:",
        options=daftar_provinsi,
        key="perbandingan_nikah_cerai_provinsi_dropdown"
    )

    if selected_provinsi_nikah_cerai:
        # Filter data sesuai provinsi yang dipilih
        data_perbandingan = df_perceraian[df_perceraian['Provinsi'] == selected_provinsi_nikah_cerai] #

        # Agregasi per tahun
        perbandingan_df = data_perbandingan.groupby('Tahun')[['Nikah', 'Jumlah Cerai']].sum().reset_index() #

        fig_perbandingan, ax_perbandingan = plt.subplots(figsize=(9, 6)) #
        sns.set(style='white') #

        x = np.arange(len(perbandingan_df['Tahun'])) #
        bar_width = 0.4 #

        # Buat bar chart
        ax_perbandingan.bar(x - bar_width/2, perbandingan_df['Nikah'], width=bar_width, label='Jumlah Pernikahan', color='green') #
        ax_perbandingan.bar(x + bar_width/2, perbandingan_df['Jumlah Cerai'], width=bar_width, label='Jumlah Perceraian', color='gold') #

        # Label di atas bar
        for i in range(len(x)): #
            nikah_val = perbandingan_df['Nikah'][i]
            cerai_val = perbandingan_df['Jumlah Cerai'][i]
            
            # Pastikan nilai numerik, jika bukan, lewati atau konversi
            if pd.isna(nikah_val): nikah_val = 0
            if pd.isna(cerai_val): cerai_val = 0

            ax_perbandingan.text(x[i] - bar_width/2, nikah_val + max(perbandingan_df['Nikah'].max(),0)*0.01,
                                 f"{int(nikah_val):,}", ha='center', va='bottom', fontsize=8, color='green') #
            ax_perbandingan.text(x[i] + bar_width/2, cerai_val + max(perbandingan_df['Jumlah Cerai'].max(),0)*0.01,
                                 f"{int(cerai_val):,}", ha='center', va='bottom', fontsize=8, color='gold') #

        # Judul dan tampilan
        ax_perbandingan.set_xticks(x) #
        ax_perbandingan.set_xticklabels(perbandingan_df['Tahun'].astype(str), rotation=45) #
        ax_perbandingan.set_title(f'Perbandingan Pernikahan vs Perceraian per Tahun di Provinsi {selected_provinsi_nikah_cerai}', fontsize=14, weight='bold') #
        ax_perbandingan.set_xlabel('Tahun') #
        ax_perbandingan.set_ylabel('Jumlah Kasus') #
        ax_perbandingan.legend() #
        plt.tight_layout() #
        st.pyplot(fig_perbandingan) # Menampilkan peta di Streamlit


    # --- Tambahkan kode visualisasi faktor perceraian per tahun di sini ---
    st.subheader("Faktor Penyebab Perceraian per Tahun (Interaktif)")
    st.write("Gunakan dropdown di bawah untuk melihat faktor perceraian dominan di Indonesia untuk setiap tahun.")

    # Daftar kolom faktor perceraian
    faktor_cols = [
        "Zina", "Mabuk", "Madat", "Judi", "Meninggalkan Salah satu Pihak",
        "Dihukum Penjara", "Poligami", "Kekerasan Dalam Rumah Tangga",
        "Cacat Badan", "Perselisihan dan Pertengkaran Terus Menerus",
        "Kawin Paksa", "Murtad", "Ekonomi", "Lain-lain",
    ]

    # Pastikan 'Tahun' adalah tipe data numerik dan ambil tahun yang tersedia
    df_perceraian['Tahun'] = pd.to_numeric(df_perceraian['Tahun'], errors='coerce')
    available_years = sorted(df_perceraian['Tahun'].dropna().unique().astype(int))

    # Dropdown interaktif menggunakan st.selectbox
    selected_year_faktor = st.selectbox(
        "Pilih Tahun untuk Faktor Perceraian:",
        options=available_years,
        index=len(available_years) - 1, # Default ke tahun terakhir
        key="divorce_faktor_year_selectbox"
    )

    if selected_year_faktor:
        data_filtered_faktor = df_perceraian[df_perceraian['Tahun'] == selected_year_faktor]
        
        current_faktor_cols = [col for col in faktor_cols if col in data_filtered_faktor.columns]
        if not current_faktor_cols:
            st.warning(f"Tidak ada kolom faktor perceraian yang ditemukan untuk Tahun {selected_year_faktor}. Data mungkin tidak lengkap.")
        else:
            total_per_faktor = data_filtered_faktor[current_faktor_cols].sum().sort_values(ascending=True)

            # Set gaya visual seaborn
            sns.set(style="whitegrid") # [cite: 0_[Main]

            fig_factors, ax_factors = plt.subplots(figsize=(12, 8))
            bars = ax_factors.barh(total_per_faktor.index, total_per_faktor.values,
                                    color=sns.color_palette("RdPu", len(total_per_faktor)))

            # Tambah label di ujung bar
            for bar in bars:
                ax_factors.text(bar.get_width() + (total_per_faktor.max() * 0.01), bar.get_y() + bar.get_height()/2,
                                f'{int(bar.get_width()):,}', va='center', fontsize=10)

            ax_factors.set_title(f"Faktor Perceraian di Indonesia Tahun {selected_year_faktor}", fontsize=16, fontweight='bold')
            ax_factors.set_xlabel("Jumlah Kasus", fontsize=12)
            ax_factors.set_ylabel("Faktor Perceraian", fontsize=12)
            ax_factors.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_factors)


    st.subheader("Peta Rata-rata Rasio Perceraian per Provinsi")
    st.write("Peta ini menampilkan rata-rata rasio perceraian (jumlah perceraian per 1000 pernikahan) di setiap provinsi di Indonesia dari tahun 2018 hingga 2024.")

    @st.cache_data
    def load_and_merge_geodata(df_divorce_data):
        try:
            # GANTI 'path/to/indonesia_provinces.shp' DENGAN PATH KE FILE .shp ANDA
            # Pastikan semua file shapefile (.shp, .shx, .dbf, dll.) ada di direktori yang sama
            # Contoh: 'indonesia_provinces.shp' jika semua file ada di folder proyek
            gdf = gpd.read_file(r'all_maps_state_indo.geojson') # <<< GANTI DENGAN PATH SHAPEFILE ANDA
            st.success("Shapefile peta Indonesia berhasil dimuat!")
        except FileNotFoundError:
            st.error("Error: Shapefile peta Indonesia ('all_maps_state_indo.geojson' atau nama file Anda) tidak ditemukan. Mohon unggah semua file shapefile ke direktori proyek.")
            return None
        except Exception as e:
            st.error(f"Error saat memuat shapefile: {e}")
            return None

        # 2. Normalisasi nama provinsi
        alias = {
            'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
            'DKI JAKARTA': 'JAKARTA' # Tambahkan alias jika diperlukan
        }
        gdf['provinsi_fix'] = gdf['state'].replace(alias).str.upper() # Ganti 'state' dengan nama kolom provinsi di shapefile Anda
        df_divorce_data['Provinsi_fix'] = df_divorce_data['Provinsi'].str.upper()

        # 3. Drop provinsi pemekaran baru dari df (tidak ada di shapefile lama)
        provinsi_pemekaran_papua = [
            'PAPUA BARAT DAYA', 'PAPUA PEGUNUNGAN', 'PAPUA SELATAN', 'PAPUA TENGAH'
        ]
        df_divorce_data = df_divorce_data[~df_divorce_data['Provinsi_fix'].isin(provinsi_pemekaran_papua)].copy()

        # Hitung rasio perceraian (jika belum ada, atau hitung ulang rata-rata)
        # Jika kolom 'Jumlah Cerai' atau 'Nikah' tidak ada setelah merge, ini bisa error
        # Asumsi 'Nikah' dan 'Jumlah Cerai' sudah ada di df_divorce_data
        df_divorce_data['Rasio_Perceraian'] = (df_divorce_data['Jumlah Cerai'] / df_divorce_data['Nikah']) * 1000 # Rasio per 1000 pernikahan


        # 4. Hitung rata-rata rasio perceraian per provinsi (dalam persen)
        # Menggunakan kolom 'Rasio_Perceraian' yang baru dihitung
        avg_rasio = df_divorce_data.groupby('Provinsi_fix')['Rasio_Perceraian'].mean().reset_index()
        avg_rasio['Rasio_Perceraian'] = avg_rasio['Rasio_Perceraian'] # Sudah dalam per 1000, tidak perlu * 100 lagi untuk persen.

        # 5. Gabungkan ke GeoDataFrame
        merged = gdf.merge(avg_rasio, how='left', left_on='provinsi_fix', right_on='Provinsi_fix')
        
        # Mengisi NaN pada kolom rasio perceraian setelah merge (provinsi tanpa data)
        merged['Rasio_Perceraian'] = merged['Rasio_Perceraian'].fillna(0) # Atau np.nan untuk 'Data Tidak Tersedia'

        return merged

    merged_data = load_and_merge_geodata(df_perceraian.copy()) # Pass a copy of df_perceraian

    if merged_data is None or merged_data.empty:
        st.error("Gagal membuat data peta. Pastikan shapefile dan data perceraian tersedia dan benar.")
        st.stop()

    # 6. Visualisasi
    fig_map, ax_map = plt.subplots(1, 1, figsize=(20, 15))
    merged_data.plot(column='Rasio_Perceraian',
                     cmap='YlOrRd',
                     linewidth=0.8,
                     edgecolor='0.8',
                     legend=True,
                     legend_kwds={'label': "Rasio Perceraian (per 1000 Pernikahan)", 'shrink': 0.5},
                     ax=ax_map,
                     missing_kwds={
                         "color": "lightgrey",
                         "label": "Data Tidak Tersedia"
                     })

    # 7. Tambahkan label nama provinsi
    for idx, row in merged_data.iterrows():
        # Handle cases where centroid might be empty for some geometries (e.g., MultiPolygon)
        if row['geometry'] and not row['geometry'].is_empty:
            try:
                # Use representative_point for better label placement in complex geometries
                centroid = row['geometry'].representative_point()
                ax_map.annotate(text=row['provinsi_fix'].replace('DI YOGYAKARTA', 'DIY').title(), # Singkat DIY
                                xy=(centroid.x, centroid.y),
                                horizontalalignment='center',
                                fontsize=8,
                                color='black',
                                weight='bold')
            except Exception as e:
                st.warning(f"Tidak dapat menambahkan label untuk provinsi {row['provinsi_fix']}: {e}")

    # 8. Judul dan tampilan
    ax_map.set_title('Peta Rata-rata Rasio Perceraian per Provinsi di Indonesia (2018–2024)', fontsize=20)
    ax_map.axis('off') # Sembunyikan sumbu
    plt.tight_layout()
    st.pyplot(fig_map) # Menampilkan peta di Streamlit

    st.subheader("Prediksi Jumlah Perceraian Nasional hingga 2025")
    st.write("Model regresi linear digunakan untuk memprediksi tren jumlah perceraian nasional hingga tahun 2025.")

    # 1) Agregasi total nasional per tahun
    # Menggunakan df_perceraian dan kolom 'Jumlah Cerai'
    yearly = (
        df_perceraian.groupby('Tahun', as_index=False)
        .agg({'Jumlah Cerai': 'sum', 'Nikah': 'sum'})
        .sort_values('Tahun')
    ) #_Salinan_Load_Data (1).ipynb]

    # Pastikan nama kolom sesuai dengan yang diharapkan oleh kode selanjutnya
    yearly = yearly.rename(columns={'Jumlah Cerai': 'Jumlah'}) #_Salinan_Load_Data (1).ipynb]


    # 2) Siapkan data latih
    X = yearly[['Tahun', 'Nikah']] #_Salinan_Load_Data (1).ipynb]
    y = yearly['Jumlah'] #_Salinan_Load_Data (1).ipynb]

    # Menggunakan @st.cache_resource untuk menyimpan model
    @st.cache_resource
    def train_divorce_prediction_model(X_data, y_data):
        model = LinearRegression() #_Salinan_Load_Data (1).ipynb]
        model.fit(X_data, y_data) #_Salinan_Load_Data (1).ipynb]
        return model
    
    model_prediksi_perceraian = train_divorce_prediction_model(X, y)

    # 3) Estimasi nilai fitur untuk 2025 (prediksi Nikah 2025)
    @st.cache_resource
    def train_nikah_prediction_model(yearly_data):
        nikah_model = LinearRegression() #_Salinan_Load_Data (1).ipynb]
        nikah_model.fit(yearly_data[['Tahun']], yearly_data['Nikah']) #_Salinan_Load_Data (1).ipynb]
        return nikah_model
    
    nikah_model_prediksi = train_nikah_prediction_model(yearly)
    pred_nikah_2025 = nikah_model_prediksi.predict(pd.DataFrame({'Tahun': [2025]}))[0] #_Salinan_Load_Data (1).ipynb]

    # 4) Prediksi cerai 2025
    X_2025 = pd.DataFrame({'Tahun': [2025], 'Nikah': [pred_nikah_2025]}) #_Salinan_Load_Data (1).ipynb]
    pred_cerai_2025 = model_prediksi_perceraian.predict(X_2025)[0] #_Salinan_Load_Data (1).ipynb]

    # 5) Gabungkan data historis + prediksi
    pred_row = pd.DataFrame({
        'Tahun': [2025],
        'Nikah': [pred_nikah_2025],
        'Jumlah': [pred_cerai_2025]
    }) #_Salinan_Load_Data (1).ipynb]
    yearly_extended = pd.concat([yearly, pred_row], ignore_index=True) #_Salinan_Load_Data (1).ipynb]

    # 6) Plot
    fig_prediksi, ax_prediksi = plt.subplots(figsize=(12, 6)) #_Salinan_Load_Data (1).ipynb]
    # Garis data historis
    ax_prediksi.plot(yearly['Tahun'], yearly['Jumlah'], marker='o', label='Data Historis', color='blue') #_Salinan_Load_Data (1).ipynb]
    # Titik prediksi 2025
    ax_prediksi.plot(2025, pred_cerai_2025, 'ro', label='Prediksi 2025') #_Salinan_Load_Data (1).ipynb]
    # Garis putus-putus dari tahun terakhir ke 2025
    ax_prediksi.plot([yearly['Tahun'].iloc[-1], 2025],
                     [yearly['Jumlah'].iloc[-1], pred_cerai_2025],
                     linestyle='--', color='gray', label='Garis Prediksi') #_Salinan_Load_Data (1).ipynb]

    # Tambahkan label jumlah di atas setiap titik
    for i, row in yearly_extended.iterrows(): #_Salinan_Load_Data (1).ipynb]
        ax_prediksi.text(row['Tahun'], row['Jumlah'] + (yearly_extended['Jumlah'].max() * 0.01), # Disesuaikan posisi teks
                         f"{int(row['Jumlah']):,}", ha='center', fontsize=9) #_Salinan_Load_Data (1).ipynb]

    # Tambahkan garis vertikal sebagai penanda awal prediksi (misal tahun prediksi mulai setelah 2024)
    ax_prediksi.axvline(x=2024.5, color='black', linestyle='--', label='Prediksi dimulai') #_Salinan_Load_Data (1).ipynb]


    # Format plot
    ax_prediksi.set_title('Prediksi Jumlah Perceraian Nasional hingga 2025') #_Salinan_Load_Data (1).ipynb]
    ax_prediksi.set_xlabel('Tahun') #_Salinan_Load_Data (1).ipynb]
    ax_prediksi.set_ylabel('Jumlah Perceraian') #_Salinan_Load_Data (1).ipynb]
    ax_prediksi.grid(True) #_Salinan_Load_Data (1).ipynb]
    ax_prediksi.legend() #_Salinan_Load_Data (1).ipynb]
    plt.tight_layout() #_Salinan_Load_Data (1).ipynb]
    st.pyplot(fig_prediksi) # Menampilkan plot di Streamlit

    st.subheader("Evaluasi Model Prediksi Perceraian")
    st.write("Metrik evaluasi untuk menilai performa model regresi linear dalam memprediksi jumlah perceraian.")

    # Gunakan variabel yang sudah ada dari bagian prediksi
    # X: data latih fitur (Tahun, Nikah)
    # y: data latih target (Jumlah Cerai, renamed to 'Jumlah')
    # model_prediksi_perceraian: model LinearRegression yang sudah dilatih

    y_pred = model_prediksi_perceraian.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    col_mae, col_mse, col_rmse, col_r2 = st.columns(4)

    with col_mae:
        st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    with col_mse:
        st.metric("MSE (Mean Squared Error)", f"{mse:.2f}")
    with col_rmse:
        st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
    with col_r2:
        st.metric("R-squared", f"{r2:.2f}")

# --- FUNGSI HALAMAN BARU: ANALISIS KEMISKINAN ---
def page_analisis_kemiskinan():
    display_section_header("📈 Analisis Faktor Kemiskinan di Jawa Tengah")
    st.write("""
    Proyek ini bertujuan untuk mengetahui faktor-faktor yang paling mempengaruhi kemiskinan di kabupaten/kota di Jawa Tengah guna menjadi saran untuk perbaikan dan meminimalisir kemiskinan.
    """)

    @st.cache_data
    def load_and_preprocess_kemiskinan_data():
        try:
            st.write("--- Memuat Dataset Gabungan ---")
            # Memuat file data_gabungan.csv
            # Pastikan ini adalah file CSV yang sudah Anda gabungkan
            df_gabungan = pd.read_csv('data_gabungan.csv') # <<< PATH FILE GABUNGAN ANDA
            st.success(f"Dataset gabungan berhasil dimuat! ({len(df_gabungan)} baris)")

            if df_gabungan.empty:
                st.error("Dataset gabungan kosong setelah dimuat. Tidak dapat melanjutkan.")
                return pd.DataFrame()

            st.write("--- Memulai Pembersihan dan Normalisasi Dataset Gabungan ---")

            # Normalisasi 'kabupaten'
            if 'kabupaten' in df_gabungan.columns:
                df_gabungan['kabupaten'] = df_gabungan['kabupaten'].astype(str).str.strip().str.lower()
            else:
                st.error("Kolom 'kabupaten' tidak ditemukan di dataset gabungan. Mohon periksa nama kolom.")
                return pd.DataFrame()

            # Kolom-kolom yang seharusnya numerik
            numeric_cols_to_process = [
                'persentase_kemiskinan_2024', 'APS 7-12', 'APS 13-15', 'APS 16-18', 'UMR', 'PDRB',
                'jumlah penduduk 2024', 'Tingkat Penganguran Terbuka (TPT)',
                'Rata - Rata Lama Sekolah', 'ketimpangan', 'laju pertumbuhan'
            ]
            
            # Lakukan konversi ke numerik secara agresif
            for col in numeric_cols_to_process:
                if col in df_gabungan.columns:
                    # Mengganti '...' dengan NaN, menghapus spasi ekstra, lalu konversi ke numerik
                    df_gabungan[col] = df_gabungan[col].astype(str).str.replace('...', '', regex=False).str.strip()
                    df_gabungan[col] = pd.to_numeric(df_gabungan[col], errors='coerce')
                else:
                    st.warning(f"Kolom '{col}' tidak ditemukan di dataset gabungan. Mungkin ada kesalahan penggabungan atau nama kolom.")

            # Hapus baris yang memiliki NaN di kolom-kolom numerik utama setelah konversi
            # Ini akan memastikan hanya baris dengan data lengkap yang digunakan
            df_gabungan = df_gabungan.dropna(subset=numeric_cols_to_process).reset_index(drop=True)

            st.success(f"Dataset gabungan berhasil diproses dan dibersihkan! ({len(df_gabungan)} baris akhir)")

            if df_gabungan.empty:
                st.error("Dataset gabungan kosong setelah pembersihan. Ini menandakan tidak ada kabupaten yang memiliki data lengkap atau konversi data gagal.")
                st.stop()

            return df_gabungan

        except FileNotFoundError as e:
            st.error(f"Error: File gabungan ('data_gabungan.csv') tidak ditemukan. Pastikan nama file dan ekstensi sudah benar serta file ada di direktori proyek. Detail: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Terjadi error saat memuat atau memproses dataset gabungan: {e}")
            return pd.DataFrame()

    gabungan_df = load_and_preprocess_kemiskinan_data()

    if gabungan_df.empty:
        st.stop()

    st.subheader("Data Gabungan (Preview)")
    st.dataframe(gabungan_df.head())
    st.write("Tipe Data Gabungan:")
    st.dataframe(gabungan_df.dtypes.reset_index().rename(columns={'index': 'Kolom', 0: 'Tipe Data'}))


    st.subheader("Visualisasi Utama Kemiskinan")

    # --- Plot: Top 5 Kabupaten Termiskin ---
    st.write("#### Persentase TOP 5 Kabupaten Termiskin di Jawa Tengah (2024)")
    top5_termiskin = gabungan_df.sort_values(by='persentase_kemiskinan_2024', ascending=False).head(5)
    fig_miskin, ax_miskin = plt.subplots(figsize=(10, 6))
    sns.barplot(x='persentase_kemiskinan_2024', y='kabupaten', data=top5_termiskin, palette='Reds_r', hue='kabupaten', legend=False, ax=ax_miskin)
    ax_miskin.set_xlabel('Persentase Kemiskinan (%)')
    ax_miskin.set_ylabel('Kabupaten')
    ax_miskin.set_title('Persentase TOP 5 Kabupaten Termiskin di Jawa Tengah (2024)')
    for i, v in enumerate(top5_termiskin['persentase_kemiskinan_2024']):
        ax_miskin.text(v + (top5_termiskin['persentase_kemiskinan_2024'].max() * 0.005), i, f"{v:.1f}%", va='center', fontsize=10, color='white')
    ax_miskin.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_miskin)


    # --- Plot: Top 5 Kabupaten dengan UMR Terendah ---
    st.write("#### 5 Kabupaten dengan UMR Terendah (2024)")
    umr_terendah = gabungan_df.sort_values(by='UMR', ascending=True).head(5)
    fig_umr, ax_umr = plt.subplots(figsize=(10,6))
    sns.barplot(x='UMR', y='kabupaten', data=umr_terendah, palette='Reds_r', hue='kabupaten', legend=False, ax=ax_umr)
    ax_umr.set_xlabel('UMR (Rp)')
    ax_umr.set_ylabel('Kabupaten')
    ax_umr.set_title('5 Kabupaten dengan UMR Terendah (2024)')
    for i, v in enumerate(umr_terendah['UMR']):
        ax_umr.text(v + (umr_terendah['UMR'].max()*0.005), i, f"Rp {int(v):,}", va='center', fontsize=9, color='white')
    ax_umr.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_umr)


    # --- Plot: Top 5 Kabupaten dengan APS 16-18 Terendah ---
    st.write("#### 5 Kabupaten dengan Angka Partisipasi Sekolah 16 - 18 (SMA) Rendah")
    top5_APS_rendah = gabungan_df.sort_values(by='APS 16-18', ascending=True).head(5)
    fig_aps, ax_aps = plt.subplots(figsize=(10,6))
    sns.barplot(x='APS 16-18', y='kabupaten', data=top5_APS_rendah, palette='Reds_r', hue='kabupaten', legend=False, ax=ax_aps)
    ax_aps.set_xlabel('Angka Partisipasi Sekolah')
    ax_aps.set_ylabel('Kabupaten')
    ax_aps.set_title('5 Kabupaten dengan Angka Partisipasi Sekolah 16 - 18 (SMA) Rendah')
    for i, v in enumerate(top5_APS_rendah['APS 16-18']):
        ax_aps.text(v + (top5_APS_rendah['APS 16-18'].max()*0.005), i, f"{v:.1f}%", va='center', fontsize=10, color='white')
    ax_aps.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_aps)


    # --- Scatter Plots: APS 16-18 vs. Faktor Lain ---
    st.write("#### Korelasi Angka Partisipasi Sekolah (APS 16-18) dengan Faktor Lain")
    st.write("Hubungan antara Angka Partisipasi Sekolah usia 16-18 tahun dengan Persentase Kemiskinan, Tingkat Pengangguran Terbuka (TPT), dan Rata-rata Lama Sekolah.")
    
    st.write("Korelasi APS 7–12 dengan Kemiskinan:", gabungan_df['APS 7-12'].corr(gabungan_df['persentase_kemiskinan_2024']))
    st.write("Korelasi APS 13–15 dengan Kemiskinan:", gabungan_df['APS 13-15'].corr(gabungan_df['persentase_kemiskinan_2024']))
    st.write("Korelasi APS 16–18 dengan Kemiskinan:", gabungan_df['APS 16-18'].corr(gabungan_df['persentase_kemiskinan_2024']))

    fig_scatter_aps, axes_scatter_aps = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: APS 16–18 vs Kemiskinan
    sns.scatterplot(x='APS 16-18', y='persentase_kemiskinan_2024', data=gabungan_df, ax=axes_scatter_aps[0], hue='kabupaten', legend=False)
    axes_scatter_aps[0].set_title('APS 16–18 vs Kemiskinan')
    axes_scatter_aps[0].set_xlim(50, 100)
    axes_scatter_aps[0].set_xlabel('APS 16-18 (%)')
    axes_scatter_aps[0].set_ylabel('Persentase Kemiskinan (%)')

    # Subplot 2: APS 16–18 vs TPT
    sns.scatterplot(x='APS 16-18', y='Tingkat Penganguran Terbuka (TPT)', data=gabungan_df, ax=axes_scatter_aps[1], hue='kabupaten', legend=False)
    axes_scatter_aps[1].set_title('APS 16–18 vs TPT')
    axes_scatter_aps[1].set_xlim(50, 100)
    axes_scatter_aps[1].set_xlabel('APS 16-18 (%)')
    axes_scatter_aps[1].set_ylabel('TPT (%)')

    # Subplot 3: APS 16–18 vs Rata-rata Lama Sekolah
    sns.scatterplot(x='APS 16-18', y='Rata - Rata Lama Sekolah', data=gabungan_df, ax=axes_scatter_aps[2], hue='kabupaten', legend=False)
    axes_scatter_aps[2].set_title('APS 16–18 vs Rata-Rata Lama Sekolah')
    axes_scatter_aps[2].set_xlim(50, 100)
    axes_scatter_aps[2].set_xlabel('APS 16-18 (%)')
    axes_scatter_aps[2].set_ylabel('Rata-rata Lama Sekolah (tahun)')

    plt.tight_layout()
    st.pyplot(fig_scatter_aps)


    # --- Plot: Top 10 Kabupaten dengan PDRB Terendah ---
    st.write("#### 10 Kabupaten dengan PDRB Rendah")
    top10_PDRB = gabungan_df.sort_values(by='PDRB', ascending=True).head(10)
    fig_pdrb, ax_pdrb = plt.subplots(figsize=(12,8))
    sns.barplot(x='PDRB', y='kabupaten', data=top10_PDRB, palette='Reds_r', hue='kabupaten', legend=False, ax=ax_pdrb)
    ax_pdrb.set_xlabel('PDRB (juta Rp)')
    ax_pdrb.set_ylabel('Kabupaten')
    ax_pdrb.set_title('10 Kabupaten dengan PDRB Rendah')
    for i, v in enumerate(top10_PDRB['PDRB']):
        ax_pdrb.text(v + (top10_PDRB['PDRB'].max()*0.005), i, f"{v:,.1f}", va='center', fontsize=10, color='white')
    ax_pdrb.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_pdrb)


    # --- Plot: Top 5 Kabupaten dengan TPT Tertinggi ---
    st.write("#### 5 Kabupaten dengan Persentase Pengangguran Tertinggi")
    top5_pengangguran_persen = gabungan_df.sort_values(by='Tingkat Penganguran Terbuka (TPT)', ascending=False).head(5)
    fig_tpt, ax_tpt = plt.subplots(figsize=(10,6))
    sns.barplot(x='Tingkat Penganguran Terbuka (TPT)', y='kabupaten', data=top5_pengangguran_persen, palette='Reds_r', hue='kabupaten', legend=False, ax=ax_tpt)
    ax_tpt.set_xlabel('Persentase Pengangguran (%)')
    ax_tpt.set_ylabel('Kabupaten')
    ax_tpt.set_title('5 Kabupaten dengan Persentase Pengangguran Tertinggi')
    for i, v in enumerate(top5_pengangguran_persen['Tingkat Penganguran Terbuka (TPT)']):
        ax_tpt.text(v + (top5_pengangguran_persen['Tingkat Penganguran Terbuka (TPT)'].max()*0.005), i, f"{v:.1f}%", va='center', fontsize=10, color='white')
    ax_tpt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_tpt)


    # --- XGBoost Model & Feature Importance ---
    st.subheader("Model Prediksi Kemiskinan: XGBoost Regressor")
    st.write("""
    Model XGBoost Regressor digunakan untuk memprediksi persentase kemiskinan berdasarkan berbagai faktor sosial-ekonomi di kabupaten/kota.
    """)

    X = gabungan_df.drop(columns=['persentase_kemiskinan_2024', 'kabupaten'])
    y = gabungan_df['persentase_kemiskinan_2024']

    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            X.loc[:, col] = X[col].astype('category').cat.codes
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')

    X = X.dropna(axis=1, how='all')
    y = y.dropna()
    common_indices = X.index.intersection(y.index)
    X = X.loc[common_indices]
    y = y.loc[common_indices]


    if X.empty or y.empty:
        st.error("Data fitur atau target kosong setelah preprocessing untuk model XGBoost. Tidak dapat melatih model.")
        st.stop()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_xgboost_model(X_train_data, y_train_data):
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train_data, y_train_data)
        return model
    
    xgboost_model = train_xgboost_model(X_train, y_train)
    st.success("Model XGBoost Regressor berhasil dilatih!")

    st.subheader("Evaluasi Model")
    y_pred = xgboost_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col_r2, col_rmse = st.columns(2)
    with col_r2:
        st.metric("R2 Score", f"{r2:.2f}")
    with col_rmse:
        st.metric("RMSE", f"{rmse:.2f}")

    st.subheader("Faktor yang Mempengaruhi Kemiskinan (Feature Importance)")
    st.write("""
    Grafik ini menunjukkan fitur-fitur yang paling berpengaruh dalam prediksi persentase kemiskinan, berdasarkan model XGBoost.
    """)
    
    importances = xgboost_model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]

    fig_fi_xgb, ax_fi_xgb = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx], palette='Blues_r', hue=feature_names[sorted_idx], legend=False, ax=ax_fi_xgb)
    
    for i, v in enumerate(importances[sorted_idx]):
        ax_fi_xgb.text(v + (importances.max()*0.005), i, f"{v:.3f}", color='white', va='center', fontsize=10)
    
    ax_fi_xgb.set_title('Faktor yang Mempengaruhi Kemiskinan model machine learning XGBoost', fontsize=14, weight='bold')
    ax_fi_xgb.set_xlabel('Importance Score')
    ax_fi_xgb.set_ylabel('Fitur')
    ax_fi_xgb.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_fi_xgb)

    # --- FUNGSI HALAMAN BARU: BOT TELEGRAM ---
# utils.py
# ... (kode import dan fungsi lainnya) ...

# utils.py
# ... (kode import dan fungsi lainnya) ...

# --- FUNGSI HALAMAN BARU: BOT TELEGRAM ---
def page_bot_telegram():
    display_section_header("🤖 Bot Pencatat Keuangan Pribadi (Telegram)")
    
    # Menggunakan st.columns untuk tata letak bersebelahan
    # Kolom pertama untuk gambar, kolom kedua untuk SEMUA deskripsi dan detail
    # Proporsi diubah menjadi [1, 3] untuk memberi lebih banyak ruang ke teks
    col_image, col_description_and_details = st.columns([1, 3]) # <<< Proporsi diubah

    with col_image:
        # Ganti dengan path foto bot Anda
        st.image("tele.png", caption="Tampilan Antarmuka Bot Telegram", width=300) # <<< Sesuaikan width jika perlu
        # Mengurangi width sedikit agar lebih mungkin muat berdampingan

    with col_description_and_details:
        st.write("""
        Bot ini adalah asisten keuangan pribadi berbasis Telegram yang dirancang untuk membantu individu mencatat dan melacak pemasukan serta pengeluaran mereka secara efisien. Dibangun dengan fokus pada kemudahan penggunaan dan akurasi data, bot ini memanfaatkan kemampuan pemrosesan bahasa alami (NLP) untuk mengotomatisasi pencatatan dan memberikan wawasan keuangan instan.
        """)

        st.subheader("Metode Pengembangan & Teknologi")
        st.markdown("""
        * **Bahasa Pemrograman**: Bot ini dikembangkan sepenuhnya menggunakan Python, yang dikenal karena fleksibilitas dan ekosistem pustaka yang kaya untuk data science dan otomatisasi.
        * **Framework Bot**: Menggunakan pustaka `python-telegram-bot` untuk berinteraksi dengan Telegram Bot API, menyederhanakan proses pengiriman dan pengiriman pesan.
        * **Manajemen Data**: Data transaksi dan anggaran disimpan secara lokal dalam database SQLite, dikelola melalui `SQLAlchemy`, sebuah Object-Relational Mapper (ORM) Python. Pendekatan ini memastikan persistensi data yang stabil dan terstruktur.
        * **Akurasi Finansial**: Untuk menghindari masalah presisi *floating-point* yang umum dalam perhitungan keuangan, bot ini menggunakan modul `decimal` Python, memastikan semua perhitungan saldo dan jumlah transaksi dilakukan dengan akurasi tinggi.
        * **Pendekatan AI**: Bot ini mengimplementasikan bentuk dasar Artificial Intelligence (AI), khususnya dalam bidang Natural Language Processing (NLP), melalui pendekatan Rule-Based AI.
        """)

        st.subheader("Fitur Utama")
        st.markdown("""
        * **Pencatatan Transaksi Otomatis**:
        * **Kategorisasi Transaksi Cerdas**:
        * **Pelacakan Saldo Real-time**:
        * **Pelacakan Anggaran & Peringatan**
        * **Rekap Transaksi Lengkap**:
        * **Manajemen Data Aman**:
        * **Antarmuka Pengguna Intuitif**:
            
        """)

        st.subheader("Aspek AI yang Diimplementasikan")
        st.markdown("""
        * **Natural Language Understanding (NLU)**: Fungsi `parse_message` adalah inti NLU bot, yang mengubah teks bebas pengguna menjadi data terstruktur (jumlah, tipe, deskripsi).
        * **Klasifikasi (Rule-Based)**: Fungsi `suggest_category` adalah sistem klasifikasi berbasis aturan yang secara otomatis mengelompokkan transaksi ke dalam kategori.
        * **Sistem Pengambilan Keputusan**: Logika pengecekan dan peringatan anggaran adalah contoh sistem berbasis aturan yang membuat keputusan cerdas berdasarkan data yang dicatat.
        """)

       
        st.markdown("""
    

        Secara keseluruhan, bot ini adalah alat yang cerdas dan praktis untuk manajemen keuangan pribadi, menunjukkan kemampuan AI dalam menyederhanakan tugas sehari-hari.
        """)

    st.write("---")
    st.markdown("[Lihat Kode Proyek di GitHub](https://github.com/Irkhamdwiramadhan)")

# ... (kode fungsi-fungsi halaman lainnya) ...