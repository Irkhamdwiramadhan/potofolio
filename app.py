import streamlit as st
import pandas as pd
import numpy as np

# Mengimpor semua fungsi dan variabel yang diperlukan dari utils.py
from utils import (
    set_page_config,
    display_about_me_summary,
    display_skills_visualization,
    display_tools,
    display_project_summary_visualization,
    display_sidebar_profile,
    CONTACT_INFO,
    MY_NAME,
    ABOUT_ME_TEXT,
    page_prediksi_pemain_bola,
    page_analisis_ecommerce,
    page_prediksi_kredivo,
    page_analisis_perceraian,
    page_analisis_kemiskinan, # Halaman baru untuk analisis kemiskinan
    page_bot_telegram # Halaman baru untuk bot Telegram
)

# Mengatur konfigurasi dasar halaman Streamlit.
set_page_config(title="Portofolio Data M Irkham") 

# --- Kustomisasi Background ---
def add_custom_css():
    try:
        with open("style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Error: style.css not found. Make sure it's in the same directory as app.py")
        
add_custom_css()
# --- AKHIR KUSTOMISASI BACKGROUND ---

# =========================================================================
# BAGIAN PENTING: KONTEN SIDEBAR DENGAN NAVIGASI MANUAL
# =========================================================================
with st.sidebar:
    # 1. Menampilkan foto profil dan nama di bagian paling atas
    display_sidebar_profile() 


    # 2. Membuat navigasi manual menggunakan st.radio
    st.subheader("Navigasi Proyek")
    
    pages = {
        "Beranda": "home",
        "Analisis E-commerce": "ecommerce",
        "Prediksi Kredivo": "kredivo",
        "Prediksi Pemain Bola": "pemain_bola",
        "Analisis Perceraian": "perceraian",
        "Analisis Kemiskinan": "kemiskinan", # Tambahkan entri navigasi untuk analisis kemiskinan
        "Bot AI Manage Keuangan": "bot_telegram" # Tambahkan entri navigasi untuk bot Telegram
    }

    # Inisialisasi session_state jika belum ada
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    selected_page_label = st.radio(
        "Pilih Halaman",
        list(pages.keys()),
        index=list(pages.values()).index(st.session_state.current_page),
        key="main_navigation_radio"
    )

    # Simpan pilihan halaman ke session_state untuk rendering
    st.session_state.current_page = pages[selected_page_label]
# =========================================================================
# AKHIR KONTEN SIDEBAR
# =========================================================================

# =========================================================================
# KONTEN UTAMA BERDASARKAN PILIHAN NAVIGASI
# =========================================================================
if st.session_state.current_page == "home":
    st.markdown("<h1 style='text-align: center;'>Selamat Datang di Portofolio Data Saya!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Jelajahi proyek-proyek saya di bidang Data Science, Data Analysis, dan Data Engineering.</p>", unsafe_allow_html=True)
    st.markdown("---")
    display_about_me_summary() 
    display_skills_visualization()
    display_tools()
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Sekilas Proyek-Proyek Data Saya</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Analisis mendalam dan model Machine Learning yang telah saya kembangkan.</p>", unsafe_allow_html=True)
    display_project_summary_visualization()
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Terhubung Dengan Saya!</h3>", unsafe_allow_html=True)
    contact_cols = st.columns(len(CONTACT_INFO))
    for i, (platform, link) in enumerate(CONTACT_INFO.items()):
        if platform in ["LinkedIn", "GitHub", "Medium"]:
            with contact_cols[i]:
                st.markdown(f"<p style='text-align: center;'><a href='{link}' target='_blank'>{platform}</a></p>", unsafe_allow_html=True)
        elif platform == "Email":
            with contact_cols[i]:
                st.markdown(f"<p style='text-align: center;'>Email: <a href='mailto:{link}'>{link}</a></p>", unsafe_allow_html=True)
        else:
            with contact_cols[i]:
                st.markdown(f"<p style='text-align: center;'>{platform}: {link}</p>", unsafe_allow_html=True)
    st.markdown("""
    <br>
    <p style='text-align: center; font-size: 0.8em;'>Dibuat dengan ❤️ dan Streamlit</p>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "ecommerce":
    page_analisis_ecommerce()
elif st.session_state.current_page == "kredivo":
    page_prediksi_kredivo()
elif st.session_state.current_page == "pemain_bola":
    page_prediksi_pemain_bola()
elif st.session_state.current_page == "perceraian":
    page_analisis_perceraian()
elif st.session_state.current_page == "kemiskinan":
    page_analisis_kemiskinan()
elif st.session_state.current_page == "bot_telegram": # <<< KONDISI UNTUK HALAMAN BOT TELEGRAM
    page_bot_telegram()