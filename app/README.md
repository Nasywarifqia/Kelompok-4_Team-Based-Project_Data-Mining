# Prototipe Dashboard Streamlit (Tahap 6)

**Penanggung Jawab:** Irfan (Anggota 4 - Deployment Engineer)

Folder ini berisi kode sumber aplikasi web dashboard interaktif untuk **Sistem Deteksi Situs Web Phishing**.

## Cara Menjalankan Aplikasi
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Fitur Utama Aplikasi
1. **Cek URL Otomatis & Simulasi Fitur:** Pengguna dapat menempelkan alamat URL untuk ekstraksi fitur otomatis (atau menyetel 30 indikator teknis secara manual) lalu melihat hasil deteksi (*Legitimate* vs *Phishing*), probabilitas risiko, dan rekomendasi keamanan.
2. **Komparasi 5 Algoritma:** Tabel *Leaderboard* dan perbandingan metrik (*Accuracy, Precision, Recall Phishing, F1-Score, ROC-AUC*).
3. **Tentang Proyek:** Informasi tim dan ringkasan dataset UCI.
