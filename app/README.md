# Prototipe Dashboard Streamlit (Tahap 6)

**Penanggung Jawab:** Irfan (Anggota 4 - Deployment Engineer)

Folder ini berisi kode sumber aplikasi web dashboard interaktif untuk sistem peringatan dini (*Early Warning System*) kelulusan mahasiswa.

## Cara Menjalankan Aplikasi
1. Buka terminal di folder ini:
   ```bash
   cd app
   ```
2. Pastikan dependencies telah terpasang:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

## Fitur Utama Aplikasi
1. **Prediksi Interaktif:** Form input data profil mahasiswa untuk memprediksi status kelulusan (Lulus / Berisiko Dropout) beserta skor probabilitas dan rekomendasi akademik.
2. **Komparasi 5 Algoritma:** Tampilan visual tabel metrik performa (*Accuracy, Precision, Recall, F1, ROC-AUC*) dan grafik perbandingan 5 algoritma.
3. **Eksplorasi Data:** Ringkasan dataset dan statistik deskriptif.
