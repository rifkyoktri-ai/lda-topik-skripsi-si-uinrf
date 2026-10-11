# CHANGELOG PELABELAN TOPIK (`CHANGELOG_LABEL.md`)
**Tanggal Execution:** 2 Oktober 2026  
**Peran:** Engineer NLP / Statistik  

---

## Perubahan & Artefak yang Dihasilkan di `_label_work/`

1. **`labels.json`**:
   - Berkas JSON resmi berisi 8 label topik terkunci yang disetujui penulis, dilengkapi hash SHA256 fingerprint top-10 kata kunci, deskripsi 1 kalimat, tingkat kepercayaan, dan status `disetujui`.
2. **`labels_legacy.json`**:
   - Berkas JSON label lama dengan status `deprecated`.
3. **`topic_labels.py`**:
   - Modul validasi integritas fingerprint label `validate_labels()`. Memastikan fail-fast (gagal keras) jika fingerprint kata kunci di `labels.json` tidak cocok 100% dengan model Gensim tersimpan.
4. **`auto_labeling_legacy.py`**:
   - Salinan asli modul `auto_labeling.py` sebelum pembaruan.
5. **`labels_final_for_thesis.md`**:
   - Berkas naskah penyajian label final resmi untuk Bab IV skripsi. Memuat top-10 kata kunci, 3 judul dokumen representatif, catatan tumpang tindih Jaccard, dan pernyataan eksplisit keterbatasan model untuk Topik 2 dan 8.
6. **`test_labels.py`**:
   - Berkas pengujian unit test otomatis untuk memverifikasi fungsionalitas `validate_labels()`.

---

## Perubahan Kode pada Repositori Utama

1. **`auto_labeling.py`**:
   - Diperbarui untuk memuat dan memvalidasi label dari `_label_work/labels.json` melalui `validate_labels()`, menghapus dependensi pada pelabelan dinamis yang tidak konsisten.
2. **`pipeline.py`**:
   - Ditambahkan pemanggilan `validate_labels()` di awal fungsi `main()` sebelum memproses data pipeline. Program akan berhenti (raise ValueError) jika validasi gagal.
3. **`app/main.py`**:
   - Ditambahkan pengecekan `validate_labels()` di awal rendering Streamlit dashboard. Jika validasi gagal, banner error merah ditampilkan dan rendering dihentikan (`st.stop()`).
