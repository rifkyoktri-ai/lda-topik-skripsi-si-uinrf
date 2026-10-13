<!-- ============================================================
  LAPORAN EKSPLORASI - MENUNJUKKAN LIMITASI, BUKAN KEUNGGULAN
  Eksperimen multi-seed N=10 pada K=8.

  HASIL PENTING: seluruh 8 topik punya Jaccard Index < 0,50
  (rata-rata 0,22 - 0,41) -> BELUM STABIL antar-seed.

  Ini temuan limitasi yang jujur. Bila dosen bertanya soal kestabilan
  model, jawaban WAJIB menyebut laporan ini. Klaim "model stabil"
  tidak boleh dibuat tanpa catatan ini.
  ============================================================ -->

# LAPORAN STABILITAS MULTI-SEED LDA (`stability_report.md`)

## 1. Hasil Eksperimen Multi-Seed (N=10, K=8, Passes=30)
Eksperimen dijalankan pada 10 seed acak untuk menguji sensitivitas model $K=8$ terhadap inisialisasi awal.

- **Coherence $C_V$:** Rata-rata = `0.3696`, SD = `0.0215`, Min = `0.3351`, Max = `0.4048`.
- **Coherence $U_{mass}$:** Rata-rata = `-9.6985`, SD = `0.5363`, Min = `-10.2192`, Max = `-8.2291`.

---

## 2. Stabilitas Kata Kunci (Hungarian Matching vs Seed 42)
Pencocokan topik antar-seed dilakukan menggunakan algoritma **Hungarian Matching** berdasarkan kemiripan *Jaccard Index* pada Top-15 kata per topik.

| Topik (Base Seed 42) | Rata-rata Jaccard Index | Status Stabil ($\ge 0.5$) |
| :---: | :---: | :---: |
| **Topik 1** | `0.4143` | TIDAK STABIL |
| **Topik 2** | `0.2765` | TIDAK STABIL |
| **Topik 3** | `0.3121` | TIDAK STABIL |
| **Topik 4** | `0.2483` | TIDAK STABIL |
| **Topik 5** | `0.3076` | TIDAK STABIL |
| **Topik 6** | `0.2351` | TIDAK STABIL |
| **Topik 7** | `0.3074` | TIDAK STABIL |
| **Topik 8** | `0.2227` | TIDAK STABIL |

- **Jumlah Topik Stabil (Jaccard $\ge 0.5$):** **0 / 8 topik**.
- **Tingkat Kesepakatan Penugasan Dokumen (Topic Assignment Match Rate):** `27.12%`.
- **Adjusted Rand Index (ARI):** `0.1213`.

---

## 3. Implikasi Kritis
Stabilitas topik antar-seed tergolong **rendah**. Hal ini terjadi karena ukuran korpus yang relatif kecil (285 dokumen) dan overlap terminologi antar-topik. Penugasan dokumen dominan sangat bergantung pada penetapan `random_state` yang tetap (misalnya seed 42).
