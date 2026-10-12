<!-- ============================================================
  LAPORAN EKSPLORASI - BUKAN ANGKA RESMI SKRIPSI
  Dihitung pada korpus prapemrosesan VERSI LAMA (4.289 kata unik),
  sedangkan model final memakai 1.849 kata unik.

  Topic Diversity versi lama  : Top-10 0,7875 | Top-15 0,7250
  Topic Diversity versi final : Top-10 0,7125 | Top-15 0,6917

  Perbedaan ini WAJIB (efek ukuran kosakata), bukan kesalahan hitung.

  TABEL KOHERENSI PER-TOPIK DI BAWAH INI tidak punya salinan di artefak
  resmi - pakai hanya sebagai umbra perbandingan kualitatif.
  ============================================================ -->

# LAPORAN KUALITAS & DIVERSITAS TOPIK (`topic_quality.md`)

## 1. Topik Diversitas (Topic Diversity)
- **Topic Diversity (Top-10 kata):** `0.7875` (78.75% kata unik di seluruh 8 topik).
- **Topic Diversity (Top-15 kata):** `0.7250` (72.50% kata unik di seluruh 8 topik).

---

## 2. Coherence Skor $C_V$ Per-Topik (Model Seed 42)

| Topik ID | Coherence Score ($C_V$) | Evaluasi Kualitas |
| :---: | :---: | :--- |
| **Topik 1** | `0.6812` | Sangat Tinggi (Sangat Fokus) |
| **Topik 2** | `0.4257` | Moderat |
| **Topik 3** | `0.5213` | Tinggi |
| **Topik 4** | `0.4883` | Moderat-Tinggi |
| **Topik 5** | `0.2909` | Rendah |
| **Topik 6** | `0.2399` | Sangat Rendah |
| **Topik 7** | `0.3024` | Rendah |
| **Topik 8** | `0.3473` | Rendah-Moderat |

---

## 3. Kandidat Stopword Domain (Muncul di $\ge 3$ Topik)
Kata-kata berikut muncul di top-15 kata pada 3 atau lebih topik berbeda. Kata-kata ini merupakan kandidat *domain-specific stopword* yang disarankan untuk dipertimbangkan dalam iterasi mendatang (tidak diterapkan pada eksperimen ini):
- `puas_guna`
- `kualitas`
- `teknologi`
- `tahu`
- `model`
- `pengaruh`
- `kelola_data`
- `extreme_programming`
- `lapor`
