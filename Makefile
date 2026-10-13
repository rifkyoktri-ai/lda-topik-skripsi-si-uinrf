# Makefile - satu pintu masuk untuk seluruh perintah repo ini.
#
# formerly 12+ perintah yang harus dihafal; sekarang cukup satu kata:
#     make help     lihat semua perintah
#
# Semua perintah di bawahnya memanggil modul src/ atau runner yang sudah ada.

# Windows dan Linux punya lokasi venv yang berbeda; dipakai make test dll.

NB      := notebooks
NB_ORDER := 01_business_understanding 02_data_understanding 03_data_preparation \
            04_modeling_lda 04b_grid_search 05_evaluation_lda \
            06_evaluation_forecast 07_labeling 08_build_final_artifacts

.DEFAULT_GOAL := help
.PHONY: help install test notebook notebook-fast audit dashboard all clean structure check

## help      - tampilkan daftar perintah
help:
	@echo ""
	@echo "  REPO SKRIPSI LDA - UIN RADEN FATAH"
	@echo "  ---------------------------------"
	@echo ""
	@echo "  make install     pasang dependensi"
	@echo "  make test        jalankan 27 unit test"
	@echo "  make notebook    jalankan notebook 01-08 (lama, ~20 menit)"
	@echo "  make notebook-fast  jalankan 01-03 + 05-08 (lewati training, ~2 menit)"
	@echo "  make audit       regenerate outputs/audit/assignment_audit.md"
	@echo "  make structure   validasi struktur repo"
	@echo "  make dashboard   jalankan dashboard Streamlit"
	@echo "  make all         test + notebook-fast + audit  (urutan aman)"
	@echo "  make clean       hapus artefak turunan"
	@echo ""
	@echo "  Dokumentasi penuh: baca README.md"
	@echo ""

## install   - pasang dependensi
install:
	$(PYTHON) -m pip install -r requirements.txt

## test      - 27 unit test (pipeline + integritas + label + statistik + metrik)
test:
	$(PYTHON) run_tests.py
	$(PYTHON) tests/test_final.py
	$(PYTHON) tests/test_labels.py
	$(PYTHON) tests/test_stats.py

## structure - validasi struktur repo (path, git tracking, folder kosong)
structure:
	$(PYTHON) tests/test_structure.py

## notebook  - jalankan semua notebook 01-08 dan simpan outputnya
notebook:
	@for n in $(NB_ORDER); do \
		echo "--- $$n ---"; \
		$(PYTHON) run_notebook.py $(NB)/$$n.ipynb || exit 1; \
	done

## notebook-fast - lewati notebook yang dilatih ulang model (04, 04b)
notebook-fast:
	@for n in 01_business_understanding 02_data_understanding 03_data_preparation \
	          05_evaluation_lda 06_evaluation_forecast 07_labeling \
	          08_build_final_artifacts; do \
		echo "--- $$n ---"; \
		$(PYTHON) run_notebook.py $(NB)/$$n.ipynb || exit 1; \
	done

## audit     - hitung ulang laporan keyakinan penugasan dokumen
audit:
	$(PYTHON) -m src.assignment_audit

## dashboard - jalankan dashboard Streamlit
dashboard:
	$(PYTHON) -m streamlit run app/main.py

## all       - urutan aman: test -> notebook cepat -> audit
all: test notebook-fast audit
	@echo ""
	@echo "  SEMUA SELESAI. Verifikasi GROUND_TRUTH.md untuk angka resmi."

## clean     - hapus artefak turunan (JAWANG HAPUS model!)
clean:
	@echo "  Menghapus __pycache__, logs, dan .ipynb_checkpoints"
	@rm -rf logs .ipynb_checkpoints
	@find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "  Selesai. outputs/ TIDAK disentuh - jangan dihapus manual."