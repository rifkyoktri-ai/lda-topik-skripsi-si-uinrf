# LDA Topic Modeling Dashboard

Dashboard Streamlit interaktif untuk visualisasi dan analisis hasil LDA Topic Modeling.

## 📋 Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run app/main.py
```

Dashboard akan membuka di browser pada `http://localhost:8501`

## 📊 Features

### 📈 Overview
- Dashboard overview dengan statistik utama
- Distribusi topik dominan
- Distribusi probabilitas
- Timeline skripsi per tahun

### 📊 Model Metrics
- Evaluation metrics (Coherence Score, Log Perplexity)
- Number of Topics
- Penjelasan setiap metric

### 🏷️ Topic Analysis
- Top topics berdasarkan jumlah documents
- Average probability per topic
- Detail analysis untuk setiap topic
- Daftar documents per topic

### 🔍 Document Search
- Search by: Nama Penulis, Judul, Tahun, atau Topic ID
- Filter documents berdasarkan criteria
- Download results as CSV

### 📖 Data Info
- Dataset statistics
- Probability statistics
- Data preview
- Data description

## 📁 Directory Structure
```
tes/
├── app/
│   └── main.py              (Dashboard main file)
├── data/
│   ├── raw/                 (Raw data)
│   └── proses/
│       ├── dataset_preprocessed.csv
│       ├── dictionary.gensim
│       └── ...
├── model/
│   ├── evaluation_metrics.csv
│   ├── topic_distribution.csv
│   ├── lda_model.gensim
│   └── lda_visualization.html
├── notebook/                (Jupyter notebooks)
└── requirements.txt         (Python dependencies)
```

## 🎨 Customization

Anda dapat mengubah:
- Colors dan styling di custom CSS section
- Page layout
- Visualization tipe dan parameter
- Filter options

## 📝 Notes

- Dashboard menggunakan caching untuk performa optimal
- Data diload dari CSV files
- Plotly digunakan untuk interactive visualizations
- Streamlit auto-reloads ketika ada perubahan kode

## 🆘 Troubleshooting

Jika ada error "No such file or directory":
- Pastikan Anda menjalankan command dari root directory (tes/)
- Periksa path ke data files

Jika dashboard lambat:
- Clear Streamlit cache: `streamlit cache clear`
- Kurangi jumlah data yang ditampilkan

## 📧 Support

Untuk pertanyaan atau issues, silakan hubungi tim data science.
