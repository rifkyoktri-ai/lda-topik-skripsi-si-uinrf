"""
Unit Test Verifikasi Integritas Modul topic_labels & labels.json (outputs/model/)
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Unbuffer stdout for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

from src.topic_labels import validate_labels

def test_validate_labels_success():
    """Memastikan validate_labels() lolos pada model final + labels.json yang benar"""
    assert validate_labels("outputs/model/lda_model.gensim", "outputs/model/labels.json") is True
    print("PASSED: test_validate_labels_success")

def test_validate_labels_tampered_fingerprint(tmp_path):
    """Memastikan validate_labels() gagal keras jika fingerprint diubah 1 karakter"""
    with open("outputs/model/labels.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Tamper fingerprint topic 1
    data[0]['top_words_fingerprint'] = data[0]['top_words_fingerprint'][:-1] + ('0' if data[0]['top_words_fingerprint'][-1] != '0' else '1')
    
    tampered_file = tmp_path / "labels_tampered.json"
    with open(tampered_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    try:
        validate_labels("outputs/model/lda_model.gensim", str(tampered_file))
        assert False, "Expected ValueError for tampered fingerprint"
    except ValueError as e:
        assert "[Gagal Validasi Fingerprint]" in str(e)
        print("PASSED: test_validate_labels_tampered_fingerprint")

def test_validate_labels_invalid_topic_count(tmp_path):
    """Memastikan validate_labels() gagal jika jumlah topik != 8"""
    with open("outputs/model/labels.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Remove last topic
    data_short = data[:7]
    
    short_file = tmp_path / "labels_short.json"
    with open(short_file, 'w', encoding='utf-8') as f:
        json.dump(data_short, f, indent=2)
        
    try:
        validate_labels("outputs/model/lda_model.gensim", str(short_file))
        assert False, "Expected ValueError for invalid topic count"
    except ValueError as e:
        assert "tidak cocok dengan jumlah topik model" in str(e)
        print("PASSED: test_validate_labels_invalid_topic_count")

def test_labels_json_schema():
    """Memastikan labels.json memuat tepat 8 topik, status disetujui, dan tingkat kepercayaan valid"""
    with open("outputs/model/labels.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    assert len(data) == 8, f"Expected 8 topics, got {len(data)}"
    valid_confidences = {"Tinggi", "Sedang", "Rendah"}
    
    for item in data:
        assert item["status"] == "disetujui", f"Topic {item['topic_id']} status is not 'disetujui'"
        assert item["label"].strip() != "", f"Topic {item['topic_id']} has empty label"
        assert item["tingkat_kepercayaan"] in valid_confidences, f"Topic {item['topic_id']} confidence invalid"
    print("PASSED: test_labels_json_schema")

def test_auto_labeling_spot_check_topic4():
    """Memastikan src/labeling.py menghasilkan label yang sama untuk topik 4"""
    from src.labeling import load_approved_labels
    labels_map = load_approved_labels("outputs/model/labels.json", "outputs/model/lda_model.gensim")
    
    # Topic 4 internal key 0-indexed is "3"
    t4_label = labels_map["3"]["label_final"]
    assert t4_label == "Kepuasan Pengguna Akhir (EUCS & TAM)", f"Spot check failed: got '{t4_label}'"
    print("PASSED: test_auto_labeling_spot_check_topic4")

def run_all_tests():
    import tempfile
    test_validate_labels_success()
    test_labels_json_schema()
    test_auto_labeling_spot_check_topic4()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_validate_labels_tampered_fingerprint(tmp_path)
        test_validate_labels_invalid_topic_count(tmp_path)
        
    print("\nALL 5 UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
