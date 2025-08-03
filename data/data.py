from datasets import load_dataset
import os

# Load data
data = load_dataset("uitnlp/vietnamese_students_feedback")

# Lấy thư mục hiện tại
current_dir = os.getcwd()
for split in data:
    # Chuyển data sang json
    data[split].to_json(f"{os.path.join(current_dir, 'data', split)}.json", force_ascii=False)
    data[split].to_csv(f"{os.path.join(current_dir, 'data', split)}.csv")