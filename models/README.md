# Model Storage (.pkl)

Folder ini menyimpan model hasil pelatihan dari kelima algoritma yang diteliti.

## Standar Penamaan File
- `decision_tree_model.pkl` (Laura)
- `random_forest_model.pkl` (Wawa)
- `naive_bayes_model.pkl` (Gisca)
- `knn_model.pkl` (Irfan)
- `logistic_regression_model.pkl` (Rapip)

## Cara Menyimpan Model
```python
import joblib

# Contoh menyimpan model
joblib.dump(model, '../models/nama_model.pkl')
```

Model dengan performa terbaik (*Best Model*) dari komparasi di folder `evaluation/` akan dimuat oleh aplikasi Streamlit di folder `app/`.
