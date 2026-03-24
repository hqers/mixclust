"""mixclust.core — Low-level primitives: Gower distance, features, KNN, landmarks, prototypes."""
from .gower import gower_matrix, gower_to_one_mixed, rerank_gower_from_candidates
from .features import build_features, prepare_mixed_arrays
from .knn_index import KNNIndex
from .landmarks import select_landmarks_cluster_aware, select_landmarks_kcenter
from .prototypes import build_prototypes_by_cluster_gower
from .preprocess import preprocess_mixed_data, prepare_mixed_arrays_no_label
from .adaptive import adaptive_landmark_count
