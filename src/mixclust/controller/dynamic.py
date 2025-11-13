# src/mixclust/controller/dynamic.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ControllerConfig:
    # batas adaptasi landmark
    m_max_frac: float = 0.20            # batas atas m terhadap n
    per_cluster_proto: int = 1          # prototipe per klaster (awal)
    k_lnc: int = 50                     # k untuk LNC*
    lnc_weight: float = 0.7             # alpha di LNC* (NC vs Delta)

    # ambang & toleransi switching
    ss_gap_warn: float = 0.05           # |L-Sil - SS_hat| untuk warning kalibrasi
    improve_tol: float = 0.01           # peningkatan minimal (absolute) agar switch diterima
    max_boost_rounds: int = 2           # berapa kali boleh menaikkan m/per_proto
    max_klnc: int = 120                 # batas atas k untuk LNC*
    min_klnc: int = 20                  # batas bawah k untuk LNC*
    boost_m_step_frac: float = 0.10     # jika perlu boost m, naikkan 10% n
    boost_proto_step: int = 1           # jika perlu boost prototipe, +1 per cluster

class DynamicController:
    """
    Orkestrator *heuristik* untuk Phase-B:
    - Terima ringkasan metrik (L-Sil_proto, LNC*, SS_Gower, MAE, n, m, Clusters)
    - Putuskan aksi: 'tune_m', 'tune_proto', 'tune_klnc', 'finalize'
    - Kembalikan dict parameter override untuk iterasi berikutnya
    """
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self._round = 0
        self._boost_count = 0
        self._last_score = None  # pakai SS_Gower sebagai acuan utama

    def decide(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        stats:
          {
            "N": int,
            "Clusters": int,
            "m": int,
            "L-Sil_proto": float,
            "LNC*": float,
            "SS_Gower": float,
            "MAE": float,                # |L-Sil - SS_Gower|
            "k_lnc": int,                # jika tersedia
            "per_proto": int,            # jika tersedia
            "m_max_frac": float,         # jika tersedia
          }

        Returns
        -------
        action: dict
          {
            "action": str,               # 'tune_m' | 'tune_proto' | 'tune_klnc' | 'finalize'
            "overrides": { ... }         # param yang diubah untuk iterasi selanjutnya
            "reason": str
          }
        """
        self._round += 1
        n = int(stats.get("N", 0))
        m = int(stats.get("m", 0))
        C = int(stats.get("Clusters", 0))
        lsil = float(stats.get("L-Sil_proto", 0.0) or 0.0)
        lnc = float(stats.get("LNC*", 0.0) or 0.0)
        ss = float(stats.get("SS_Gower", 0.0) or 0.0)
        mae = float(stats.get("MAE", abs(lsil - ss)))
        k_lnc = int(stats.get("k_lnc", self.cfg.k_lnc))
        per_proto = int(stats.get("per_proto", self.cfg.per_cluster_proto))
        m_max_frac = float(stats.get("m_max_frac", self.cfg.m_max_frac))

        # Hitung kenaikan skor dibanding iterasi sebelumnya
        improved = None
        if self._last_score is not None:
            improved = (ss - self._last_score) >= self.cfg.improve_tol
        self._last_score = ss

        # 1) Jika gap kalibrasi besar → fokus selaraskan L-Sil (proxy) dengan SS
        if mae > self.cfg.ss_gap_warn and self._boost_count < self.cfg.max_boost_rounds:
            # prioritas: tambah per_proto dulu (menginformasikan a(·) lebih baik)
            new_pp = min(per_proto + self.cfg.boost_proto_step, 3)  # batasi wajar
            if new_pp > per_proto:
                self._boost_count += 1
                return {
                    "action": "tune_proto",
                    "overrides": {"per_cluster_proto_if_many": new_pp, "per_proto": new_pp},
                    "reason": f"MAE={mae:.3f} > {self.cfg.ss_gap_warn:.3f}: tambah prototipe per cluster → {new_pp}."
                }

            # lalu tambah m jika masih bisa
            m_cap = int(self.cfg.m_max_frac * n)
            m_step = max(1, int(self.cfg.boost_m_step_frac * n))
            new_m = min(m + m_step, m_cap)
            if new_m > m:
                self._boost_count += 1
                return {
                    "action": "tune_m",
                    "overrides": {"m_target": new_m, "lm_max_frac": self.cfg.m_max_frac},
                    "reason": f"MAE tetap tinggi, boost m: {m} → {new_m} (cap {m_cap})."
                }

        # 2) Jika LNC* rendah → naikkan k_lnc (memperlebar neighborhood untuk stabilitas)
        #    Heuristik: 'rendah' bila LNC* < 0.5 dan skor SS belum membaik
        if (lnc < 0.50) and (improved is False or improved is None):
            new_k = min(k_lnc + 10, self.cfg.max_klnc)
            if new_k > k_lnc:
                return {
                    "action": "tune_klnc",
                    "overrides": {"knn_k_lnc": new_k},
                    "reason": f"LNC*={lnc:.3f} rendah; tingkatkan k_lnc: {k_lnc} → {new_k}."
                }

        # 3) Jika skor membaik tapi masih ada ruang (C banyak dan per_proto masih 1) → tambah per_proto
        if improved and C >= 6 and per_proto < 2:
            return {
                "action": "tune_proto",
                "overrides": {"per_cluster_proto_if_many": per_proto + 1, "per_proto": per_proto + 1},
                "reason": f"Skor membaik & C besar (C={C}). Tambah prototipe: {per_proto} → {per_proto + 1}."
            }

        # 4) Kalau tidak ada aksi krusial → finalize
        return {
            "action": "finalize",
            "overrides": {},
            "reason": "Tidak ada sinyal kuat untuk penyesuaian lebih lanjut; finalize."
        }
