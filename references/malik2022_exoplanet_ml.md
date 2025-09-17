Title: Exoplanet detection using machine learning
Citation: Malik, A., Moster, B. P., & Obermeier, C. (2022). Exoplanet detection using machine learning. Monthly Notices of the Royal Astronomical Society, 513(4), 5505-5516. https://doi.org/10.1093/mnras/stab3692

Abstract (summary):
This paper demonstrates a classical ML approach for automated exoplanet candidate vetting using time series feature extraction (TSFRESH) combined with LightGBM classifiers. The methods outperform several deep learning methods on Kepler/TESS, achieving strong AUC and recall. They recommend classical ML for speed and interpretability in scientific vetting.

Key Points:
- Feature extraction from light curves
- Time series analysis (TSFRESH)
- Gradient boosting (LightGBM)
- AUC up to 0.948 (Kepler), recall up to 0.98 (TESS)
- Recommends classical ML for real-world acceptance/vetting

Methods/Results (for benchmarking):
- Dataset(s): Kepler, TESS
- Features/Approach: TSFRESH for feature extraction, LightGBM for classification
- Metrics: AUC up to 0.948 (Kepler), recall up to 0.98 (TESS)

PDF:
- If you have access, download via the publisher link and store as `references/malik2022_exoplanet_ml.pdf`.
