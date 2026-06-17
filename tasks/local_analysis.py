from src.explainability.local_analysis import LocalFeatureAnalyzer


analyzer = LocalFeatureAnalyzer()
analyzer.plot_beeswarm()
analyzer.explain_samples(3)
