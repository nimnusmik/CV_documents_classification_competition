.
├── README.md
├── configs
│   ├── 20250909
│   │   ├── train_optimized_20250909_1428.yaml
│   │   ├── train_optimized_20250909_1741.yaml
│   │   ├── train_optimized_20250909_1823.yaml
│   │   └── train_optimized_20250909_1922.yaml
│   ├── 20250910
│   │   └── train_optimized_20250910_0032.yaml
│   ├── infer.yaml
│   ├── infer_highperf.yaml
│   ├── infer_multi_model_ensemble.yaml
│   ├── optuna_config.yaml
│   ├── optuna_fast_config.yaml
│   ├── train.yaml
│   ├── train_fast_optimized.yaml
│   ├── train_highperf.yaml
│   └── train_multi_model_ensemble.yaml
├── data
│   └── raw
│       ├── meta.csv
│       ├── sample_submission.csv
│       ├── test
│       ├── train
│       └── train.csv
├── docs
│   ├── configs_폴더_설정 파일_생성
│   │   ├── 최적화_설정_파일_생성_가이드.md
│   │   ├── 추론_설정_파일_생성_가이드.md
│   │   └── 학습_설정_파일_생성_가이드.md
│   ├── 모델
│   │   └── 모델_설정_가이드.md
│   ├── 시스템
│   │   ├── 문제해결_가이드.md
│   │   └── 시각화_시스템_가이드.md
│   ├── 최적화
│   │   ├── GPU_최적화_가이드.md
│   │   └── 시간_최적화_가이드.md
│   └── 파이프라인
│       ├── 전체_파이프라인_가이드.md
│       ├── 추론_파이프라인_가이드.md
│       └── 학습_파이프라인_가이드.md
├── experiments
│   ├── notebook_test
│   │   └── images
│   │       ├── fold_performance.png
│   │       ├── inference_analysis.png
│   │       └── training_history.png
│   ├── optimization
│   │   ├── 20250908
│   │   │   ├── 20250907_0957_swin-sighperf
│   │   │   │   └── best_params_20250907_0957.yaml
│   │   │   ├── 20250907_1729_swin-sighperf
│   │   │   │   └── best_params_20250907_1729.yaml
│   │   │   ├── 20250907_1734_swin-sighperf
│   │   │   │   └── best_params_20250907_1734.yaml
│   │   │   ├── 20250907_1820_swin-sighperf
│   │   │   │   └── best_params_20250907_1820.yaml
│   │   │   ├── 20250907_1825_swin-sighperf
│   │   │   │   └── best_params_20250907_1825.yaml
│   │   │   ├── 20250908_0256_swin-fast-opt
│   │   │   │   └── best_params_20250908_0256.yaml
│   │   │   ├── 20250908_0308_swin-fast-opt
│   │   │   │   └── best_params_20250908_0308.yaml
│   │   │   ├── 20250908_0317_swin-fast-opt
│   │   │   │   └── best_params_20250908_0317.yaml
│   │   │   ├── 20250908_1616_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250908_1616.yaml
│   │   │   │   ├── images
│   │   │   │   │   ├── 01_optimization_progress.png
│   │   │   │   │   ├── 02_cumulative_best.png
│   │   │   │   │   ├── 03_parameter_importance.png
│   │   │   │   │   ├── 04_top_trials_analysis.png
│   │   │   │   │   ├── 05_optimization_summary.png
│   │   │   │   │   └── 06_convergence_analysis.png
│   │   │   │   └── study_20250908_1616.pkl
│   │   │   ├── 20250908_1626_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250908_1626.yaml
│   │   │   │   └── study_20250908_1626.pkl
│   │   │   └── 20250908_1736_multi-model-ensemble-rtx4090
│   │   │       ├── best_params_20250908_1736.yaml
│   │   │       ├── best_params_20250908_1742.yaml
│   │   │       ├── images
│   │   │       │   ├── 01_optimization_progress.png
│   │   │       │   ├── 02_cumulative_best.png
│   │   │       │   ├── 03_parameter_importance.png
│   │   │       │   ├── 04_top_trials_analysis.png
│   │   │       │   ├── 05_optimization_summary.png
│   │   │       │   └── 06_convergence_analysis.png
│   │   │       ├── study_20250908_1736.pkl
│   │   │       └── study_20250908_1742.pkl
│   │   ├── 20250909
│   │   │   ├── 20250909_1258_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250909_1258.yaml
│   │   │   │   ├── images
│   │   │   │   │   ├── 01_optimization_progress.png
│   │   │   │   │   ├── 02_cumulative_best.png
│   │   │   │   │   ├── 03_parameter_importance.png
│   │   │   │   │   ├── 04_top_trials_analysis.png
│   │   │   │   │   ├── 05_optimization_summary.png
│   │   │   │   │   └── 06_convergence_analysis.png
│   │   │   │   └── study_20250909_1258.pkl
│   │   │   ├── 20250909_1259_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250909_1259.yaml
│   │   │   │   ├── images
│   │   │   │   │   ├── 01_optimization_progress.png
│   │   │   │   │   ├── 02_cumulative_best.png
│   │   │   │   │   ├── 03_parameter_importance.png
│   │   │   │   │   ├── 04_top_trials_analysis.png
│   │   │   │   │   ├── 05_optimization_summary.png
│   │   │   │   │   └── 06_convergence_analysis.png
│   │   │   │   └── study_20250909_1259.pkl
│   │   │   ├── 20250909_1317_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250909_1317.yaml
│   │   │   │   ├── images
│   │   │   │   │   ├── 01_optimization_progress.png
│   │   │   │   │   ├── 02_cumulative_best.png
│   │   │   │   │   ├── 03_parameter_importance.png
│   │   │   │   │   ├── 04_top_trials_analysis.png
│   │   │   │   │   ├── 05_optimization_summary.png
│   │   │   │   │   └── 06_convergence_analysis.png
│   │   │   │   └── study_20250909_1317.pkl
│   │   │   ├── 20250909_1326_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250909_1326.yaml
│   │   │   │   ├── images
│   │   │   │   │   ├── 01_optimization_progress.png
│   │   │   │   │   ├── 02_cumulative_best.png
│   │   │   │   │   ├── 03_parameter_importance.png
│   │   │   │   │   ├── 04_top_trials_analysis.png
│   │   │   │   │   ├── 05_optimization_summary.png
│   │   │   │   │   └── 06_convergence_analysis.png
│   │   │   │   └── study_20250909_1326.pkl
│   │   │   ├── 20250909_1428_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250909_1428.yaml
│   │   │   │   ├── images
│   │   │   │   │   ├── 01_optimization_progress.png
│   │   │   │   │   ├── 02_cumulative_best.png
│   │   │   │   │   ├── 03_parameter_importance.png
│   │   │   │   │   ├── 04_top_trials_analysis.png
│   │   │   │   │   ├── 05_optimization_summary.png
│   │   │   │   │   └── 06_convergence_analysis.png
│   │   │   │   └── study_20250909_1428.pkl
│   │   │   ├── 20250909_1741_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250909_1741.yaml
│   │   │   │   ├── images
│   │   │   │   │   ├── 01_optimization_progress.png
│   │   │   │   │   ├── 02_cumulative_best.png
│   │   │   │   │   ├── 03_parameter_importance.png
│   │   │   │   │   ├── 04_top_trials_analysis.png
│   │   │   │   │   ├── 05_optimization_summary.png
│   │   │   │   │   └── 06_convergence_analysis.png
│   │   │   │   └── study_20250909_1741.pkl
│   │   │   ├── 20250909_1823_multi-model-ensemble-rtx4090
│   │   │   │   ├── best_params_20250909_1823.yaml
│   │   │   │   ├── images
│   │   │   │   │   ├── 01_optimization_progress.png
│   │   │   │   │   ├── 02_cumulative_best.png
│   │   │   │   │   ├── 03_parameter_importance.png
│   │   │   │   │   ├── 04_top_trials_analysis.png
│   │   │   │   │   ├── 05_optimization_summary.png
│   │   │   │   │   └── 06_convergence_analysis.png
│   │   │   │   └── study_20250909_1823.pkl
│   │   │   └── 20250909_1922_swin_base_384
│   │   │       ├── best_params_20250909_1922.yaml
│   │   │       ├── images
│   │   │       │   ├── 01_optimization_progress.png
│   │   │       │   ├── 02_cumulative_best.png
│   │   │       │   ├── 03_parameter_importance.png
│   │   │       │   ├── 04_top_trials_analysis.png
│   │   │       │   ├── 05_optimization_summary.png
│   │   │       │   └── 06_convergence_analysis.png
│   │   │       └── study_20250909_1922.pkl
│   │   ├── 20250910
│   │   │   └── 20250910_0032_multi-model-ensemble-rtx4090
│   │   │       ├── best_params_20250910_0032.yaml
│   │   │       ├── images
│   │   │       │   ├── 01_optimization_progress.png
│   │   │       │   ├── 02_cumulative_best.png
│   │   │       │   ├── 03_parameter_importance.png
│   │   │       │   ├── 04_top_trials_analysis.png
│   │   │       │   ├── 05_optimization_summary.png
│   │   │       │   └── 06_convergence_analysis.png
│   │   │       └── study_20250910_0032.pkl
│   │   └── lastest-optimization
│   │       ├── best_params_20250910_0032.yaml
│   │       ├── images
│   │       │   ├── 01_optimization_progress.png
│   │       │   ├── 02_cumulative_best.png
│   │       │   ├── 03_parameter_importance.png
│   │       │   ├── 04_top_trials_analysis.png
│   │       │   ├── 05_optimization_summary.png
│   │       │   └── 06_convergence_analysis.png
│   │       └── study_20250910_0032.pkl
│   ├── test_viz
│   │   ├── images
│   │   │   ├── 01_class_distribution.png
│   │   │   ├── 01_fold_f1_performance.png
│   │   │   ├── 02_confidence_distribution.png
│   │   │   ├── 02_fold_accuracy_comparison.png
│   │   │   ├── 03_class_confidence_comparison.png
│   │   │   ├── 03_f1_vs_accuracy_scatter.png
│   │   │   ├── 04_confidence_bins.png
│   │   │   ├── 04_performance_distribution.png
│   │   │   ├── 05_confidence_boxplot.png
│   │   │   ├── 05_performance_statistics.png
│   │   │   ├── 06_inference_summary.png
│   │   │   ├── 06_training_history.png
│   │   │   ├── 07_loss_comparison.png
│   │   │   ├── 07_probability_heatmap.png
│   │   │   ├── fold_performance.png
│   │   │   ├── inference_analysis.png
│   │   │   ├── korean_font_test.png
│   │   │   └── training_history.png
│   │   └── test_model
│   │       └── images
│   │           └── fold_performance_comparison.png
│   └── train
│       ├── 20250907
│       │   ├── efficientnet_b3
│       │   │   └── fold_results.yaml
│       │   ├── efficientnet_b3_20250907_1753
│       │   │   ├── ckpt
│       │   │   │   ├── best_fold0.pth
│       │   │   │   ├── best_fold1.pth
│       │   │   │   ├── best_fold2.pth
│       │   │   │   ├── best_fold3.pth
│       │   │   │   └── best_fold4.pth
│       │   │   ├── config.yaml
│       │   │   ├── metrics.jsonl
│       │   │   └── oof
│       │   │       ├── oof_logits.npy
│       │   │       └── oof_targets.npy
│       │   ├── swin-sighperf
│       │   │   ├── ckpt
│       │   │   │   ├── best_model_fold_1.pth
│       │   │   │   ├── best_model_fold_2.pth
│       │   │   │   ├── best_model_fold_3.pth
│       │   │   │   ├── best_model_fold_4.pth
│       │   │   │   └── best_model_fold_5.pth
│       │   │   └── fold_results.yaml
│       │   ├── swin-sighperf_20250907_1803
│       │   │   └── ckpt
│       │   │       └── best_model_fold_1.pth
│       │   ├── swin-sighperf_20250907_1820
│       │   │   └── ckpt
│       │   └── swin-sighperf_20250907_1825
│       │       ├── ckpt
│       │       │   ├── best_model_fold_1.pth
│       │       │   ├── best_model_fold_2.pth
│       │       │   ├── best_model_fold_3.pth
│       │       │   ├── best_model_fold_4.pth
│       │       │   └── best_model_fold_5.pth
│       │       └── fold_results.yaml
│       ├── 20250908
│       │   ├── efficientnet_b3_20250908_0313
│       │   │   ├── ckpt
│       │   │   │   ├── best_fold0.pth
│       │   │   │   ├── best_fold1.pth
│       │   │   │   ├── best_fold2.pth
│       │   │   │   └── best_fold3.pth
│       │   │   ├── config.yaml
│       │   │   └── metrics.jsonl
│       │   ├── efficientnet_b3_20250908_0333
│       │   │   ├── ckpt
│       │   │   │   ├── best_fold0.pth
│       │   │   │   ├── best_fold1.pth
│       │   │   │   ├── best_fold2.pth
│       │   │   │   ├── best_fold3.pth
│       │   │   │   └── best_fold4.pth
│       │   │   ├── config.yaml
│       │   │   ├── metrics.jsonl
│       │   │   └── oof
│       │   │       ├── oof_logits.npy
│       │   │       └── oof_targets.npy
│       │   ├── efficientnet_b3_20250908_0434
│       │   │   ├── ckpt
│       │   │   │   ├── best_fold0.pth
│       │   │   │   ├── best_fold1.pth
│       │   │   │   ├── best_fold2.pth
│       │   │   │   ├── best_fold3.pth
│       │   │   │   └── best_fold4.pth
│       │   │   ├── config.yaml
│       │   │   ├── metrics.jsonl
│       │   │   └── oof
│       │   │       ├── oof_logits.npy
│       │   │       └── oof_targets.npy
│       │   ├── multi-model-ensemble-rtx4090_20250908_1155
│       │   │   └── ckpt
│       │   ├── multi-model-ensemble-rtx4090_20250908_1616
│       │   │   └── ckpt
│       │   ├── multi-model-ensemble-rtx4090_20250908_1617
│       │   │   ├── ckpt
│       │   │   └── config.yaml
│       │   ├── multi-model-ensemble-rtx4090_20250908_1618
│       │   │   ├── ckpt
│       │   │   └── config.yaml
│       │   ├── multi-model-ensemble-rtx4090_20250908_1619
│       │   │   ├── ckpt
│       │   │   └── config.yaml
│       │   ├── multi-model-ensemble-rtx4090_20250908_1622
│       │   │   ├── ckpt
│       │   │   │   ├── best_fold0.pth
│       │   │   │   ├── best_fold1.pth
│       │   │   │   └── best_fold2.pth
│       │   │   ├── config.yaml
│       │   │   └── metrics.jsonl
│       │   ├── multi-model-ensemble-rtx4090_20250908_1626
│       │   │   └── ckpt
│       │   │       ├── best_model_fold_1.pth
│       │   │       └── best_model_fold_2.pth
│       │   ├── multi-model-ensemble-rtx4090_20250908_1736
│       │   │   └── ckpt
│       │   ├── multi-model-ensemble-rtx4090_20250908_1742
│       │   │   └── ckpt
│       │   ├── swin-fast-opt_20250908_0256
│       │   │   └── ckpt
│       │   ├── swin-fast-opt_20250908_0308
│       │   │   └── ckpt
│       │   ├── swin-fast-opt_20250908_0317
│       │   │   └── ckpt
│       │   │       └── best_model_fold_1.pth
│       │   ├── swin-sighperf_20250908_0318
│       │   │   └── ckpt
│       │   ├── swin-sighperf_20250908_0338
│       │   │   └── ckpt
│       │   │       └── best_model_fold_1.pth
│       │   ├── swin-sighperf_20250908_0601
│       │   │   └── ckpt
│       │   ├── swin-sighperf_20250908_0705
│       │   │   └── ckpt
│       │   └── swin-sighperf_20250908_1618
│       │       ├── ckpt
│       │       └── config.yaml
│       ├── 20250909
│       │   ├── 20250909_1258_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   ├── 20250909_1259_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   ├── 20250909_1308_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   ├── 20250909_1309_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   ├── 20250909_1317_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   │       ├── best_model_fold_1.pth
│       │   │       ├── best_model_fold_2.pth
│       │   │       └── best_model_fold_3.pth
│       │   ├── 20250909_1326_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   │       ├── best_model_fold_1.pth
│       │   │       └── best_model_fold_2.pth
│       │   ├── 20250909_1428_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   │       ├── best_model_fold_1.pth
│       │   │       ├── best_model_fold_2.pth
│       │   │       ├── best_model_fold_3.pth
│       │   │       └── best_model_fold_4.pth
│       │   ├── 20250909_1741_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   │       └── best_model_fold_1.pth
│       │   ├── 20250909_1823_multi-model-ensemble-rtx4090
│       │   │   └── ckpt
│       │   │       ├── best_model_fold_1.pth
│       │   │       ├── best_model_fold_2.pth
│       │   │       └── best_model_fold_3.pth
│       │   ├── 20250909_1922_swin_base_384
│       │   │   └── ckpt
│       │   │       └── best_model_fold_1.pth
│       │   └── swin-sighperf
│       │       └── ckpt
│       │           └── best_model_fold_1.pth
│       ├── 20250910
│       │   └── 20250910_0032_multi-model-ensemble-rtx4090
│       │       └── ckpt
│       │           ├── best_model_fold_1.pth
│       │           ├── best_model_fold_2.pth
│       │           ├── best_model_fold_3.pth
│       │           └── best_model_fold_4.pth
│       └── lastest-train
│           ├── 20250909_1741_multi-model-ensemble-rtx4090
│           │   └── ckpt
│           │       └── best_model_fold_1.pth
│           ├── 20250909_1823_multi-model-ensemble-rtx4090
│           │   └── ckpt
│           │       ├── best_model_fold_1.pth
│           │       ├── best_model_fold_2.pth
│           │       └── best_model_fold_3.pth
│           ├── 20250909_1922_swin_base_384
│           │   └── ckpt
│           │       └── best_model_fold_1.pth
│           ├── 20250910_0032_multi-model-ensemble-rtx4090
│           │   └── ckpt
│           │       ├── best_model_fold_1.pth
│           │       ├── best_model_fold_2.pth
│           │       ├── best_model_fold_3.pth
│           │       └── best_model_fold_4.pth
│           ├── efficientnet_b3
│           │   ├── ckpt
│           │   │   ├── best_fold0.pth
│           │   │   ├── best_fold1.pth
│           │   │   ├── best_fold2.pth
│           │   │   ├── best_fold3.pth
│           │   │   └── best_fold4.pth
│           │   ├── config.yaml
│           │   ├── metrics.jsonl
│           │   └── oof
│           │       ├── oof_logits.npy
│           │       └── oof_targets.npy
│           └── swin-sighperf
│               ├── ckpt
│               └── fold_results.yaml
├── font
│   └── NanumGothic.ttf
├── logs
│   ├── 20250904
│   │   ├── infer
│   │   │   └── infer_20250904-1721_v087.log
│   │   ├── optimization
│   │   ├── pipeline
│   │   └── train
│   │       └── train_20250904-1715_v087-ef727c.log
│   ├── 20250906
│   │   ├── infer
│   │   ├── optimization
│   │   ├── pipeline
│   │   └── train
│   ├── 20250907
│   │   ├── infer
│   │   │   ├── infer_highperf_20250907_0021.log
│   │   │   ├── infer_highperf_20250907_0054.log
│   │   │   ├── infer_highperf_20250907_0709.log
│   │   │   ├── infer_highperf_20250907_0813.log
│   │   │   ├── infer_highperf_20250907_0831.log
│   │   │   ├── infer_highperf_20250907_0832.log
│   │   │   ├── infer_highperf_20250907_0927.log
│   │   │   ├── infer_highperf_20250907_1018.log
│   │   │   └── infer_highperf_20250907_1720.log
│   │   ├── optimization
│   │   │   ├── optuna_20250907_0957.log
│   │   │   ├── optuna_20250907_1729.log
│   │   │   ├── optuna_20250907_1734.log
│   │   │   ├── optuna_20250907_1819.log
│   │   │   └── optuna_20250907_1824.log
│   │   ├── pipeline
│   │   │   ├── full_pipeline_20250907_0016.log
│   │   │   ├── full_pipeline_20250907_0035.log
│   │   │   ├── full_pipeline_20250907_0709.log
│   │   │   ├── full_pipeline_20250907_0813.log
│   │   │   ├── full_pipeline_20250907_0831.log
│   │   │   ├── full_pipeline_20250907_0832.log
│   │   │   ├── full_pipeline_20250907_0906.log
│   │   │   ├── full_pipeline_20250907_0957.log
│   │   │   ├── full_pipeline_20250907_1702.log
│   │   │   ├── full_pipeline_20250907_1729.log
│   │   │   ├── full_pipeline_20250907_1734.log
│   │   │   ├── full_pipeline_20250907_1820.log
│   │   │   └── full_pipeline_20250907_1825.log
│   │   └── train
│   │       ├── train_20250907-0016_efficientnet_b3-43c495.log
│   │       ├── train_20250907-1753_efficientnet_b3.log
│   │       ├── train_highperf_20250907-0035_swin-sighperf-d03ed5.log
│   │       ├── train_highperf_20250907-0906_swin-sighperf-4e8abb.log
│   │       ├── train_highperf_20250907-1702_swin-sighperf-3fc6bc.log
│   │       └── train_highperf_20250907-1825_swin-sighperf-d12239_basic_augmentation.log
│   ├── 20250908
│   │   ├── infer
│   │   │   ├── infer_20250908-0319_efficientnet-inference.log
│   │   │   ├── infer_20250908-0320_efficientnet-inference.log
│   │   │   ├── infer_20250908-0322_efficientnet-inference.log
│   │   │   ├── infer_20250908-0323_efficientnet-inference.log
│   │   │   ├── infer_20250908-0331_efficientnet-inference.log
│   │   │   ├── infer_20250908-0333_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0405_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0407_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0408_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0433_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0439_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0445_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0502_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0505_efficientnet-basic-inference.log
│   │   │   ├── infer_20250908-0515_efficientnet-basic-inference.log
│   │   │   ├── infer_highperf_20250908_0212.log
│   │   │   ├── infer_highperf_20250908_0420.log
│   │   │   └── infer_highperf_20250908_0516.log
│   │   ├── optimization
│   │   │   ├── optuna_20250908_0253.log
│   │   │   ├── optuna_20250908_0306.log
│   │   │   ├── optuna_20250908_0317.log
│   │   │   ├── optuna_20250908_1616.log
│   │   │   ├── optuna_20250908_1625.log
│   │   │   ├── optuna_20250908_1735.log
│   │   │   └── optuna_20250908_1741.log
│   │   ├── pipeline
│   │   │   ├── full_pipeline_20250908_0256.log
│   │   │   ├── full_pipeline_20250908_0308.log
│   │   │   ├── full_pipeline_20250908_0317.log
│   │   │   ├── full_pipeline_20250908_0318.log
│   │   │   ├── full_pipeline_20250908_1155.log
│   │   │   ├── full_pipeline_20250908_1616.log
│   │   │   ├── full_pipeline_20250908_1626.log
│   │   │   ├── full_pipeline_20250908_1736.log
│   │   │   └── full_pipeline_20250908_1742.log
│   │   └── train
│   │       ├── train_20250908-0313_efficientnet_b3_basic_augmentation.log
│   │       ├── train_20250908-0333_efficientnet_b3_basic_augmentation.log
│   │       ├── train_20250908-0434_efficientnet_b3_basic_augmentation.log
│   │       ├── train_20250908-1617_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_20250908-1618_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_20250908-1618_swin-sighperf_advanced_augmentation.log
│   │       ├── train_20250908-1619_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_20250908-1622_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_fast_opt_20250908-0256_swin-fast-opt-59cbe4_advanced_augmentation.log
│   │       ├── train_fast_opt_20250908-0308_swin-fast-opt-5d94b2_advanced_augmentation.log
│   │       ├── train_fast_opt_20250908-0317_swin-fast-opt-620daa_advanced_augmentation.log
│   │       ├── train_highperf_20250908-0318_swin-sighperf-c6e47c_advanced_augmentation.log
│   │       ├── train_highperf_20250908-0338_swin-sighperf-043a62_advanced_augmentation.log
│   │       ├── train_highperf_20250908-0601_swin-sighperf-bfb9c9_advanced_augmentation.log
│   │       ├── train_highperf_20250908-0705_swin-sighperf-fae9c4_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250908-1155_multi-model-ensemble-rtx4090-3620dd_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250908-1616_multi-model-ensemble-rtx4090-d992d6_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250908-1626_multi-model-ensemble-rtx4090-2227a8_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250908-1736_multi-model-ensemble-rtx4090-3ad51b_advanced_augmentation.log
│   │       └── train_multi_ensemble_20250908-1742_multi-model-ensemble-rtx4090-da680a_advanced_augmentation.log
│   ├── 20250909
│   │   ├── optimization
│   │   │   ├── optuna_20250909_1257.log
│   │   │   ├── optuna_20250909_1258.log
│   │   │   ├── optuna_20250909_1316.log
│   │   │   ├── optuna_20250909_1325.log
│   │   │   ├── optuna_20250909_1427.log
│   │   │   ├── optuna_20250909_1740.log
│   │   │   ├── optuna_20250909_1822.log
│   │   │   └── optuna_20250909_1921.log
│   │   ├── pipeline
│   │   │   ├── full_pipeline_20250909_1258.log
│   │   │   ├── full_pipeline_20250909_1259.log
│   │   │   ├── full_pipeline_20250909_1317.log
│   │   │   ├── full_pipeline_20250909_1326.log
│   │   │   ├── full_pipeline_20250909_1428.log
│   │   │   ├── full_pipeline_20250909_1741.log
│   │   │   ├── full_pipeline_20250909_1823.log
│   │   │   └── full_pipeline_20250909_1922.log
│   │   └── train
│   │       ├── train_highperf_20250909-1125_swin-sighperf-723ff5.log
│   │       ├── train_highperf_20250909-1131_swin-sighperf-d25270.log
│   │       ├── train_highperf_20250909-1922_swin_base_384_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250909-1258_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250909-1259_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250909-1308_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250909-1309_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250909-1317_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250909-1326_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250909-1428_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       ├── train_multi_ensemble_20250909-1741_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   │       └── train_multi_ensemble_20250909-1823_multi-model-ensemble-rtx4090_advanced_augmentation.log
│   └── 20250910
│       ├── optimization
│       │   └── optuna_20250910_0031.log
│       ├── pipeline
│       │   └── full_pipeline_20250910_0032.log
│       └── train
│           └── train_multi_ensemble_20250910-0032_multi-model-ensemble-rtx4090_advanced_augmentation.log
├── notebooks
│   ├── base
│   │   └── baseline_code.ipynb
│   ├── modular
│   │   ├── results_comparison
│   │   │   ├── submission_results_date_comparison
│   │   │   │   ├── 20250907_065923
│   │   │   │   │   ├── data
│   │   │   │   │   ├── images
│   │   │   │   │   │   ├── korean_font_test.png
│   │   │   │   │   │   ├── prediction_differences.png
│   │   │   │   │   │   ├── prediction_distribution.png
│   │   │   │   │   │   └── temporal_metrics.png
│   │   │   │   │   ├── logs
│   │   │   │   │   │   └── submission_results_date_comparison_20250907_065923.log
│   │   │   │   │   └── results
│   │   │   │   ├── 20250907_201413
│   │   │   │   │   ├── data
│   │   │   │   │   ├── images
│   │   │   │   │   │   ├── korean_font_test.png
│   │   │   │   │   │   ├── prediction_differences.png
│   │   │   │   │   │   ├── prediction_distribution.png
│   │   │   │   │   │   └── temporal_metrics.png
│   │   │   │   │   ├── logs
│   │   │   │   │   │   └── submission_results_date_comparison_20250907_201413.log
│   │   │   │   │   └── results
│   │   │   │   └── 20250908_091840
│   │   │   │       ├── data
│   │   │   │       ├── images
│   │   │   │       │   ├── korean_font_test.png
│   │   │   │       │   ├── prediction_differences.png
│   │   │   │       │   ├── prediction_distribution.png
│   │   │   │       │   └── temporal_metrics.png
│   │   │   │       ├── logs
│   │   │   │       │   └── submission_results_date_comparison_20250908_091840.log
│   │   │   │       └── results
│   │   │   ├── submission_results_date_comparison.ipynb
│   │   │   ├── swin_vs_efficientnet
│   │   │   │   ├── 20250907_062744
│   │   │   │   │   ├── data
│   │   │   │   │   │   ├── differences.csv
│   │   │   │   │   │   ├── differences_meta.json
│   │   │   │   │   │   ├── full_comparison.csv
│   │   │   │   │   │   └── full_comparison_meta.json
│   │   │   │   │   ├── images
│   │   │   │   │   │   ├── difference_analysis.png
│   │   │   │   │   │   ├── difference_analysis_meta.json
│   │   │   │   │   │   ├── korean_font_test.png
│   │   │   │   │   │   ├── korean_font_test_meta.json
│   │   │   │   │   │   ├── overall_comparison.png
│   │   │   │   │   │   └── overall_comparison_meta.json
│   │   │   │   │   ├── logs
│   │   │   │   │   │   └── swin_vs_efficientnet_20250907_062744.log
│   │   │   │   │   ├── results
│   │   │   │   │   │   ├── basic_metrics_metrics.json
│   │   │   │   │   │   └── comparison_result.json
│   │   │   │   │   └── summary.json
│   │   │   │   └── 20250907_064551
│   │   │   │       ├── data
│   │   │   │       │   ├── differences.csv
│   │   │   │       │   ├── differences_meta.json
│   │   │   │       │   ├── full_comparison.csv
│   │   │   │       │   └── full_comparison_meta.json
│   │   │   │       ├── images
│   │   │   │       │   ├── difference_analysis.png
│   │   │   │       │   ├── difference_analysis_meta.json
│   │   │   │       │   ├── overall_comparison.png
│   │   │   │       │   └── overall_comparison_meta.json
│   │   │   │       ├── logs
│   │   │   │       │   └── swin_vs_efficientnet_20250907_064551.log
│   │   │   │       ├── results
│   │   │   │       │   ├── basic_metrics_metrics.json
│   │   │   │       │   └── comparison_result.json
│   │   │   │       └── summary.json
│   │   │   ├── swin_vs_swin_advanced_augmentation
│   │   │   │   └── 20250907_093456
│   │   │   │       ├── data
│   │   │   │       │   ├── differences.csv
│   │   │   │       │   ├── differences_meta.json
│   │   │   │       │   ├── full_comparison.csv
│   │   │   │       │   └── full_comparison_meta.json
│   │   │   │       ├── images
│   │   │   │       │   ├── difference_analysis.png
│   │   │   │       │   ├── difference_analysis_meta.json
│   │   │   │       │   ├── overall_comparison.png
│   │   │   │       │   └── overall_comparison_meta.json
│   │   │   │       ├── logs
│   │   │   │       │   └── swin_vs_swin_advanced_augmentation_20250907_093456.log
│   │   │   │       ├── results
│   │   │   │       │   ├── basic_metrics_metrics.json
│   │   │   │       │   └── comparison_result.json
│   │   │   │       └── summary.json
│   │   │   ├── swin_vs_swin_base_384_ensemble_tta_basic_augmentation
│   │   │   │   └── 20250908_022117
│   │   │   │       ├── data
│   │   │   │       │   ├── differences.csv
│   │   │   │       │   ├── differences_meta.json
│   │   │   │       │   ├── full_comparison.csv
│   │   │   │       │   └── full_comparison_meta.json
│   │   │   │       ├── images
│   │   │   │       │   ├── difference_analysis.png
│   │   │   │       │   ├── difference_analysis_meta.json
│   │   │   │       │   ├── overall_comparison.png
│   │   │   │       │   └── overall_comparison_meta.json
│   │   │   │       ├── logs
│   │   │   │       │   └── swin_vs_swin_base_384_ensemble_tta_basic_augmentation_20250908_022117.log
│   │   │   │       ├── results
│   │   │   │       │   ├── basic_metrics_metrics.json
│   │   │   │       │   └── comparison_result.json
│   │   │   │       └── summary.json
│   │   │   └── two_files_detailed_comparison.ipynb
│   │   └── unit_tests
│   │       ├── 01_highperf_dataset
│   │       │   ├── 20250907_071613
│   │       │   │   ├── data
│   │       │   │   │   ├── class_distribution.csv
│   │       │   │   │   └── class_distribution_meta.json
│   │       │   │   ├── images
│   │       │   │   │   ├── 고품질_샘플_라벨_16.png
│   │       │   │   │   └── 고품질_샘플_라벨_16_meta.json
│   │       │   │   ├── logs
│   │       │   │   │   └── 01_highperf_dataset_20250907_071613.log
│   │       │   │   ├── results
│   │       │   │   │   └── basic_test_result.json
│   │       │   │   └── summary.json
│   │       │   └── 20250907_073910
│   │       │       ├── data
│   │       │       │   ├── class_distribution.csv
│   │       │       │   └── class_distribution_meta.json
│   │       │       ├── images
│   │       │       │   ├── 고품질_샘플_라벨_16.png
│   │       │       │   └── 고품질_샘플_라벨_16_meta.json
│   │       │       ├── logs
│   │       │       │   └── 01_highperf_dataset_20250907_073910.log
│   │       │       ├── results
│   │       │       │   └── basic_test_result_result.json
│   │       │       └── summary.json
│   │       ├── 01_highperf_dataset.ipynb
│   │       ├── 02_mixup_augmentation
│   │       │   ├── 20250907_070747
│   │       │   │   ├── data
│   │       │   │   ├── images
│   │       │   │   ├── logs
│   │       │   │   │   └── 02_mixup_augmentation_20250907_070747.log
│   │       │   │   └── results
│   │       │   └── 20250907_074452
│   │       │       ├── data
│   │       │       ├── images
│   │       │       │   ├── mixup_샘플_1_라벨_16.png
│   │       │       │   ├── mixup_샘플_1_라벨_16_meta.json
│   │       │       │   ├── mixup_샘플_2_라벨_10.png
│   │       │       │   ├── mixup_샘플_2_라벨_10_meta.json
│   │       │       │   ├── mixup_샘플_3_라벨_10.png
│   │       │       │   ├── mixup_샘플_3_라벨_10_meta.json
│   │       │       │   ├── mixup_샘플_4_라벨_4.png
│   │       │       │   ├── mixup_샘플_4_라벨_4_meta.json
│   │       │       │   ├── mixup_샘플_5_라벨_16.png
│   │       │       │   └── mixup_샘플_5_라벨_16_meta.json
│   │       │       ├── logs
│   │       │       │   └── 02_mixup_augmentation_20250907_074452.log
│   │       │       ├── results
│   │       │       │   └── mixup_test_result.json
│   │       │       └── summary.json
│   │       ├── 02_mixup_augmentation.ipynb
│   │       ├── 03_swin_model_test
│   │       │   ├── 20250907_070837
│   │       │   │   ├── data
│   │       │   │   ├── images
│   │       │   │   ├── logs
│   │       │   │   │   └── 03_swin_model_test_20250907_070837.log
│   │       │   │   └── results
│   │       │   └── 20250907_080613
│   │       │       ├── data
│   │       │       ├── images
│   │       │       ├── logs
│   │       │       │   └── 03_swin_model_test_20250907_080613.log
│   │       │       ├── results
│   │       │       │   └── swin_model_test_result.json
│   │       │       └── summary.json
│   │       ├── 03_swin_model_test.ipynb
│   │       ├── 04_pipeline_integration
│   │       │   └── 20250907_083243
│   │       │       ├── data
│   │       │       ├── images
│   │       │       ├── logs
│   │       │       │   └── 04_pipeline_integration_20250907_083243.log
│   │       │       ├── results
│   │       │       │   └── pipeline_integration_test_result_result.json
│   │       │       └── summary.json
│   │       ├── 04_pipeline_integration.ipynb
│   │       ├── 05_wandb_integration
│   │       │   └── 20250907_083320
│   │       │       ├── data
│   │       │       ├── images
│   │       │       ├── logs
│   │       │       │   └── 05_wandb_integration_20250907_083320.log
│   │       │       ├── results
│   │       │       │   └── wandb_integration_test_result_result.json
│   │       │       └── summary.json
│   │       ├── 05_wandb_integration.ipynb
│   │       ├── 06_gpu_auto_check
│   │       │   └── 20250907_083513
│   │       │       ├── data
│   │       │       ├── images
│   │       │       ├── logs
│   │       │       │   └── 06_gpu_auto_check_20250907_083513.log
│   │       │       ├── results
│   │       │       │   └── gpu_check_test_result_result.json
│   │       │       └── summary.json
│   │       ├── 06_gpu_auto_check.ipynb
│   │       └── 07_한글폰트_시각화_검증.ipynb
│   └── team
│       ├── CHH
│       │   ├── 00_data_schema_and_integrity.ipynb
│       │   ├── 01_transforms_gallery.ipynb
│       │   ├── 02_dataloader_smoke_test.ipynb
│       │   ├── 03_weighted_sampler_sanity.ipynb
│       │   └── 04_split_and_leakage_checks.ipynb
│       ├── KBH
│       │   ├── 0.87_main.ipynb
│       │   ├── F1_0.933_convnext.ipynb
│       │   ├── F1_0.934_swinb.ipynb
│       │   ├── F1_0_934_swinb
│       │   │   └── 20250908_095705
│       │   │       ├── data
│       │   │       ├── images
│       │   │       ├── logs
│       │   │       │   └── F1_0_934_swinb_20250908_095705.log
│       │   │       └── results
│       │   └── main.ipynb
│       └── KSM
│           ├── base_line_wandb
│           │   └── 20250908_100101
│           │       ├── data
│           │       ├── images
│           │       ├── logs
│           │       │   └── base_line_wandb_20250908_100101.log
│           │       └── results
│           ├── base_line_wandb.ipynb
│           └── team_guide.md
├── requirements.txt
├── scripts
│   ├── monitor_training.sh
│   ├── reorganize_experiments.py
│   ├── run_fast_training.sh
│   ├── run_highperf_training.sh
│   ├── setup_visualization.sh
│   ├── test_korean_font.py
│   ├── test_visualization.py
│   └── update_inference_date.sh
├── src
│   ├── __init__.py
│   │   ├── __init__.cpython-311.pyc
│   │   └── __init__.cpython-312.pyc
│   ├── calibration
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── calibration_utils.cpython-311.pyc
│   │   │   └── temperature_scaling.cpython-311.pyc
│   │   ├── calibration_utils.py
│   │   └── temperature_scaling.py
│   ├── data
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── dataset.cpython-311.pyc
│   │   │   └── transforms.cpython-311.pyc
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── inference
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── infer.cpython-311.pyc
│   │   │   ├── infer_calibrated.cpython-311.pyc
│   │   │   ├── infer_highperf.cpython-311.pyc
│   │   │   └── infer_main.cpython-311.pyc
│   │   ├── infer.py
│   │   ├── infer_calibrated.py
│   │   ├── infer_highperf.py
│   │   └── infer_main.py
│   ├── logging
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── logger.cpython-311.pyc
│   │   │   ├── notebook_logger.cpython-311.pyc
│   │   │   ├── unit_test_logger.cpython-311.pyc
│   │   │   └── wandb_logger.cpython-311.pyc
│   │   ├── logger.py
│   │   ├── notebook_logger.py
│   │   └── wandb_logger.py
│   ├── metrics
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── f1.cpython-311.pyc
│   │   └── f1.py
│   ├── models
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   ├── build.cpython-311.pyc
│   │   │   └── build.cpython-312.pyc
│   │   └── build.py
│   ├── optimization
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── hyperopt_utils.cpython-311.pyc
│   │   │   └── optuna_tuner.cpython-311.pyc
│   │   ├── hyperopt_utils.py
│   │   └── optuna_tuner.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── full_pipeline.cpython-311.pyc
│   │   └── full_pipeline.py
│   ├── training
│   │   ├── __init__.py
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── train.cpython-311.pyc
│   │   │   ├── train_highperf.cpython-311.pyc
│   │   │   └── train_main.cpython-311.pyc
│   │   ├── train.py
│   │   ├── train_highperf.py
│   │   └── train_main.py
│   └── utils
│       ├── __init__.py
│       │   ├── __init__.cpython-311.pyc
│       │   ├── __init__.cpython-312.pyc
│       │   ├── common.cpython-311.pyc
│       │   ├── common.cpython-312.pyc
│       │   ├── logger.cpython-311.pyc
│       │   ├── output_manager.cpython-311.pyc
│       │   ├── seed.cpython-311.pyc
│       │   ├── seed.cpython-312.pyc
│       │   ├── simple_visualization.cpython-311.pyc
│       │   ├── team_gpu_check.cpython-311.pyc
│       │   ├── unit_test_logger.cpython-311.pyc
│       │   ├── visualization.cpython-311.pyc
│       │   └── wandb_logger.cpython-311.pyc
│       ├── code_management
│       │   ├── __init__.py
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   ├── check_code_blocks.cpython-311.pyc
│       │   │   ├── fix_all_code_blocks.cpython-311.pyc
│       │   │   ├── fix_code_blocks.cpython-311.pyc
│       │   │   └── smart_fix_code_blocks.cpython-311.pyc
│       │   ├── check_code_blocks.py
│       │   ├── fix_all_code_blocks.py
│       │   ├── fix_code_blocks.py
│       │   └── smart_fix_code_blocks.py
│       ├── config
│       │   ├── __init__.py
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   ├── seed.cpython-311.pyc
│       │   │   └── update_config_dates.cpython-311.pyc
│       │   ├── seed.py
│       │   └── update_config_dates.py
│       ├── core
│       │   ├── __init__.py
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   └── common.cpython-311.pyc
│       │   └── common.py
│       ├── gpu_optimization
│       │   ├── __init__.py
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   ├── auto_batch_size.cpython-311.pyc
│       │   │   ├── auto_batch_size_broken.cpython-311.pyc
│       │   │   ├── auto_batch_size_fixed.cpython-311.pyc
│       │   │   └── team_gpu_check.cpython-311.pyc
│       │   ├── auto_batch_size.py
│       │   ├── auto_batch_size_broken.py
│       │   ├── auto_batch_size_fixed.py
│       │   └── team_gpu_check.py
│       └── visualizations
│           ├── __init__.py
│           │   ├── __init__.cpython-311.pyc
│           │   ├── base_visualizer.cpython-311.pyc
│           │   ├── inference_viz.cpython-311.pyc
│           │   ├── optimization_viz.cpython-311.pyc
│           │   ├── output_manager.cpython-311.pyc
│           │   ├── simple_visualization.cpython-311.pyc
│           │   └── training_viz.cpython-311.pyc
│           ├── base_visualizer.py
│           ├── inference_viz.py
│           ├── optimization_viz.py
│           ├── output_manager.py
│           ├── simple_visualization.py
│           └── training_viz.py
├── submissions
│   ├── 20250903
│   │   └── baseline_code_pred.csv
│   ├── 20250904
│   │   └── infer.csv
│   ├── 20250905
│   │   ├── efficientnet_b3_ensemble_20250905_1345.csv
│   │   └── swin-sighperf_ensemble_20250905_1522.csv
│   ├── 20250906
│   │   ├── efficientnet_b3_ensemble_20250906_2157.csv
│   │   └── swin-sighperf_ensemble_20250906_2213.csv
│   ├── 20250907
│   │   ├── efficientnet_b3_ensemble_20250907_0016.csv
│   │   ├── swin-sighperf_ensemble_20250907_0035.csv
│   │   ├── swin-sighperf_ensemble_20250907_0709.csv
│   │   └── swin-sighperf_ensemble_20250907_0906.csv
│   └── 20250908
│       ├── 20250908_0217_swin_base_384_ensemble_tta_basic_augmentation.csv
│       ├── 20250908_0322_efficientnet_b3_tta_basic_augmentation.csv
│       ├── 20250908_0408_efficientnet_b3_tta_basic_augmentation.csv
│       ├── 20250908_0424_swin_base_384_ensemble_tta_basic_augmentation.csv
│       ├── 20250908_0515_efficientnet_b3_tta_basic_augmentation.csv
│       └── 20250908_0521_swin_base_384_ensemble_tta_basic_augmentation.csv
├── tree.md
└── wandb
    ├── debug-cli.ieyeppo.log
    ├── debug-internal.log -> run-20250910_033714-t6hjolk2/logs/debug-internal.log
    ├── debug.log -> run-20250910_033714-t6hjolk2/logs/debug.log
    ├── lastest-run -> run-20250910_033714-t6hjolk2
    ├── run-20250907_003600-ngwjvpyc
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_003600.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ngwjvpyc.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_003943-2flie7tt
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_003600.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-2flie7tt.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_004330-de0s2kxm
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_003600.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-de0s2kxm.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_004715-x7vt6rg8
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_003600.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-x7vt6rg8.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_005102-pbpqq4ma
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_003600.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-pbpqq4ma.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_083321-p02engu3
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_083322.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-p02engu3.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_090657-btj72mjc
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_090657.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-btj72mjc.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_091107-sxvp7j5o
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_090657.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-sxvp7j5o.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_091512-o4k4xexv
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_090657.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-o4k4xexv.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_091933-iock2oiq
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_090657.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-iock2oiq.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_092346-u7nm0h69
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_090657.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-u7nm0h69.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_095731-swju6ldd
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_095731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-swju6ldd.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_100150-ue5j51a3
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_095731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ue5j51a3.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_100554-ezrsrjc8
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_095731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ezrsrjc8.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_100947-yxwlvz0y
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_095731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-yxwlvz0y.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_101400-4mp6rd8g
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_095731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-4mp6rd8g.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_170210-l98169yb
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_170210.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-l98169yb.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_170549-rm3vu7oj
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_170210.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-rm3vu7oj.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_170925-7okhdj00
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_170210.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-7okhdj00.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_171303-ics9oay9
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_170210.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ics9oay9.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_171640-ybg3jk43
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_170210.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ybg3jk43.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_173442-3tyajt78
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_173442.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-3tyajt78.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_173824-rawrxpm1
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_173442.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-rawrxpm1.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_174206-pppgti21
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_173442.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-pppgti21.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_174552-02i64g63
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_173442.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-02i64g63.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_180302-o8akh6c0
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_180302.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-o8akh6c0.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_182012-h3bm7e0a
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_182012.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-h3bm7e0a.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_182529-292zfd69
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_182529.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-292zfd69.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_200059-n6uhd2jy
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_182529.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-n6uhd2jy.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_212612-qgt0n73e
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_182529.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-qgt0n73e.wandb
    │   └── tmp
    │       └── code
    ├── run-20250907_224824-gm0s4yv4
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_182529.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-gm0s4yv4.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_001337-l3niy70s
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250907_182529.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-l3niy70s.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_030853-l12lxuz8
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250908_030853.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-l12lxuz8.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_031748-2fmx5qgp
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250908_031748.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-2fmx5qgp.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_031826-761maffs
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250908_031826.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-761maffs.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_033849-9nskssyw
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250908_033849.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-9nskssyw.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_060116-bqxdrynu
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250908_060116.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-bqxdrynu.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_070509-w4r46a8b
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250908_070509.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-w4r46a8b.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_162626-t10irwm4
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250908_162626.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-t10irwm4.wandb
    │   └── tmp
    │       └── code
    ├── run-20250908_162918-7wqad52m
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250908_162626.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-7wqad52m.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_112530-xlu39kib
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_112530.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-xlu39kib.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_113126-scgh8ian
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_113126.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-scgh8ian.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_120956-hszra18b
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_113126.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-hszra18b.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_130926-un71lwh0
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_130926.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-un71lwh0.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_131731-a3xe0r35
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_131731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-a3xe0r35.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_131823-6td1cjqs
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_131731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-6td1cjqs.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_131859-35f5w7ps
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_131731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-35f5w7ps.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_132024-ukud57hg
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_131731.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ukud57hg.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_132638-qcojj42g
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_132638.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-qcojj42g.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_134406-5xzb8hz0
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_132638.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-5xzb8hz0.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_135840-1152vr2p
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_132638.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-1152vr2p.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_142808-q2ormp9c
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_142808.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-q2ormp9c.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_151244-oc4r211h
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_142808.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-oc4r211h.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_155458-k1usdo6e
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_142808.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-k1usdo6e.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_162408-ftleg8tn
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_142808.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ftleg8tn.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_174129-p51za1o8
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_174129.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-p51za1o8.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_182349-r8i2xwom
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_182349.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-r8i2xwom.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_184205-hp58qzl6
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_182349.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-hp58qzl6.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_185700-izfseq36
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_182349.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-izfseq36.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_191229-ub6cd13t
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_182349.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ub6cd13t.wandb
    │   └── tmp
    │       └── code
    ├── run-20250909_192232-vctac4c2
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250909_192232.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-vctac4c2.wandb
    │   └── tmp
    │       └── code
    ├── run-20250910_003300-u8d5619j
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250910_003300.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-u8d5619j.wandb
    │   └── tmp
    │       └── code
    ├── run-20250910_005655-ckcd5r5s
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250910_003300.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ckcd5r5s.wandb
    │   └── tmp
    │       └── code
    ├── run-20250910_021643-80cnwubs
    │   ├── files
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250910_003300.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-80cnwubs.wandb
    │   └── tmp
    │       └── code
    └── run-20250910_033714-t6hjolk2
        ├── files
        │   ├── output.log
        │   ├── requirements.txt
        │   └── wandb-metadata.json
        ├── logs
        │   ├── debug-core.log -> /home/ieyeppo/.cache/wandb/logs/core-debug-20250910_003300.log
        │   ├── debug-internal.log
        │   └── debug.log
        ├── run-t6hjolk2.wandb
        └── tmp
            └── code

664 directories, 5944 files
