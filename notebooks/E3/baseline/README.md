E3+++ LOSO 4-FOLD — 3 TRAIN SUBJECTS / 1 UNSEEN TEST SUBJECT PER FOLD
============================================================================

PROTOCOL
--------
4-fold Leave-One-Subject-Out (LOSO) trên: hau, khoi, tai, vy

Mỗi OUTER fold:
- 3 subjects dùng để train.
- 1 subject còn lại là TEST hoàn toàn unseen.
- Outer-test subject không tham gia train, validation hoặc chọn epoch.

MODEL SELECTION BÊN TRONG MỖI FOLD
----------------------------------
Trên 3 outer-train subjects, chạy inner subject-CV:
- lần lượt giữ 1/3 train subject làm validation;
- chọn số epoch từ các inner validation curves;
- fit lại model trên đủ 3 outer-train subjects;
- cuối cùng mới test đúng 1 lần trên outer-test subject.

4 OUTER FOLDS
-------------
Fold 1: train=[khoi,tai,vy] | test=hau | acc=81.25% | macro_f1=80.12% | epochs=28
Fold 2: train=[hau,tai,vy] | test=khoi | acc=76.88% | macro_f1=77.75% | epochs=14
Fold 3: train=[hau,khoi,vy] | test=tai | acc=72.50% | macro_f1=73.25% | epochs=29
Fold 4: train=[hau,khoi,tai] | test=vy | acc=92.50% | macro_f1=93.23% | epochs=20

CV RESULT — NÊN DÙNG ĐỂ BÁO CÁO
-------------------------------
Accuracy mean ± std : 80.78% ± 8.59%
Macro-F1 mean ± std : 81.09% ± 8.58%
Pooled OOF accuracy : 80.78%
Pooled OOF Macro-F1 : 82.09%

BEST FOLD THEO OUTER-TEST — CHỈ THAM KHẢO / LƯU CHECKPOINT
----------------------------------------------------------
Fold        : 4
Test person : vy
Accuracy    : 92.50%
Macro-F1    : 93.23%
Checkpoint  : BEST_FOLD_BY_TEST.pt

CẢNH BÁO QUAN TRỌNG
-------------------
Best fold được chọn sau khi đã thấy score của 4 outer-test folds. Vì vậy không được
lấy accuracy của best fold làm accuracy chung của hệ thống. Làm vậy sẽ cherry-pick
và làm kết quả cao ảo. Metric đánh giá tổng quát hóa đúng là 4-fold mean/std hoặc
pooled out-of-fold metrics ở phía trên.

FILES
-----
LOSO_4FOLD_summary.csv/json
LOSO_4FOLD_metrics_by_fold.csv
LOSO_4FOLD_out_of_fold_predictions.csv
LOSO_4FOLD_pooled_classification_report.csv
LOSO_4FOLD_pooled_confusion_matrix.png
LOSO_4FOLD_scores_by_fold.png
fold_*_test_*_inner_trainloss_valacc.csv
EVAL_05_fold_*_trainloss_valacc.png
EVAL_05_all_folds_trainloss_valacc.png
BEST_FOLD_BY_TEST.pt
fold_checkpoints/fold_1_test_*.pt ... fold_4_test_*.pt

Dữ liệu commit trên repo đã được copy ra từ file gốc, tuyệt đối không chạy baseline ngay tại thư mục baseline này để tránh việc xuất ra nhiều file khác
Tham khảo link https://www.kaggle.com/code/ktoxz205/4-fold/output?scriptVersionId=351033845 để lấy output chuẩn