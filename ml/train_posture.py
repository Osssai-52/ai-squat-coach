"""체형 분류 모델 학습·평가 (거북목/어깨 말림).

사용법:
    python train_posture.py ../data/posture_features.csv

- 규칙 기반(임계값) vs Random Forest 비교 — 스쿼트와 동일 방법론
- 그룹(사람/출처) 단위 분할로 leakage 방지
- models/posture_model.json 덤프 → frontend/에 복사하면 체형 분석이 ML로 전환
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

from train import export_tree

FEATURES = ["forward_head", "round_shoulder", "trunk_lean"]

# 규칙 기반 베이스라인 (frontend/posture.js의 T 임계값과 동일 유지)
FORWARD_HEAD_T = 0.12
ROUND_SHOULDER_T = 0.10


def rule_based_predict(row) -> str:
    if row["forward_head"] > FORWARD_HEAD_T:
        return "neck"
    if row["round_shoulder"] > ROUND_SHOULDER_T:
        return "shoulder"
    return "normal"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    X, y, groups = df[FEATURES], df["label"], df["group"]
    n_groups = groups.nunique()
    print(f"데이터: {len(df)}행, 그룹 {n_groups}개")
    print(f"클래스 분포:\n{y.value_counts()}\n")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rule_pred = X_test.apply(rule_based_predict, axis=1)
    rule_acc = accuracy_score(y_test, rule_pred)
    print(f"[규칙 기반] 정확도: {rule_acc:.3f}")

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    n_folds = min(5, n_groups)
    if n_folds >= 2:
        cv = cross_val_score(model, X, y, cv=GroupKFold(n_splits=n_folds), groups=groups)
        cv_txt = f"({n_folds}-fold 그룹 CV: {cv.mean():.3f} ± {cv.std():.3f})"
    else:
        cv_txt = "(그룹 1개 — CV 생략)"
    model.fit(X_train, y_train)
    ml_pred = model.predict(X_test)
    ml_acc = accuracy_score(y_test, ml_pred)
    print(f"[Random Forest] 정확도: {ml_acc:.3f} {cv_txt}")
    print(f"\n=== 체형 분류: 규칙 기반 {rule_acc:.1%} → ML {ml_acc:.1%} ===\n")

    labels = sorted(y.unique())
    print("Confusion Matrix (행=실제, 열=예측):")
    print(pd.DataFrame(confusion_matrix(y_test, ml_pred, labels=labels),
                       index=labels, columns=labels))
    print()
    print(classification_report(y_test, ml_pred, zero_division=0))
    print("Feature Importance:")
    for name, imp in zip(FEATURES, model.feature_importances_):
        print(f"  {name}: {imp:.3f}")

    # 평가는 홀드아웃으로, 배포 모델은 전체 데이터로 재학습
    model.fit(X, y)
    out_dir = Path(__file__).parent / "models"
    out_dir.mkdir(exist_ok=True)
    dump = {
        "features": FEATURES,
        "classes": list(model.classes_),
        "trees": [export_tree(est.tree_, list(model.classes_)) for est in model.estimators_],
    }
    out_path = out_dir / "posture_model.json"
    out_path.write_text(json.dumps(dump), encoding="utf-8")
    print(f"\nJS 추론용 모델 저장 → {out_path} (frontend/에 복사해서 사용)")


if __name__ == "__main__":
    main()
