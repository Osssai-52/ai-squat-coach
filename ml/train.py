"""자세 분류 모델 학습·평가.

사용법:
    python train.py ../data/features.csv

- Random Forest 학습, 그룹(사람/출처) 단위 교차검증
  * 같은 사람의 프레임이 train과 test에 같이 들어가면 정확도가 뻥튀기됨 (leakage)
  * group 컬럼(영상=사람 이름, 이미지=파일명) 기준으로 분할해서 정직한 수치를 만든다
- confusion matrix + feature importance 출력 (발표 자료용)
- 규칙 기반 베이스라인과 정확도 비교
- 학습된 트리 규칙을 models/model_rules.json으로 덤프 → 프론트 JS에서 로드해 추론
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

FEATURES = ["knee_angle", "hip_angle", "trunk_lean"]

# 규칙 기반 베이스라인 (frontend CONFIG와 동일한 임계값 유지)
DEPTH_KNEE_ANGLE = 100
TRUNK_LEAN_MAX = 50


def rule_based_predict(row) -> str:
    if row["knee_angle"] > DEPTH_KNEE_ANGLE:
        return "depth"
    if row["trunk_lean"] > TRUNK_LEAN_MAX:
        return "back"
    return "good"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    X, y, groups = df[FEATURES], df["label"], df["group"]
    n_groups = groups.nunique()
    print(f"데이터: {len(df)}행, 그룹(사람/출처) {n_groups}개")
    print(f"클래스 분포:\n{y.value_counts()}\n")

    # --- 그룹 단위 홀드아웃 분할 (사람/출처가 train·test에 안 섞이게) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    missing = set(y.unique()) - set(y_test.unique())
    if missing:
        print(f"주의: 테스트셋에 없는 클래스 {missing} — 그룹 수가 적어서 생기는 현상. 데이터(특히 사람 수)를 늘릴 것\n")

    # --- 규칙 기반 베이스라인 ---
    rule_pred = X_test.apply(rule_based_predict, axis=1)
    rule_acc = accuracy_score(y_test, rule_pred)
    print(f"[규칙 기반] 정확도: {rule_acc:.3f}")

    # --- Random Forest ---
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    n_folds = min(5, n_groups)
    if n_folds >= 2:
        cv = cross_val_score(model, X, y, cv=GroupKFold(n_splits=n_folds), groups=groups)
        cv_txt = f"({n_folds}-fold 그룹 CV: {cv.mean():.3f} ± {cv.std():.3f})"
    else:
        cv_txt = "(그룹이 1개뿐이라 CV 생략 — 사람을 늘리세요)"
    model.fit(X_train, y_train)
    ml_pred = model.predict(X_test)
    ml_acc = accuracy_score(y_test, ml_pred)
    print(f"[Random Forest] 정확도: {ml_acc:.3f} {cv_txt}")
    print(f"\n=== 발표용 핵심 수치: 규칙 기반 {rule_acc:.1%} → ML {ml_acc:.1%} ===\n")

    print("Confusion Matrix (행=실제, 열=예측):")
    labels = sorted(y.unique())
    print(pd.DataFrame(confusion_matrix(y_test, ml_pred, labels=labels),
                       index=labels, columns=labels))
    print()
    print(classification_report(y_test, ml_pred, zero_division=0))

    print("Feature Importance:")
    for name, imp in zip(FEATURES, model.feature_importances_):
        print(f"  {name}: {imp:.3f}")

    # --- JS 추론용 모델 덤프 ---
    out_dir = Path(__file__).parent / "models"
    out_dir.mkdir(exist_ok=True)
    dump = {
        "features": FEATURES,
        "classes": list(model.classes_),
        "trees": [export_tree(est.tree_, list(model.classes_)) for est in model.estimators_],
    }
    out_path = out_dir / "model_rules.json"
    out_path.write_text(json.dumps(dump), encoding="utf-8")
    print(f"\nJS 추론용 모델 저장 → {out_path} (frontend/에 복사해서 사용)")


def export_tree(tree, classes):
    """sklearn 트리 → JSON (JS에서 재귀 순회로 추론)."""
    def node(i):
        if tree.children_left[i] == -1:  # leaf
            counts = tree.value[i][0]
            return {"class": classes[int(counts.argmax())]}
        return {
            "feature": int(tree.feature[i]),
            "threshold": float(tree.threshold[i]),
            "left": node(int(tree.children_left[i])),
            "right": node(int(tree.children_right[i])),
        }
    return node(0)


if __name__ == "__main__":
    main()
