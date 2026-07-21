"""자세 분류 모델 학습·평가 (2일차 오후 작업).

사용법:
    python train.py ../data/features.csv

- Random Forest 학습, 5-fold 교차검증
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
from sklearn.model_selection import cross_val_score, train_test_split

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
    print(f"데이터: {len(df)}행, 클래스 분포:\n{df['label'].value_counts()}\n")

    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- 규칙 기반 베이스라인 ---
    rule_pred = X_test.apply(rule_based_predict, axis=1)
    rule_acc = accuracy_score(y_test, rule_pred)
    print(f"[규칙 기반] 정확도: {rule_acc:.3f}")

    # --- Random Forest ---
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    cv = cross_val_score(model, X, y, cv=5)
    model.fit(X_train, y_train)
    ml_pred = model.predict(X_test)
    ml_acc = accuracy_score(y_test, ml_pred)
    print(f"[Random Forest] 정확도: {ml_acc:.3f} (5-fold CV: {cv.mean():.3f} ± {cv.std():.3f})")
    print(f"\n=== 발표용 핵심 수치: 규칙 기반 {rule_acc:.1%} → ML {ml_acc:.1%} ===\n")

    print("Confusion Matrix (행=실제, 열=예측):")
    labels = sorted(y.unique())
    print(pd.DataFrame(confusion_matrix(y_test, ml_pred, labels=labels),
                       index=labels, columns=labels))
    print()
    print(classification_report(y_test, ml_pred))

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
