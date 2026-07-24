"""자세 분류 모델 학습·평가.

사용법:
    python train.py ../data/features.csv

- Random Forest 학습, 그룹(사람/출처) 단위 교차검증
  * 같은 사람의 프레임이 train과 test에 같이 들어가면 정확도가 뻥튀기됨 (leakage)
  * group 컬럼(영상=사람 이름, 이미지=파일명) 기준으로 분할해서 정직한 수치를 만든다
- confusion matrix + feature importance 출력 (발표 자료용)
- 규칙 기반 베이스라인과 정확도 비교
- 발표용 차트 3종을 figures/에 PNG로 저장
- 학습된 트리 규칙을 models/model_rules.json으로 덤프 → 프론트 JS에서 로드해 추론
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

FEATURES = ["knee_angle", "hip_angle", "trunk_lean", "foot_angle", "knee_ankle_ratio"]

# 규칙 기반 베이스라인 (frontend CONFIG와 동일한 임계값 유지)
DEPTH_KNEE_ANGLE = 100
TRUNK_LEAN_MAX = 50
HEEL_FOOT_ANGLE = 25
KNEE_RATIO_MIN = 0.7


def rule_based_predict(row) -> str:
    if row["knee_angle"] > DEPTH_KNEE_ANGLE:
        return "depth"
    if row["trunk_lean"] > TRUNK_LEAN_MAX:
        return "back"
    if row["foot_angle"] > HEEL_FOOT_ANGLE:
        return "heel"
    if row["knee_ankle_ratio"] < KNEE_RATIO_MIN:
        return "knee"
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

    # --- XGBoost (모델 선택 근거용 비교 실험) ---
    # 배포는 RF로 함: 트리를 JSON으로 덤프해 브라우저에서 그대로 추론 가능.
    # XGBoost가 유의미하게 높게 나오면 배포 방식 재검토할 것.
    xgb_acc = None
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder().fit(y)
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                            eval_metric="mlogloss", random_state=42)
        xgb.fit(X_train, le.transform(y_train))
        xgb_acc = accuracy_score(le.transform(y_test), xgb.predict(X_test))
        print(f"[XGBoost]       정확도: {xgb_acc:.3f}")
    except ImportError:
        print("[XGBoost] 미설치 — 비교 생략 (pip install xgboost)")

    xgb_txt = f", XGBoost {xgb_acc:.1%}" if xgb_acc is not None else ""
    print(f"\n=== 발표용 핵심 수치: 규칙 기반 {rule_acc:.1%} → Random Forest {ml_acc:.1%}{xgb_txt} ===")
    print("    (배포 모델: Random Forest — 트리 JSON 덤프로 브라우저 온디바이스 추론)\n")

    print("Confusion Matrix (행=실제, 열=예측):")
    labels = sorted(y.unique())
    print(pd.DataFrame(confusion_matrix(y_test, ml_pred, labels=labels),
                       index=labels, columns=labels))
    print()
    print(classification_report(y_test, ml_pred, zero_division=0))

    print("Feature Importance:")
    for name, imp in zip(FEATURES, model.feature_importances_):
        print(f"  {name}: {imp:.3f}")

    cm = confusion_matrix(y_test, ml_pred, labels=labels)
    save_charts(rule_acc, ml_acc, cm, labels, model.feature_importances_)

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


LABEL_KO = {"good": "정상", "depth": "깊이 부족", "back": "허리 굽음",
            "heel": "발뒤꿈치 들림", "knee": "무릎 모임"}
FEATURE_KO = {"knee_angle": "무릎 각도", "hip_angle": "고관절 각도",
              "trunk_lean": "허리 기울기", "foot_angle": "발 각도",
              "knee_ankle_ratio": "무릎/발목 간격비"}


def save_charts(rule_acc, ml_acc, cm, labels, importances):
    """발표용 차트 3종을 figures/에 저장."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 라벨
    plt.rcParams["axes.unicode_minus"] = False
    BLUE, GRAY, INK, MUTED = "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e"

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    labels_ko = [LABEL_KO.get(l, l) for l in labels]

    # 1) 규칙 기반 vs ML 정확도 — 발표의 핵심 한 장
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(["규칙 기반", "Random Forest"], [rule_acc, ml_acc],
                  color=[GRAY, BLUE], width=0.55)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in (rule_acc, ml_acc)],
                 fontsize=12, fontweight="bold", color=INK, padding=4)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("정확도", color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=200)
    plt.close(fig)

    # 2) Confusion Matrix
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_ko)), labels_ko)
    ax.set_yticks(range(len(labels_ko)), labels_ko)
    ax.set_xlabel("예측", color=MUTED)
    ax.set_ylabel("실제", color=MUTED)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else INK)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # 3) Feature Importance
    order = sorted(range(len(FEATURES)), key=lambda i: importances[i])
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    names = [FEATURE_KO.get(FEATURES[i], FEATURES[i]) for i in order]
    vals = [importances[i] for i in order]
    bars = ax.barh(names, vals, color=BLUE, height=0.55)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=4, color=INK)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=200)
    plt.close(fig)

    print(f"발표용 차트 3종 저장 → {out_dir}\\accuracy_comparison.png, confusion_matrix.png, feature_importance.png")


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
