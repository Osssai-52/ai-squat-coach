// ml/train.py가 덤프한 Random Forest(model_rules.json)를 브라우저에서 추론.
// 파일이 없으면 null을 반환하고 app.js가 규칙 기반 v0으로 폴백한다.

export async function loadPostureModel(url = "model_rules.json") {
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    const model = await res.json();
    if (!model?.trees?.length || !model?.features?.length) return null;
    return model;
  } catch {
    return null;
  }
}

// sklearn 트리 규약: x[feature] <= threshold 이면 left, 아니면 right
function treePredict(node, x) {
  while (node.class === undefined) {
    node = x[node.feature] <= node.threshold ? node.left : node.right;
  }
  return node.class;
}

// 피처 이름 매핑: 모델(snake_case, ml/features.py) ↔ 프론트(camelCase, app.js)
const KEY_MAP = {
  knee_angle: "kneeAngle", hip_angle: "hipAngle",
  trunk_lean: "trunkLean", foot_angle: "footAngle",
  knee_ankle_ratio: "kneeAnkleRatio",
};

// f: computeFeatures() 결과. 반환: 클래스 라벨(다수결) 또는 null(피처 결측)
export function predictPosture(model, f) {
  const x = model.features.map((name) => f[KEY_MAP[name]]);
  if (x.some((v) => v == null)) return null;
  const votes = {};
  for (const tree of model.trees) {
    const c = treePredict(tree, x);
    votes[c] = (votes[c] ?? 0) + 1;
  }
  let best = null, bestCount = -1;
  for (const [c, n] of Object.entries(votes)) {
    if (n > bestCount) { best = c; bestCount = n; }
  }
  return best;
}
