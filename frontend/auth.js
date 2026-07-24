// Google 로그인 (Google Identity Services)
// 사용법: 아래 GOOGLE_CLIENT_ID에 Cloud Console에서 발급한 "웹 애플리케이션" 클라이언트 ID를 넣으면
// 로그인 버튼이 활성화된다. 비어 있으면 앱이 개발용 우회 버튼을 보여준다.
//
// 발급 절차 (팀 구글 계정으로, 약 10분):
// 1. https://console.cloud.google.com → 새 프로젝트 생성 (이름: bodybuddy 등)
// 2. API 및 서비스 → OAuth 동의 화면 → External → 앱 이름/이메일만 입력 → 저장
//    → "테스트 사용자"에 팀원 전원 + 데모에 쓸 구글 계정 추가 (안 하면 로그인 차단됨!)
// 3. API 및 서비스 → 사용자 인증 정보 → 사용자 인증 정보 만들기 → OAuth 클라이언트 ID
//    → 유형: 웹 애플리케이션
//    → 승인된 JavaScript 원본에 추가: http://localhost:8000  (데모 기기 주소가 다르면 그것도 추가)
// 4. 발급된 "xxxx.apps.googleusercontent.com"을 아래에 붙여넣기

export const GOOGLE_CLIENT_ID = ""; // TODO: 여기에 클라이언트 ID 입력

// JWT(ID 토큰) payload 디코딩 — 서명 검증은 하지 않음 (백엔드 없는 데모 용도)
export function decodeJwtPayload(token) {
  try {
    const base64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const json = decodeURIComponent(
      atob(base64).split("").map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2)).join("")
    );
    return JSON.parse(json);
  } catch {
    return null;
  }
}

// 로그인 버튼 렌더링. 성공 시 onUser({name, email, picture}) 호출.
// 반환: CLIENT_ID가 설정돼 있으면 true, 아니면 false
export function initGoogleSignIn(buttonEl, onUser) {
  if (!GOOGLE_CLIENT_ID) return false;

  const start = () => {
    window.google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: (res) => {
        const p = decodeJwtPayload(res.credential);
        if (p) onUser({ name: p.name ?? "", email: p.email ?? "", picture: p.picture ?? "" });
      },
    });
    window.google.accounts.id.renderButton(buttonEl, {
      theme: "outline", size: "large", shape: "pill", text: "continue_with", width: 280,
    });
  };

  // GIS 스크립트 로딩 대기 (async 로드라 타이밍이 어긋날 수 있음)
  let tries = 0;
  const wait = setInterval(() => {
    if (window.google?.accounts?.id) {
      clearInterval(wait);
      start();
    } else if (++tries > 50) {
      clearInterval(wait);
      console.warn("Google Identity Services 스크립트 로딩 실패 (네트워크 확인)");
    }
  }, 100);
  return true;
}

export function googleSignOut() {
  window.google?.accounts?.id?.disableAutoSelect?.();
}
