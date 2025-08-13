# =========================================
# app.py  —  K-드라마 케미스코어 (Sparrow UI 스킨 + 사이드 네비, Safe-Mode/403 회피 가드 포함)
# =========================================

# ---- page config MUST be first ----
import streamlit as st
st.set_page_config(page_title="케미스코어 | K-드라마 분석/예측", page_icon="🎬", layout="wide")

# =========================
# ⚙️ Safe Mode & Limits
# =========================
import os, time
SAFE_MODE = os.getenv("CHEMI_SAFE_MODE", "1") not in ("0", "false", "False")
SAFE_MAX_ROWS            = int(os.getenv("CHEMI_MAX_ROWS", "5000"))     # 튜닝에서 사용할 최대 행 수(샘플)
SAFE_MAX_GSCV_EVALS      = int(os.getenv("CHEMI_MAX_EVALS", "120"))     # (그리드 조합수 × CV 폴드수)
SAFE_MAX_PRUNED_ALPHAS   = int(os.getenv("CHEMI_MAX_PRUNED", "25"))     # Pruned 트리 후보 상한
SAFE_GS_COOLDOWN_SEC     = int(os.getenv("CHEMI_GS_COOLDOWN", "15"))    # GridSearch 연타 방지
SAFE_PRED_COOLDOWN_SEC   = int(os.getenv("CHEMI_PRED_COOLDOWN", "5"))   # 예측 연타 방지

# ---- dependency guard (optional) ----
import importlib.util
_missing = [m for m in ("numpy","scipy","sklearn","joblib","threadpoolctl","xgboost") if importlib.util.find_spec(m) is None]
if _missing:
    st.error(f"필수 라이브러리 미설치: {_missing}. requirements.txt / runtime.txt 버전을 고정해 재배포하세요.")
    st.stop()

# ---- imports ----
import ast, random, re, platform
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from matplotlib.patches import Patch

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# XGB가 설치돼 있으면 쓰도록 안전하게 추가
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ===== Global seed =====
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

# ===== Matplotlib (한글 폰트) =====
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

if st.session_state.get("font_cache_cleared") is not True:
    import shutil
    shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)
    st.session_state["font_cache_cleared"] = True

def ensure_korean_font():
    matplotlib.rcParams['axes.unicode_minus'] = False
    base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    candidates = [
        base / "fonts" / "NanumGothic-Regular.ttf",
        base / "fonts" / "NanumGothic.ttf",
    ]
    if platform.system() == "Windows":
        candidates += [Path(r"C:\Windows\Fonts\malgun.ttf"), Path(r"C:\Windows\Fonts\malgunbd.ttf")]
    wanted = ("nanum","malgun","applegothic","notosanscjk","sourcehan","gulim","dotum","batang",
              "pretendard","gowun","spoqa")
    for f in fm.findSystemFonts(fontext="ttf"):
        if any(k in os.path.basename(f).lower() for k in wanted):
            candidates.append(Path(f))
    for p in candidates:
        try:
            if p.exists():
                fm.fontManager.addfont(str(p))
                family = fm.FontProperties(fname=str(p)).get_name()
                matplotlib.rcParams['font.family'] = family
                st.session_state["kfont_path"] = str(p)  # WordCloud용
                return family
        except Exception:
            continue
    st.session_state["kfont_path"] = None
    return None

_ = ensure_korean_font()

# ===== Safe-run helpers (쿨다운 & 403 핸들링) =====
def _cooldown_ok(key: str, cooldown: int) -> bool:
    now = time.time()
    last = st.session_state.get(key)
    if last is None or (now - last) >= cooldown:
        st.session_state[key] = now
        return True
    return False

def run_safely(fn, *args, **kwargs):
    """403(Fair-use) 등 치명 오류를 잡아 UI에서 안내하고 같은 런에서 추가 실행을 멈춤"""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "fair-use" in msg or "blocked" in msg:
            st.error("호스팅 환경의 Fair-use 제한(403)으로 요청이 차단되었습니다. "
                     "튜닝 그리드/폴드/후보 수를 줄이거나 Safe Mode를 유지해 주세요.")
            st.stop()
        else:
            st.exception(e)
            st.stop()

def count_param_evals(grid: Dict[str, List[Any]]) -> int:
    total = 1
    for k, v in grid.items():
        total *= max(1, len(v))
    return total

# ====== Sparrow UI CSS ======
def _inject_sparrow_css():
    st.markdown("""
    <style>
      /* ---------- Layout / Typography ---------- */
      :root{ --page-top-pad: 5.0rem; }  /* ← 필요하면 5.5~8rem 사이로 조절 */
      .block-container{padding-top: var(--page-top-pad) !important;
        padding-bottom: 2.6rem !important;}
      h1,h2,h3{font-weight:800;}
      /* Plotly 컨테이너 위쪽 간격 조금 */
      div[data-testid="stPlotlyChart"]{ margin-top:8px; }
      /* ---------- Sidebar ---------- */
      section[data-testid="stSidebar"]{
        width:220px !important; min-width:220px;
        background:#ffffff; color:#111827; border-right:1px solid #070c16;
      }
      .sb-wrap{display:flex; flex-direction:column; height:100%;}
      .sb-brand{display:flex; align-items:center; gap:10px; padding:14px 12px 10px;}
      .sb-brand .logo{font-size:20px}
      .sb-brand .name{font-size:16px; font-weight:800; letter-spacing:.2px}
      .sb-menu{padding:6px 8px 8px; display:flex; flex-direction:column;}
      .sb-nav{margin:2px 0;}
      .sb-nav .stButton>button{
        width:100% !important; display:flex; align-items:center; gap:10px; justify-content:flex-start;
        background:transparent !important; color:#e5e7eb !important;
        border:1px solid #162033 !important; border-radius:10px !important;
        padding:8px 10px !important; font-size:14px !important; box-shadow:none !important; opacity:1 !important;
      }
      .sb-nav .stButton>button:hover{ background:#111a2b !important; border-color:#25324a !important; }
      .sb-nav.active .stButton>button{ background:#2563eb !important; border-color:#2563eb !important; color:#ffffff !important; }
      .sb-card{background:#0f172a; border:1px solid #1f2937; border-radius:12px; padding:10px; margin-top:8px;}
      .sb-card h4{margin:0 0 6px 0; font-size:12px; color:#cbd5e1; font-weight:800;}
      .sb-footer{margin-top:auto; padding:10px 12px; font-size:11px; color:#9ca3af; border-top:1px solid #070c16;}
      /* ---------- Cards ---------- */
      .kpi-row{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:8px 0 6px;}
      .kpi{ background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px;
            box-shadow:0 6px 18px rgba(17,24,39,.04); }
      .kpi h6{margin:0 0 4px; font-size:12px; color:#6b7280; font-weight:800;}
      .kpi .v{font-size:22px; font-weight:800; line-height:1;}
      .kpi .d{font-size:12px; color:#10b981; font-weight:700;}
      div[data-testid="stPlotlyChart"], div.stPlot {margin-top:8px;}
      .block-container{padding-top:1.4rem; padding-bottom:2.2rem;}
      .kpi-row{ margin-bottom: 18px; }
      div[data-testid="stPlotlyChart"]{ margin-top:8px; }
      .sb-safe{
  width:100%;
  display:flex; align-items:center; justify-content:center;
  padding:10px 12px;                 /* 버튼과 유사한 패딩 */
  border:1px solid #e5e7eb;          /* 버튼 테두리 */
  background:#ffffff;                /* 버튼 배경 */
  color:#111827;                     /* 글자색 */
  font-weight:700; font-size:14px;   /* 버튼과 동일 크기 */
  border-radius:10px;                /* 버튼과 동일 라운드 */
  box-shadow:0 1px 0 rgba(17,24,39,.02);
  margin:8px 0 0;                    /* 위 여백만 살짝 */
}
     .sb-safe .dot{
  width:8px; height:8px; border-radius:9999px;
  background:#10b981;                /* ON 초록 점 */
  margin-right:8px;
}

/* ▶ 모델 설정 카드: 메타(작게, 한 줄) */
    .sb-card.sb-config { background:#ffffff !important;
  border:1px solid #e5e7eb !important;
  color:#111827 !important;}
    .sb-card.sb-config.sb-card-title{
      margin:0 0 4px 0; font-size:12px; font-weight:800; color:#0f172a !important;
    }
    .sb-card.sb-config .meta{
      font-size:11px; color:#6b7280;
      white-space:nowrap;                /* 한 줄 고정 */
      overflow:hidden; text-overflow:ellipsis; /* 너무 길면 … */
      line-height:1.1;
    }
        
    </style>
    """, unsafe_allow_html=True)

_inject_sparrow_css()

# ===== Data helpers =====
def clean_cell_colab(x):
    if isinstance(x, list):
        return [str(i).strip() for i in x if pd.notna(i)]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed if pd.notna(i)]
            return [x.strip()]
        except Exception:
            return [x.strip()]
    return [str(x).strip()]

@st.cache_data(show_spinner=False)
def load_data():
    raw = pd.read_json('drama_d.json')
    if isinstance(raw, pd.Series):
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    else:
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    # Safe mode: 너무 큰 데이터면 샘플링
    if SAFE_MODE and len(raw) > SAFE_MAX_ROWS:
        raw = raw.sample(SAFE_MAX_ROWS, random_state=SEED).reset_index(drop=True)
    return raw

raw_df = load_data()

# ===== Multi-label encoding base =====
def colab_multilabel_fit_transform(df: pd.DataFrame, cols=('genres','day','network')) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)
        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(out[col])
        new_cols = [f"{col}_{c.strip().upper()}" for c in mlb.classes_]
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors='ignore')
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)
        st.session_state[f"mlb_{col}"] = mlb
        st.session_state[f"mlb_classes_{col}"] = mlb.classes_.tolist()
    return out

def colab_multilabel_transform(df: pd.DataFrame, cols=('genres','day','network')) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)
        mlb = st.session_state.get(f"mlb_{col}", None)
        if mlb is None:
            classes = st.session_state.get(f"mlb_classes_{col}", [])
            mlb = MultiLabelBinarizer()
            if classes:
                mlb.classes_ = np.array(classes)
            else:
                try:
                    prefix = f"{col}_"
                    labels = [c[len(prefix):] for c in df_mlb.columns if c.startswith(prefix)]
                    if labels:
                        mlb.classes_ = np.array(labels)
                    else:
                        mlb.fit(out[col])
                except Exception:
                    mlb.fit(out[col])
        arr = mlb.transform(out[col])
        new_cols = [f"{col}_{c.strip().upper()}" for c in mlb.classes_]
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors='ignore')
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)
    return out

df_mlb = colab_multilabel_fit_transform(raw_df, cols=('genres','day','network'))

# ===== Feature base =====
drop_cols = [c for c in ['배우명','드라마명','genres','day','network','score','start airing'] if c in df_mlb.columns]
if 'score' in df_mlb.columns:
    df_mlb['score'] = pd.to_numeric(df_mlb['score'], errors='coerce')

X_colab_base = df_mlb.drop(columns=drop_cols, errors='ignore')
y_all = df_mlb['score']

categorical_features = [c for c in ['role','gender','air_q','married','age_group'] if c in X_colab_base.columns]
try:
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[('cat',
                   Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='(missing)')),
                                   ('ohe', ohe)]),
                   categorical_features)],
    remainder='passthrough'
)

# ===== EDA lists =====
genre_list = [g for sub in raw_df.get('genres', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for g in sub]
broadcaster_list = [b for sub in raw_df.get('network', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for b in sub]
week_list = [w for sub in raw_df.get('day', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for w in sub]
unique_genres = sorted(set(genre_list))

# ===== Age-group helper =====
def age_to_age_group(age: int) -> str:
    s = raw_df.get('age_group')
    if s is None or s.dropna().empty:
        if age < 20: return "10대"
        if age < 30: return "20대"
        if age < 40: return "30대"
        if age < 50: return "40대"
        if age < 60: return "50대"
        return "60대 이상"
    series = s.dropna().astype(str)
    vocab = series.unique().tolist()
    counts = series.value_counts()
    decade = (int(age)//10)*10
    exact = [g for g in vocab if re.search(rf"{decade}\s*대", g)]
    if exact: return counts[exact].idxmax()
    loose = [g for g in vocab if str(decade) in g]
    if loose: return counts[loose].idxmax()
    if decade >= 60:
        over = [g for g in vocab if ('60' in g) or ('이상' in g)]
        if over: return counts[over].idxmax()
    with_num = []
    for g in vocab:
        m = re.search(r'(\d+)', g)
        if m: with_num.append((g, int(m.group(1))))
    if with_num:
        nearest = min(with_num, key=lambda t: abs(t[1]-decade))[1]
        candidates = [g for g,n in with_num if n==nearest]
        return counts[candidates].idxmax()
    return counts.idxmax()

# =============================
# 네비게이션 정의 & 쿼리파람 동기화
# =============================
def _get_nav_from_query():
    if hasattr(st, "query_params"):  # Streamlit 1.30+
        val = st.query_params.get("nav", None)
        return val[0] if isinstance(val, list) else val
    else:
        qp = st.experimental_get_query_params()
        val = qp.get("nav", [None])
        return val[0]

def _set_nav_query(slug: str):
    if hasattr(st, "query_params"):
        st.query_params["nav"] = slug
    else:
        st.experimental_set_query_params(nav=slug)

# ---------- 각 페이지 ----------
def page_overview():
    total_titles = int(raw_df['드라마명'].nunique()) if '드라마명' in raw_df.columns else int(raw_df.shape[0])
    total_actors = int(raw_df['배우명'].nunique()) if '배우명' in raw_df.columns else \
                   (int(raw_df['actor'].nunique()) if 'actor' in raw_df.columns else int(raw_df.shape[0]))
    avg_score = float(pd.to_numeric(raw_df['score'], errors='coerce').mean())

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><h6>TOTAL TITLES</h6><div class="v">{total_titles}</div><div class="d">전체 작품</div></div>
      <div class="kpi"><h6>TOTAL ACTORS</h6><div class="v">{total_actors}</div><div class="d">명</div></div>
      <div class="kpi"><h6>AVG CHEMI SCORE</h6><div class="v">{0.0 if np.isnan(avg_score) else round(avg_score,2):.2f}</div><div class="d">전체 평균</div></div>
      <div class="kpi"><h6>GENRES</h6><div class="v">{len(unique_genres)}</div><div class="d">유니크</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("연도별 평균 케미스코어")
    df_year = raw_df.copy()
    df_year['start airing'] = pd.to_numeric(df_year['start airing'], errors='coerce')
    df_year['score'] = pd.to_numeric(df_year['score'], errors='coerce')
    df_year = df_year.dropna(subset=['start airing','score'])
    fig = px.line(df_year.groupby('start airing')['score'].mean().reset_index(),
                  x='start airing', y='score', markers=True)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("최근 연도 상위 작품")
        _df = raw_df.copy()
        _df['start airing'] = pd.to_numeric(_df['start airing'], errors='coerce')
        _df['score'] = pd.to_numeric(_df['score'], errors='coerce')
        _df = _df.dropna(subset=['start airing', 'score'])
    
        if not _df.empty:
            last_year = int(_df['start airing'].max())
            recent = _df[_df['start airing'].between(last_year-1, last_year)]
            name_col = '드라마명' if '드라마명' in recent.columns else ('title' if 'title' in recent.columns else recent.columns[0])
    
            recent_unique = (
                recent.sort_values('score', ascending=False)
                      .drop_duplicates(subset=[name_col], keep='first')
            )
            top_recent = recent_unique.sort_values('score', ascending=False).head(10)
    
            ymax = float(top_recent['score'].max())
    
            fig_recent = px.bar(top_recent, x=name_col, y='score', text='score')
            fig_recent.update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside',
                cliponaxis=False   # ✅ 축 밖 텍스트 클리핑 방지
            )
            fig_recent.update_yaxes(range=[0, ymax * 1.15])  # ✅ 머리 공간
            fig_recent.update_layout(
                height=420,
                margin=dict(l=12, r=12, t=72, b=60),          # ✅ top 여백 확대
                xaxis=dict(tickangle=-30, automargin=True),
                uniformtext_minsize=10, uniformtext_mode='hide'
            )
            st.plotly_chart(fig_recent, use_container_width=True)
        else:
            st.info("최근 연도 데이터가 없습니다.")

    with c2:
        st.subheader("플랫폼별 작품 수 (TOP 10)")
    
        # 1) 집계
        p_cnt = (
            raw_df.assign(network=raw_df["network"].apply(clean_cell_colab))
                  .explode("network")
                  .dropna(subset=["network"])
                  .groupby("network")
                  .size()
                  .reset_index(name="count")
        )
        p_cnt = p_cnt.loc[:, ~p_cnt.columns.duplicated()].copy()
    
        # 2) etc_p 식별 (대소문자/기호 무시)
        import re
        def _norm(s: str) -> str:
            return re.sub(r'[^a-z0-9]+', '', str(s).lower())
    
        p_cnt["__is_etc"] = p_cnt["network"].map(lambda x: _norm(x) == "etcp")
    
        # 3) 정렬: etc_p 제외한 항목을 count 내림차순 → 상위 10개
        main_sorted = p_cnt.loc[~p_cnt["__is_etc"]].sort_values("count", ascending=False).head(10)
        etc_rows    = p_cnt.loc[p_cnt["__is_etc"]]
    
        # 4) 최종 순서: 메인(내림차순) + etc_p(맨 뒤)
        p_sorted = pd.concat([main_sorted, etc_rows], ignore_index=True)
    
        # 5) Plotly가 우리가 만든 순서를 그대로 쓰도록 카테고리 순서 고정
        p_sorted["network"] = pd.Categorical(p_sorted["network"],
                                             categories=p_sorted["network"].tolist(),
                                             ordered=True)
    
        # 6) 차트
        fig_p = px.bar(p_sorted, x="network", y="count", text="count")
        fig_p.update_traces(textposition="outside", cliponaxis=False)
        fig_p.update_layout(height=360, margin=dict(l=12, r=12, t=36, b=80))
        st.plotly_chart(fig_p, use_container_width=True)

def page_basic():
    st.header("기초 통계: score")
    st.write(pd.to_numeric(raw_df['score'], errors='coerce').describe())
    fig,ax=plt.subplots(figsize=(6,3))
    ax.hist(pd.to_numeric(raw_df['score'], errors='coerce'), bins=20)
    ax.set_title("전체 평점 분포")
    st.pyplot(fig)

def page_dist():
    st.header("분포 및 교차분석")
    df = raw_df.copy()

    # -------- 공통 유틸/컬럼 매핑 --------
    def _find_col(cands):
        lower_map = {c.lower(): c for c in df.columns}
        for cand in cands:
            if cand in df.columns:
                return cand
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None
    
    def _ensure_list(x):
        if isinstance(x, (list, tuple)): return list(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if x is None: return []
        try:
            if pd.isna(x): return []
        except Exception:
            pass
        if isinstance(x, str):
            s = x.strip()
            if not s or s.lower() in {"nan","none","null"}: return []
            if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                try:
                    p = ast.literal_eval(s)
                    return list(p) if isinstance(p, (list, tuple, np.ndarray)) else [str(p)]
                except Exception:
                    return [s]
            return [s]
        return [str(x)]

    score_col   = _find_col(['점수','score'])
    role_col    = _find_col(['역할','role'])
    gender_col  = _find_col(['성별','gender'])
    ageg_col    = _find_col(['연령대','age_group'])
    day_col     = _find_col(['방영요일','day'])
    genre_col   = _find_col(['장르','genres'])
    year_col    = _find_col(['방영년도','start airing'])
    married_col = _find_col(['결혼여부','married'])
    plat_col    = _find_col(['방영 플랫폼','방영플랫폼','플랫폼','플렛폼','Network','network','네트워크','방영채널','채널','방송사','Station','station','방영사']) or _find_col(['network'])

    # -------- 1) 역할별 --------
    st.subheader("1) 역할별 작품수 및 평균 점수")
    if score_col and role_col and df[role_col].notna().any():
        role_df = df[df[score_col].notna() & df[role_col].notna()].copy()
        count_by_role = role_df[role_col].value_counts()
        avg_by_role = role_df.groupby(role_col)[score_col].mean().round(3)
        roles = count_by_role.index.tolist()

        def _role_color(v:str):
            s = str(v).lower()
            if any(k in s for k in ['주연','lead','main']): return 'tab:orange'
            if any(k in s for k in ['조연','support']):     return 'tab:green'
            return 'lightgray'

        bar_colors = [_role_color(r) for r in roles]

        fig, ax1 = plt.subplots(figsize=(6,5))
        bars = ax1.bar(roles, count_by_role[roles], color=bar_colors, alpha=0.75)
        ax1.set_ylabel('작품 수'); ax1.set_xlabel('역할')
        ax1.set_ylim(0, count_by_role.max()*1.25); ax1.grid(axis='y', ls='--', alpha=0.5)
        for r,b in zip(roles, bars):
            v = count_by_role[r]; ax1.text(b.get_x()+b.get_width()/2, v+count_by_role.max()*0.03, f"{v}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2 = ax1.twinx()
        ymin, ymax = float(avg_by_role.min()), float(avg_by_role.max())
        ax2.set_ylim(ymin-0.05, ymax+0.05)
        ax2.plot(roles, avg_by_role[roles], color='tab:blue', marker='o', lw=2, label='평균 점수')
        for r in roles:
            v = avg_by_role[r]; ax2.text(r, v+0.005, f"{v:.3f}", color='tab:blue', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue'); ax2.legend(loc='upper right', fontsize=9)
        plt.title('역할별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)
        st.markdown("🔎 **인사이트**: 주연과 조연 간에 작품수와 평균 평점은 뚜렷한 상관관계를 보이지 않음.")
    else:
        st.info("역할/점수 컬럼이 없어 건너뜀.")

    # -------- 2) 성별 --------
    st.subheader("2) 성별 작품수 및 평균 점수")
    if score_col and gender_col and df[gender_col].notna().any():
        gdf = df[df[gender_col].isin(['남자','여자','male','female']) & df[score_col].notna()].copy()
        # 한국어/영어 통일
        gdf['_gender'] = gdf[gender_col].astype(str).str.lower().map({'남자':'남자','여자':'여자','male':'남자','female':'여자'})
        count_by_gender = gdf['_gender'].value_counts()
        avg_by_gender = gdf.groupby('_gender')[score_col].mean().round(3)
        order = ['남자','여자']; order = [x for x in order if x in count_by_gender.index]
        bar_colors = ['dodgerblue' if g=='남자' else 'hotpink' for g in order]

        fig, ax1 = plt.subplots(figsize=(6,5))
        bars = ax1.bar(order, count_by_gender[order].values, color=bar_colors, alpha=0.75)
        ax1.set_ylabel('작품 수'); ax1.set_xlabel('성별')
        ax1.set_ylim(0, count_by_gender.max()*1.25); ax1.grid(axis='y', ls='--', alpha=0.5)
        for g,b in zip(order, bars):
            v = count_by_gender[g]; ax1.text(b.get_x()+b.get_width()/2, v+count_by_gender.max()*0.03, f"{v}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2 = ax1.twinx()
        ymin, ymax = float(avg_by_gender.min()), float(avg_by_gender.max())
        ax2.set_ylim(ymin-0.05, ymax+0.05)
        ax2.plot(order, avg_by_gender[order].values, color='tab:blue', marker='o', lw=2, label='평균 점수')
        for i,g in enumerate(order):
            v = avg_by_gender[g]; ax2.text(i, v+0.005, f"{v:.3f}", color='tab:blue', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue'); ax2.legend(loc='upper right', fontsize=9)
        plt.title('성별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)
        st.markdown("🔎 **인사이트**: 남성 배우가 여성 배우보다 캐스팅과 평점에서 약간 더 우호적 경향.")
    else:
        st.info("성별/점수 컬럼이 없어 건너뜀.")

    # -------- 3) 연령대 --------
    st.subheader("3) 연령대별 작품수 및 평균 점수")
    if score_col and ageg_col and df[ageg_col].notna().any():
        age_clean = df[ageg_col].dropna().astype(str).str.strip()
        count_by_age = age_clean.value_counts()
        def age_sort_key(s: str):
            m = re.search(r'(\d+)', s); return (int(m.group(1)) if m else 10_000, s)
        mean_by_age = (df[[ageg_col, score_col]].dropna().assign(**{ageg_col:lambda x:x[ageg_col].astype(str).str.strip()})
                       .groupby(ageg_col)[score_col].mean().round(3))
        labels = sorted(set(count_by_age.index)|set(mean_by_age.index), key=age_sort_key)
        count_by_age = count_by_age.reindex(labels).fillna(0).astype(int)
        mean_by_age = mean_by_age.reindex(labels)
        def get_decade(label:str):
            m = re.search(r'(\d+)', str(label)); return int(m.group(1)) if m else None
        color_map = {30:'tab:orange', 50:'tab:green'}
        bar_colors = [color_map.get(get_decade(lb),'lightgray') for lb in labels]

        fig, ax1 = plt.subplots(figsize=(8,5))
        x = np.arange(len(labels))
        bars = ax1.bar(x, count_by_age.values, color=bar_colors, alpha=0.85)
        ax1.set_ylabel('작품 수'); ax1.set_xlabel('연령대'); ax1.set_xticks(x); ax1.set_xticklabels(labels)
        maxv = int(count_by_age.max()) if len(count_by_age) else 0
        ax1.set_ylim(0, maxv*1.23 if maxv>0 else 1); ax1.grid(axis='y', ls='--', alpha=0.5)
        pad = maxv*0.03 if maxv>0 else 0.05
        for i,b in enumerate(bars):
            v = int(count_by_age.values[i]); ax1.text(b.get_x()+b.get_width()/2, v+pad, f"{v}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2 = ax1.twinx()
        if mean_by_age.notna().any():
            y = mean_by_age.values; valid = mean_by_age.dropna(); ymin,ymax = float(valid.min()), float(valid.max()); pad_y=0.02
            ax2.set_ylim(ymin-pad_y, ymax+pad_y)
            ax2.plot(x, y, color='tab:blue', marker='o', lw=2, label='평균 점수')
            for i,val in enumerate(y):
                if not np.isnan(val): ax2.text(i, val+0.005, f"{val:.3f}", color='tab:blue', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue'); ax2.legend(loc='upper right', fontsize=9, frameon=False)
        handles=[]
        present_decades={get_decade(lb) for lb in labels}
        for dec,color in color_map.items():
            if dec in present_decades: handles.append(Patch(facecolor=color, label=f'{dec}대 작품 수'))
        if handles: ax1.legend(handles=handles, loc='upper left', fontsize=9, frameon=False)
        plt.title('연령대별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)
        st.markdown("🔎 **인사이트**: 30대 배우의 작품 활동이 가장 많고, 50대 배우의 평균 평점이 가장 높음.")
    else:
        st.info("연령대/점수 컬럼이 없어 건너뜀.")

    # -------- 4) 요일 --------
    st.subheader("4) 요일별 작품수 및 평균 점수")
    if score_col and day_col and df[day_col].notna().any():
        tmp = df[[day_col, score_col]].copy()
        tmp[day_col] = tmp[day_col].apply(_ensure_list)
        ex = tmp.explode(day_col).dropna(subset=[day_col])
        def _to_en(x):
            s = str(x).strip().lower()
            kor = {'월':'monday','화':'tuesday','수':'wednesday','목':'thursday','금':'friday','토':'saturday','일':'sunday',
                   '월요일':'monday','화요일':'tuesday','수요일':'wednesday','목요일':'thursday','금요일':'friday','토요일':'saturday','일요일':'sunday'}
            return kor.get(s, s)
        ex['_day'] = ex[day_col].astype(str).map(_to_en)
        ordered = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
        count_by_day = ex['_day'].value_counts().reindex(ordered).fillna(0).astype(int)
        mean_by_day = ex.groupby('_day')[score_col].mean().reindex(ordered).round(3)

        weekday_color = 'lightgray'
        cmap = {'friday':'tab:purple','saturday':'tab:orange','sunday':'tab:green'}
        bar_colors = [cmap.get(d, weekday_color) for d in ordered]

        fig, ax1 = plt.subplots(figsize=(8,5))
        bars = ax1.bar(ordered, count_by_day.values, color=bar_colors, alpha=0.85)
        ax1.set_ylabel('작품 수'); ax1.set_xlabel('방영 요일')
        maxc = int(count_by_day.max()) if len(count_by_day) else 0
        ax1.set_ylim(0, maxc*1.15 if maxc>0 else 1); ax1.grid(axis='y', ls='--', alpha=0.5)
        pad = maxc*0.03 if maxc>0 else 0.05
        for d,b in zip(ordered, bars):
            v = int(count_by_day.loc[d]); ax1.text(b.get_x()+b.get_width()/2, v+pad, f"{v}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2 = ax1.twinx()
        if mean_by_day.notna().any():
            y = mean_by_day.values; valid = mean_by_day.dropna(); ymin,ymax = float(valid.min()), float(valid.max()); pad_y=0.015
            ax2.set_ylim(ymin-pad_y, ymax+pad_y)
            ax2.plot(ordered, y, color='tab:blue', marker='o', lw=2, label='평균 점수')
            for x, val in zip(ordered, y):
                if not np.isnan(val): ax2.text(x, val+0.01, f"{val:.3f}", color='tab:blue', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue'); ax2.legend(loc='upper right', fontsize=9, frameon=False)
        plt.title('요일별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)
        st.markdown("🔎 **인사이트**: 주중은 작품 수가 많지만 평균 점수는 낮고, 주말은 작품 수 대비 높은 점수 경향.")
    else:
        st.info("방영요일/점수 컬럼이 없어 건너뜀.")

    # -------- 5) 장르 --------
    st.subheader("5) 장르별 작품수 및 평균 점수")
    if score_col and genre_col and df[genre_col].notna().any():
        gtmp = df[[genre_col, score_col]].copy()
        gtmp[genre_col] = gtmp[genre_col].apply(_ensure_list)
        gex = gtmp.explode(genre_col).dropna(subset=[genre_col])
        genre_count = gex[genre_col].astype(str).value_counts()
        genre_score = gex.groupby(genre_col)[score_col].mean().round(3)
        gdf2 = (pd.DataFrame({'작품 수': genre_count, '평균 점수': genre_score})
                .reset_index().rename(columns={'index':'장르'}))
        if 'etc_g' in gdf2['장르'].values:
            gdf2 = pd.concat([
                gdf2[gdf2['장르']!='etc_g'].sort_values('작품 수', ascending=False),
                gdf2[gdf2['장르']=='etc_g']
            ])
        default_color = 'lightgray'
        color_map = {
            'romance':'#ff7f7f','drama':'#ff9999','thriller':'#4daf4a','sf':'#377eb8','action':'#984ea3',
            'hist_war':'#a65628','comedy':'#fdae61','society':'#80cdc1','family':'#8dd3c7','etc_g':'#b3b3b3'
        }
        bar_colors = [color_map.get(str(g).lower(), default_color) for g in gdf2['장르']]

        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.set_ylim(0, gdf2['작품 수'].max()*1.07)
        bars = ax1.bar(gdf2['장르'], gdf2['작품 수'], color=bar_colors, alpha=0.85, edgecolor='white')
        ax1.set_ylabel('작품 수'); ax1.set_xlabel('장르')
        ax1.set_xticklabels(gdf2['장르'], rotation=45, ha='right'); ax1.grid(axis='y', ls='--', alpha=0.5)
        pad = gdf2['작품 수'].max()*0.02
        for i,v in enumerate(gdf2['작품 수']):
            ax1.text(i, v+pad, f"{int(v)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2 = ax1.twinx()
        ax2.plot(gdf2['장르'], gdf2['평균 점수'], color='tab:blue', marker='o', lw=2, label='평균 점수')
        ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue')
        ax2.set_ylim(gdf2['평균 점수'].min()-0.02, gdf2['평균 점수'].max()+0.02)
        for i,v in enumerate(gdf2['평균 점수']):
            ax2.text(i, v+0.005, f"{v:.3f}", color='tab:blue', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.title('장르별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)
        st.markdown("🔎 **인사이트**: 로맨스/드라마는 작품 수 대비 평점이 낮고, 스릴러·SF·액션·전쟁(hist_war)은 높은 평점 경향.")
    else:
        st.info("장르/점수 컬럼이 없어 건너뜀.")

    # -------- 6) 플랫폼 --------
    st.subheader("6) 플랫폼별 작품수 및 평균 점수")
    if plat_col and score_col and df[plat_col].notna().any():
        ptmp = df[[plat_col, score_col]].copy()
        ptmp[plat_col] = ptmp[plat_col].apply(_ensure_list)
        pex = ptmp.explode(plat_col).dropna(subset=[plat_col])
        pex['_plat'] = pex[plat_col].astype(str).str.strip()
        platform_count = pex['_plat'].value_counts()
        platform_score = pex.groupby('_plat')[score_col].mean().round(3)
        pdf = (pd.DataFrame({'작품 수': platform_count, '평균 점수': platform_score})
               .reset_index().rename(columns={'index':'플랫폼'})
               .sort_values('작품 수', ascending=False).reset_index(drop=True))
        norm = pdf['플랫폼'].str.strip().str.lower()
        mask_last = norm.eq('etc_p')
        pdf = pd.concat([pdf[~mask_last], pdf[mask_last]], ignore_index=True)

        default_color = '#e5e7eb'
        cmap = {
            'KBS':'#ff7f7f','KBS2':'#6366f1','MBC':'#10b981','SBS':'#f59e0b','JTBC':'#8b5cf6','TVN':'#ef4444','OCN':'#f97316',
            'ENA':'#0ea5e9','MBN':'#84cc16','CHANNEL A':'#06b6d4','NETFLIX':'#dc2626','WAVVE':'#2563eb','TVING':'#e11d48','ETC_P':'#9ca3af'
        }
        upp = pdf['플랫폼'].astype(str).str.upper()
        bar_colors = [cmap.get(x, default_color) for x in upp]

        fig, ax1 = plt.subplots(figsize=(11,6))
        x = np.arange(len(pdf))
        bars = ax1.bar(x, pdf['작품 수'], color=bar_colors, alpha=0.9, edgecolor='white', linewidth=0.5)
        ax1.set_ylabel('작품 수'); ax1.set_xlabel('플랫폼')
        ax1.set_xticks(x); ax1.set_xticklabels(pdf['플랫폼'], rotation=45, ha='right')
        ax1.set_ylim(0, pdf['작품 수'].max()*1.13); ax1.grid(axis='y', ls='--', alpha=0.5)
        for i,b in enumerate(bars):
            v = pdf.loc[i,'작품 수']; ax1.text(b.get_x()+b.get_width()/2, v+pdf['작품 수'].max()*0.015, f"{int(v)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2 = ax1.twinx()
        if pdf['평균 점수'].notna().any():
            y = pdf['평균 점수'].values; ymin,ymax = float(pdf['평균 점수'].min()), float(pdf['평균 점수'].max())
            ax2.set_ylim(ymin-0.02, ymax+0.02)
            ax2.plot(x, y, color='tab:blue', marker='o', lw=2, label='평균 점수')
            for i,val in enumerate(y):
                if pd.notna(val): ax2.text(i, val+0.005, f"{val:.3f}", color='tab:blue', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue'); ax2.legend(loc='upper right', fontsize=9)
        plt.title('플랫폼별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)
        st.markdown("🔎 **인사이트**: 지상파는 작품 수 대비 평점이 낮고, tvN은 작품 수/평점 모두 우수하며, **NETFLIX**의 평균 평점이 두드러짐.")
    else:
        st.info("플랫폼/점수 컬럼이 없어 건너뜀.")

    # -------- 7) 방영 시기(연도) --------
    st.subheader("7) 방영 시기별 작품수 및 평균 점수")
    if score_col and year_col and df[year_col].notna().any():
        dfy = df[[year_col, score_col]].dropna().copy()
        dfy[year_col] = pd.to_numeric(dfy[year_col], errors='coerce')
        dfy = dfy.dropna(subset=[year_col]).astype({year_col:int})
        mean_by_year = dfy.groupby(year_col)[score_col].mean().round(3).sort_index()
        count_by_year = dfy[year_col].value_counts().sort_index()

        fig, ax1 = plt.subplots(figsize=(12,6))
        color1 = 'dimgray'
        ax1.set_xlabel('방영 시기'); ax1.set_ylabel('작품 수', color=color1)
        ax1.plot(count_by_year.index, count_by_year.values, marker='o', color=color1, alpha=0.85, label='작품 수')
        ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(axis='y', ls='--', alpha=0.4)
        for x_, y_ in zip(count_by_year.index, count_by_year.values):
            ax1.text(x_, y_+0.5, f"{int(y_)}", ha='center', va='bottom', fontsize=9, color=color1)

        ax2 = ax1.twinx()
        color2 = 'slateblue'
        ymin,ymax = float(mean_by_year.min()), float(mean_by_year.max())
        ax2.set_ylabel('평균 점수', color=color2)
        ax2.plot(mean_by_year.index, mean_by_year.values, marker='o', color=color2, lw=2, label='평균 점수')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(ymin-0.05, ymax+0.05)
        for x_, y_ in zip(mean_by_year.index, mean_by_year.values):
            ax2.text(x_, y_+0.01, f"{y_:.3f}", ha='center', va='bottom', fontsize=9, color=color2)
        plt.title('방영 시기별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)
        st.markdown("🔎 **인사이트**: 2018~2020년 사이 평균 점수 상승. 2019년 전후 OTT 투자로 촬영 퀄리티 향상 → 평점 상승 추정.")
    else:
        st.info("방영년도/점수 컬럼이 없어 건너뜀.")

    # -------- 8) 주연 배우 혼인 상태 --------
    st.subheader("8) 주연배우의 혼인 상태별 평점 차이")
    if score_col and role_col and married_col:
        def _is_lead(v): return any(k in str(v).lower() for k in ['주연','lead','main'])
        mdf = df[df[role_col].apply(_is_lead)].copy()
        if not mdf.empty and mdf[married_col].notna().any():
            mdf['_mar'] = mdf[married_col].apply(lambda x: '미혼' if str(x).strip()=='미혼' else '미혼 외')
            avg_by_m = mdf.groupby('_mar')[score_col].mean().round(3)
            cnt_by_m = mdf['_mar'].value_counts()
            order = [x for x in ['미혼','미혼 외'] if x in cnt_by_m.index]
            colors = ['orange' if s=='미혼' else 'gray' for s in order]

            fig, ax1 = plt.subplots(figsize=(6,5))
            bars = ax1.bar(order, cnt_by_m[order].values, color=colors, alpha=0.7)
            ax1.set_ylabel('작품 수'); ax1.set_xlabel('혼인 상태')
            ax1.set_ylim(0, cnt_by_m.max()*1.2); ax1.grid(axis='y', ls='--', alpha=0.5)
            for i,v in enumerate(cnt_by_m[order].values):
                ax1.text(i, v+0.5, f"{int(v)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax2 = ax1.twinx()
            ymin,ymax = float(avg_by_m.min()), float(avg_by_m.max())
            ax2.set_ylim(ymin-0.05, ymax+0.05)
            ax2.plot(order, avg_by_m[order].values, color='tab:blue', marker='o', lw=2, label='평균 점수')
            for i,v in enumerate(avg_by_m[order].values):
                ax2.text(i, v+0.005, f"{v:.3f}", color='tab:blue', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue'); ax2.legend(loc='upper right', fontsize=9)
            plt.title('주연배우 혼인 상태별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)
            st.markdown("🔎 **인사이트**: 미혼 배우가 주연 캐스팅에서 이점이 있으며, 평점도 혼인 여부별 차이가 관찰됨.")
        else:
            st.info("주연/결혼여부 데이터가 부족하여 건너뜀.")
    else:
        st.info("역할/결혼여부/점수 컬럼이 없어 건너뜀.")

    # -------- 9) 주연 배우 성별×혼인 상태 --------
    st.subheader("9) 주연배우 남자/여자 혼인 상태별 작품수 및 평균 점수")
    if score_col and role_col and married_col and gender_col:
        def _is_lead(v): return any(k in str(v).lower() for k in ['주연','lead','main'])
        mdf = df[(df[role_col].apply(_is_lead)) &
                 (df[gender_col].notna()) &
                 (df[married_col].notna()) &
                 (df[score_col].notna())].copy()
        if not mdf.empty:
            mdf['_gender'] = mdf[gender_col].astype(str).str.lower().map({'남자':'남자','여자':'여자','male':'남자','female':'여자'})
            mdf['_mar'] = mdf[married_col].apply(lambda x: '미혼' if str(x).strip()=='미혼' else '미혼 외')
            mdf['_grp'] = mdf['_gender'] + '-' + mdf['_mar']
            avg_by_grp = mdf.groupby('_grp')[score_col].mean().round(3)
            cnt_by_grp = mdf['_grp'].value_counts()
            order = [g for g in ['남자-미혼','남자-미혼 외','여자-미혼','여자-미혼 외'] if g in cnt_by_grp.index]
            cmap = {'남자-미혼':'dodgerblue','남자-미혼 외':'gray','여자-미혼':'hotpink','여자-미혼 외':'gray'}
            bar_colors = [cmap[g] for g in order]

            fig, ax1 = plt.subplots(figsize=(10,6))
            bars = ax1.bar(order, cnt_by_grp[order].values, color=bar_colors, alpha=0.75)
            ax1.set_ylabel('작품 수'); ax1.set_xlabel('성별-혼인 상태')
            ax1.set_ylim(0, cnt_by_grp[order].max()*1.25); ax1.grid(axis='y', ls='--', alpha=0.5)
            for g,b in zip(order, bars):
                v = cnt_by_grp[g]; ax1.text(b.get_x()+b.get_width()/2, v+cnt_by_grp[order].max()*0.03, f"{int(v)}", ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax2 = ax1.twinx()
            ymin,ymax = float(avg_by_grp[order].min()), float(avg_by_grp[order].max())
            ax2.set_ylim(ymin-0.05, ymax+0.05)
            # 남/여 각자 선으로 연결
            male_order   = [g for g in order if g.startswith('남자')]
            female_order = [g for g in order if g.startswith('여자')]
            ax2.plot(male_order,   avg_by_grp[male_order].values,   color='tab:blue', marker='o', lw=2, label='평균 점수')
            ax2.plot(female_order, avg_by_grp[female_order].values, color='tab:blue', marker='o', lw=2)
            for g in order:
                v = avg_by_grp[g]; ax2.text(g, v+0.005, f"{v:.3f}", color='tab:blue', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue'); ax2.legend(loc='upper right', fontsize=9)
            plt.title('주연배우 남자/여자 혼인 상태별 작품수 및 평균 점수'); st.pyplot(fig, use_container_width=True)

            # 간단 비교 인사이트 (데이터가 있으면 차이 계산)
            try:
                male_diff = float(avg_by_grp['남자-미혼'] - avg_by_grp['남자-미혼 외']) if {'남자-미혼','남자-미혼 외'} <= set(avg_by_grp.index) else np.nan
                female_diff = float(avg_by_grp['여자-미혼'] - avg_by_grp['여자-미혼 외']) if {'여자-미혼','여자-미혼 외'} <= set(avg_by_grp.index) else np.nan
                if np.isfinite(male_diff) and np.isfinite(female_diff):
                    st.markdown(f"🔎 **인사이트**: 남녀 모두 미혼 집단의 평균 평점이 더 높음. 여성(Δ≈{female_diff:.3f})이 남성(Δ≈{male_diff:.3f})보다 격차가 큼.")
                else:
                    st.markdown("🔎 **인사이트**: 남녀 모두 미혼 집단이 상대적으로 높은 평균 평점을 보이는 경향.")
            except Exception:
                st.markdown("🔎 **인사이트**: 남녀 모두 미혼 집단이 상대적으로 높은 평균 평점을 보이는 경향.")
        else:
            st.info("주연/성별/결혼여부/점수 유효 데이터가 부족하여 건너뜀.")
    else:
        st.info("역할/성별/결혼여부/점수 컬럼이 없어 건너뜀.")

def page_filter():
    st.header("실시간 필터")
    smin = float(pd.to_numeric(raw_df['score'], errors='coerce').min())
    smax = float(pd.to_numeric(raw_df['score'], errors='coerce').max())
    sfilter = st.slider("최소 평점", smin, smax, smin)
    y_min = int(pd.to_numeric(raw_df['start airing'], errors='coerce').min())
    y_max = int(pd.to_numeric(raw_df['start airing'], errors='coerce').max())
    yfilter = st.slider("방영년도 범위", y_min, y_max, (y_min, y_max))
    filt = raw_df[(pd.to_numeric(raw_df['score'], errors='coerce')>=sfilter) &
                  pd.to_numeric(raw_df['start airing'], errors='coerce').between(*yfilter)]
    st.dataframe(filt.head(20), use_container_width=True)

def page_all():
    st.header("원본 전체보기")
    st.dataframe(raw_df, use_container_width=True)

def make_pipeline(model_name, kind, estimator):
    if kind == "tree":
        return Pipeline([('preprocessor', preprocessor), ('model', estimator)])
    if model_name == "SVR":
        return Pipeline([('preprocessor', preprocessor), ('scaler', StandardScaler()), ('model', estimator)])
    if model_name == "KNN":
        return Pipeline([('preprocessor', preprocessor), ('poly', PolynomialFeatures(include_bias=False)),
                         ('scaler', StandardScaler()), ('knn', estimator)])
    if model_name == "Linear Regression (Poly)":
        return Pipeline([('preprocessor', preprocessor), ('poly', PolynomialFeatures(include_bias=False)),
                         ('scaler', StandardScaler()), ('linreg', estimator)])
    return Pipeline([('preprocessor', preprocessor), ('poly', PolynomialFeatures(include_bias=False)),
                     ('scaler', StandardScaler()), ('model', estimator)])

def page_tuning():
    st.header("GridSearchCV 튜닝")
    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X_colab_base, y_all, test_size=0.2, random_state=SEED, shuffle=True
        )
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(0.2)
    X_train, X_test, y_train, y_test = st.session_state["split_colab"]

    # Safe Mode 안내
    if SAFE_MODE:
        st.info("🛟 Safe Mode: 데이터가 크면 자동 샘플링하고, 과도한 GridSearch를 차단합니다. "
                "환경변수 CHEMI_SAFE_MODE=0 으로 해제할 수 있어요.")

    scoring = st.selectbox("스코어링", ["neg_root_mean_squared_error", "r2"], index=0)
    cv = st.number_input("CV 폴드 수", 3, 5 if SAFE_MODE else 10, 5 if SAFE_MODE else 5, 1)
    cv_shuffle = st.checkbox("CV 셔플(shuffle)", value=False)

    # --- 파라미터 선택기 ---
    def render_param_selector(label, options):
        display_options, to_py = [], {}
        for v in options:
            if v is None: s="(None)"; to_py[s]=None
            else:
                s = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
                to_py[s] = v
            display_options.append(s)
        sel = st.multiselect(f"{label}", display_options, default=display_options, key=f"sel_{label}")
        extra = st.text_input(f"{label} 추가값(콤마, 예: 50,75,100 또는 None)", value="", key=f"extra_{label}")
        chosen = [to_py[s] for s in sel]
        if extra.strip():
            for tok in extra.split(","):
                t = tok.strip()
                if not t: continue
                if t.lower()=="none": val=None
                else:
                    try: val=int(t)
                    except:
                        try: val=float(t)
                        except: val=t
                chosen.append(val)
        uniq=[];  [uniq.append(v) for v in chosen if v not in uniq]
        return uniq

    # --- 모델 목록 ---
    model_zoo = {
        "KNN": ("nonsparse", KNeighborsRegressor()),
        "Linear Regression (Poly)": ("nonsparse", LinearRegression()),
        "Ridge": ("nonsparse", Ridge()),
        "Lasso": ("nonsparse", Lasso()),
        "ElasticNet": ("nonsparse", ElasticNet(max_iter=10000)),
        "SGDRegressor": ("nonsparse", SGDRegressor(max_iter=10000, random_state=SEED)),
        "SVR": ("nonsparse", SVR()),
        "Decision Tree": ("tree", DecisionTreeRegressor(random_state=SEED)),
        "Decision Tree (Pruned)": ("tree", DecisionTreeRegressor(random_state=SEED)),
        "Random Forest": ("tree", RandomForestRegressor(random_state=SEED)),
    }
    if XGB_AVAILABLE:
        model_zoo["XGBRegressor"] = ("tree", XGBRegressor(
            random_state=SEED, objective="reg:squarederror", n_jobs=1, tree_method="hist"
        ))

    # --- 기본 그리드 ---
    default_param_grids = {
        "KNN": {"poly__degree":[2,3], "knn__n_neighbors":[3,4,5]},
        "Linear Regression (Poly)": {"poly__degree":[1,2,3]},
        "Ridge": {"poly__degree":[2,3], "model__alpha":[0.1,1,10,100,1000]},
        "Lasso": {"poly__degree":[2,3], "model__alpha":[0.001,0.01,0.1,1]},
        "ElasticNet": {"poly__degree":[2,3], "model__alpha":[0.001,0.01,0.1,1], "model__l1_ratio":[0.1,0.5,0.9]},
        "SGDRegressor": {"poly__degree":[1,2,3], "model__learning_rate":["constant","invscaling","adaptive"]},
        "SVR": {"model__kernel":["poly","rbf","sigmoid"], "model__degree":[1,2,3]},
        "Decision Tree": {
            "model__max_depth":[10,15,20],
            "model__min_samples_split":[5,6,7,8],
            "model__min_samples_leaf":[2,3],
            "model__max_leaf_nodes":[None,10,20],
        },
        "Decision Tree (Pruned)": {
            "model__ccp_alpha": [0.0, 3.146231327807963e-05, 7.543988269811632e-05]
        },
        "Random Forest": {
            "model__n_estimators":[200,300],
            "model__min_samples_split":[5,6,7,8],
            "model__max_depth":[10,15,20,25],
        },
    }
    if "XGBRegressor" in model_zoo:
        default_param_grids["XGBRegressor"] = {
            "model__n_estimators":[100, 200],
            "model__max_depth":[1, 5, 10],
            "model__learning_rate":[0.1,0.2,0.3],
            "model__colsample_bytree":[0.8,1.0],
        }

    model_name = st.selectbox("튜닝할 모델 선택", list(model_zoo.keys()), index=0)
    kind, estimator = model_zoo[model_name]
    pipe = make_pipeline(model_name, kind, estimator)

    st.markdown("**하이퍼파라미터 선택**")
    base_grid = dict(default_param_grids.get(model_name, {}))

    # Pruned 알파 후보 자동 생성 + 상한
    if model_name == "Decision Tree (Pruned)":
        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        tmp_tree = DecisionTreeRegressor(random_state=SEED)
        path = tmp_tree.cost_complexity_pruning_path(X_train_transformed, y_train)
        ccp_alphas = np.array(path.ccp_alphas, dtype=float)
        ccp_alphas = ccp_alphas[ccp_alphas >= 0.0]
        if ccp_alphas.size > 0:
            ccp_alphas = np.unique(ccp_alphas)
        if ccp_alphas.size > 1:
            ccp_alphas = ccp_alphas[:-1]
        must_include = np.array([3.146231327807963e-05, 7.543988269811632e-05], dtype=float)
        ccp_candidates = np.unique(np.concatenate([ccp_alphas, must_include]))
        # 상한
        if SAFE_MODE and len(ccp_candidates) > SAFE_MAX_PRUNED_ALPHAS:
            idx = np.linspace(0, len(ccp_candidates)-1, SAFE_MAX_PRUNED_ALPHAS).astype(int)
            ccp_candidates = ccp_candidates[idx]
        base_grid["model__ccp_alpha"] = list(ccp_candidates.tolist())
        st.caption(f"ccp_alpha 후보: {len(base_grid['model__ccp_alpha'])}개")

    # 파라미터 선택 UI
    def _render(label, options):
        # Safe mode일 땐 옵션을 그대로 사용(불필요한 폭증 방지)
        return options if SAFE_MODE else render_param_selector(label, options)
    user_grid = {k: _render(k, v) for k, v in base_grid.items()}

    with st.expander("선택한 파라미터 확인"):
        st.write(user_grid)

    # =========================
    # 실행 버튼 (+쿨다운/안전실행/과다탐색 차단)
    # =========================
    if st.button("GridSearch 실행", key="btn_gs"):
        if not _cooldown_ok("last_gs_time", SAFE_GS_COOLDOWN_SEC):
            st.warning("잠깐만요! 연속 실행을 잠시 제한 중이에요. 몇 초 후 다시 눌러 주세요.")
            st.stop()

        if model_name != "Decision Tree (Pruned)":
            # 과다 탐색 방지
            combos = count_param_evals(user_grid)
            total_evals = combos * int(cv)
            if SAFE_MODE and total_evals > SAFE_MAX_GSCV_EVALS:
                st.error(f"탐색량이 너무 큽니다: 조합 {combos} × CV {int(cv)} = {total_evals} > {SAFE_MAX_GSCV_EVALS}\n"
                         f"→ 옵션/폴드 수를 줄이거나 Safe Mode 해제(CHEMI_SAFE_MODE=0) 후 소규모로 실행하세요.")
                st.stop()

            cv_obj = KFold(n_splits=int(cv), shuffle=True, random_state=SEED) if cv_shuffle else int(cv)
            gs = GridSearchCV(estimator=pipe, param_grid=user_grid, cv=cv_obj,
                              scoring=scoring, n_jobs=1, refit=True, return_train_score=True)
            with st.spinner("GridSearchCV 실행 중..."):
                run_safely(gs.fit, X_train, y_train)

            st.subheader("베스트 결과")
            st.write("Best Params:", gs.best_params_)
            if scoring == "neg_root_mean_squared_error":
                st.write("Best CV RMSE (음수):", gs.best_score_)
            else:
                st.write(f"Best CV {scoring}:", gs.best_score_)

            y_pred_tr = gs.predict(X_train); y_pred_te = gs.predict(X_test)
            st.write("Train RMSE:", rmse(y_train, y_pred_tr))
            st.write("Test RMSE:", rmse(y_test, y_pred_te))
            st.write("Train R² Score:", r2_score(y_train, y_pred_tr))
            st.write("Test R² Score:", r2_score(y_test, y_pred_te))

            st.session_state["best_estimator"] = gs.best_estimator_
            st.session_state["best_params"] = gs.best_params_
            st.session_state["best_name"] = model_name
            st.session_state["best_cv_score"] = gs.best_score_
            st.session_state["best_scoring"] = scoring
            st.session_state["best_split_key"] = st.session_state.get("split_key")

            cvres = pd.DataFrame(gs.cv_results_)
            safe_cols = [c for c in ["rank_test_score","mean_test_score","std_test_score",
                                     "mean_train_score","std_train_score","params"] if c in cvres.columns]
            sorted_cvres = cvres.loc[:, safe_cols].sort_values("rank_test_score").reset_index(drop=True)
            st.dataframe(sorted_cvres, use_container_width=True)
            st.session_state["last_cvres"] = cvres

            if model_name == "XGBRegressor" and not XGB_AVAILABLE:
                st.warning("xgboost가 설치되어 있지 않습니다. requirements.txt에 `xgboost`를 추가하고 재배포해 주세요.")
            return

        # --- Pruned: 수동 스윕 ---
        with st.spinner("Cost-Complexity Pruning 실행 중..."):
            X_train_t = preprocessor.fit_transform(X_train, y_train)
            X_test_t  = preprocessor.transform(X_test)

            cand = user_grid.get("model__ccp_alpha", [])
            # Safe Mode 상한
            if SAFE_MODE and len(cand) > SAFE_MAX_PRUNED_ALPHAS:
                idx = np.linspace(0, len(cand)-1, SAFE_MAX_PRUNED_ALPHAS).astype(int)
                cand = [float(cand[i]) for i in idx]

            results = []
            for a in cand:
                m = DecisionTreeRegressor(random_state=SEED, ccp_alpha=float(a))
                run_safely(m.fit, X_train_t, y_train)
                ytr = m.predict(X_train_t); yte = m.predict(X_test_t)
                results.append({
                    "alpha": float(a),
                    "train_rmse": rmse(y_train, ytr),
                    "test_rmse":  rmse(y_test,  yte),
                    "train_r2":   float(r2_score(y_train, ytr)),
                    "test_r2":    float(r2_score(y_test,  yte)),
                    "estimator":  m
                })

            df_res = pd.DataFrame(results).sort_values("alpha").reset_index(drop=True)
            best_idx = int(df_res["test_r2"].idxmax())
            best_row = df_res.loc[best_idx]

        st.subheader("베스트 결과 (노트북 방식)")
        st.write("Best Params:\n\n", {"model__ccp_alpha": best_row["alpha"]})
        st.write("Train RMSE:", best_row["train_rmse"])
        st.write("Test RMSE:",  best_row["test_rmse"])
        st.write("Train R² Score:", best_row["train_r2"])
        st.write("Test R² Score:",  best_row["test_r2"])

        best_alpha = float(best_row["alpha"])
        best_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', DecisionTreeRegressor(random_state=SEED, ccp_alpha=best_alpha))
        ])
        best_pipeline.fit(X_train, y_train)

        st.session_state["best_estimator"] = best_pipeline
        st.session_state["best_params"] = {"model__ccp_alpha": best_alpha}
        st.session_state["best_name"] = "Decision Tree (Pruned) - NotebookStyle"
        st.session_state["best_cv_score"] = None
        st.session_state["best_scoring"] = "test_r2_max"
        st.session_state["best_split_key"] = st.session_state.get("split_key")

        st.markdown("**alpha sweep 로그**")
        st.dataframe(df_res[["alpha","train_rmse","test_rmse","train_r2","test_r2"]], use_container_width=True)

def page_ml():
    st.header("머신러닝 모델링")
    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X_colab_base, y_all, test_size=0.2, random_state=SEED, shuffle=True
        )
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(0.2)
    X_train, X_test, y_train, y_test = st.session_state["split_colab"]

    if "best_estimator" in st.session_state:
        model = st.session_state["best_estimator"]
        st.caption(f"현재 모델: GridSearch 베스트 모델 사용 ({st.session_state.get('best_name')})")
        if st.session_state.get("best_split_key") != st.session_state.get("split_key"):
            st.warning("주의: 베스트 모델은 이전 분할로 학습됨. 새 분할로 다시 튜닝해 주세요.", icon="⚠️")
    else:
        model = Pipeline([('preprocessor', preprocessor),
                          ('model', RandomForestRegressor(random_state=SEED))])
        model.fit(X_train, y_train)
        st.caption("현재 모델: 기본 RandomForest (미튜닝)")

    y_pred_tr = model.predict(X_train); y_pred_te = model.predict(X_test)
    st.metric("Train R²", f"{r2_score(y_train, y_pred_tr):.3f}")
    st.metric("Test  R²", f"{r2_score(y_test,  y_pred_te):.3f}")
    st.metric("Train RMSE", f"{rmse(y_train, y_pred_tr):.3f}")
    st.metric("Test  RMSE", f"{rmse(y_test,  y_pred_te):.3f}")

    if "best_params" in st.session_state:
        with st.expander("베스트 하이퍼파라미터 보기"):
            st.json(st.session_state["best_params"])

def page_predict():
    st.header("평점 예측")

    genre_opts   = sorted({g for sub in raw_df['genres'].dropna().apply(clean_cell_colab) for g in sub})
    week_opts    = sorted({d for sub in raw_df['day'].dropna().apply(clean_cell_colab) for d in sub})
    plat_opts    = sorted({p for sub in raw_df['network'].dropna().apply(clean_cell_colab) for p in sub})
    gender_opts  = sorted(raw_df['gender'].dropna().unique())
    role_opts    = sorted(raw_df['role'].dropna().unique())
    quarter_opts = sorted(raw_df['air_q'].dropna().unique())
    married_opts = sorted(raw_df['married'].dropna().unique())

    st.subheader("1) 입력")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**① 컨텐츠 특성**")
        input_age     = st.number_input("나이", 10, 80, 30)
        input_gender  = st.selectbox("성별", gender_opts) if gender_opts else st.text_input("성별 입력", "")
        input_role    = st.selectbox("역할", role_opts) if role_opts else st.text_input("역할 입력", "")
        input_married = st.selectbox("결혼여부", married_opts) if married_opts else st.text_input("결혼여부 입력", "")
        input_genre   = st.multiselect("장르 (멀티 선택)", genre_opts, default=genre_opts[:1] if genre_opts else [])
        derived_age_group = age_to_age_group(int(input_age))

        n_genre = len(input_genre)
        if n_genre == 0:  genre_bucket = "장르없음"
        elif n_genre <= 2: genre_bucket = "1~2개"
        elif n_genre <= 4: genre_bucket = "3~4개"
        elif n_genre <= 6: genre_bucket = "5~6개"
        else: genre_bucket = "7개 이상"
        st.caption(f"자동 연령대: **{derived_age_group}**  |  장르 개수: **{genre_bucket}**")

    with col_right:
        st.markdown("**② 편성 특성**")
        input_quarter = st.selectbox("방영분기", quarter_opts) if quarter_opts else st.text_input("방영분기 입력", "")
        input_week    = st.multiselect("방영요일 (멀티 선택)", week_opts, default=week_opts[:1] if week_opts else [])
        input_plat    = st.multiselect("플랫폼 (멀티 선택)", plat_opts, default=plat_opts[:1] if plat_opts else [])
        age_group_candidates = ["10대", "20대", "30대", "40대", "50대", "60대 이상"]
        data_age_groups = sorted(set(str(x) for x in raw_df.get("age_group", pd.Series([], dtype=object)).dropna().unique()))
        opts_age_group = data_age_groups if data_age_groups else age_group_candidates
        safe_index = 0 if not opts_age_group else min(1, len(opts_age_group)-1)
        target_age_group = st.selectbox("🎯 타겟 시청자 연령대",
                                        options=opts_age_group if opts_age_group else ["(데이터 없음)"],
                                        index=safe_index,
                                        key="target_age_group_main")
        st.session_state["target_age_group"] = target_age_group
        st.session_state["actor_age"] = int(input_age)

        # 예측 버튼(쿨다운)
        predict_btn = st.button("예측 실행", key="btn_predict")

    # ---- 예측 상태 유지/저장 ----
    if predict_btn:
        if not _cooldown_ok("last_predict_time", SAFE_PRED_COOLDOWN_SEC):
            st.info("연속 예측 요청을 잠시 제한하고 있어요. 잠시 후 다시 시도해 주세요.")
        else:
            # 1) 사용할 모델 결정
            if "best_estimator" in st.session_state:
                model_full = clone(st.session_state["best_estimator"])
                st.caption(f"예측 모델: GridSearch 베스트 재학습 사용 ({st.session_state.get('best_name')})")
            else:
                model_full = Pipeline([('preprocessor', preprocessor),
                                       ('model', RandomForestRegressor(n_estimators=100, random_state=SEED))])
                st.caption("예측 모델: 기본 RandomForest (미튜닝)")

            # 2) 전체 데이터로 재학습 (안전 실행)
            run_safely(model_full.fit, X_colab_base, y_all)

            # 3) 현재 입력으로 예측
            user_raw = pd.DataFrame([{
                'age': int(input_age), 'gender': input_gender, 'role': input_role, 'married': input_married,
                'air_q': input_quarter, 'age_group': derived_age_group,
                'genres': input_genre, 'day': input_week, 'network': input_plat, '장르구분': genre_bucket,
            }])
            st.session_state["target_age_group"] = st.session_state.get("target_age_group", derived_age_group)

            def _build_user_base_for_pred(df_raw: pd.DataFrame) -> pd.DataFrame:
                _user_mlb = colab_multilabel_transform(df_raw, cols=('genres','day','network'))
                _base = pd.concat([X_colab_base.iloc[:0].copy(), _user_mlb], ignore_index=True)
                _base = _base.drop(columns=[c for c in ['배우명','드라마명','genres','day','network','score','start airing'] if c in _base.columns], errors='ignore')
                for c in X_colab_base.columns:
                    if c not in _base.columns:
                        _base[c] = 0
                _base = _base[X_colab_base.columns].tail(1)
                num_cols_ = X_colab_base.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols_) > 0:
                    _base[num_cols_] = _base[num_cols_].apply(pd.to_numeric, errors="coerce")
                    _base[num_cols_] = _base[num_cols_].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                return _base

            user_base_now = _build_user_base_for_pred(user_raw)
            pred = float(run_safely(model_full.predict, user_base_now)[0])

            # 4) 세션 저장 (이후 위젯만 바꿔도 초기화되지 않음)
            st.session_state["cf_user_raw"] = user_raw.copy()
            st.session_state["cf_pred"] = float(pred)
            st.session_state["cf_model"] = model_full
            st.session_state["cf_inputs"] = {
                "age": int(input_age),
                "gender": input_gender,
                "role": input_role,
                "married": input_married,
                "air_q": input_quarter,
                "age_group": derived_age_group,
                "genres": list(input_genre),
                "day": list(input_week),
                "network": list(input_plat),
                "genre_bucket": genre_bucket,
            }

    # 이전 예측 상태 복구
    model_full = st.session_state.get("cf_model", None)
    user_raw   = st.session_state.get("cf_user_raw", None)
    pred       = st.session_state.get("cf_pred", None)

    if model_full is None or user_raw is None or pred is None:
        st.info("좌측 입력을 설정한 뒤 **[예측 실행]**을 눌러주세요.")
        return

    st.success(f"💡 예상 평점: {float(pred):.2f}")

    # =========================
    # 🔎 What-if (독립 액션 Top N)
    # =========================
    st.markdown("---")
    st.subheader("2) 케미스코어 평점 예측")

    target_age_group = st.session_state.get("target_age_group")
    if not target_age_group:
        target_age_group = "20대"
        st.session_state["target_age_group"] = target_age_group

    def _age_group_to_decade(s: str) -> int:
        m = re.search(r"(\d+)", str(s))
        if m:
            n = int(m.group(1))
            return 60 if "이상" in str(s) and n < 60 else n
        return 0

    actor_decade  = (int(st.session_state.get("actor_age", 30))//10)*10
    target_decade = _age_group_to_decade(target_age_group)
    gap = abs(actor_decade - target_decade)

    with st.container():
        st.markdown("**🎯 시청자-배우 연령대 정렬 가이드**")
        if target_decade <= 20:
            st.markdown("- 톤/장르: romance · comedy · action 위주, 가벼운 몰입 유도")
            st.markdown("- 편성: 토요일/주말 강세, 클립 중심 SNS 확산 고려")
        elif target_decade <= 30:
            st.markdown("- 톤/장르: romance/drama에 스릴러/미스터리 가미(하이브리드)")
            st.markdown("- 플랫폼: OTT 동시 공개로 화제성 확보")
        elif target_decade <= 40:
            st.markdown("- 톤/장르: drama / thriller / society 중심, 주제 밀도를 높임")
            st.markdown("- 편성: 주중 집중, 에피소드 퀄리티 변동 최소화")
        else:
            st.markdown("- 톤/장르: hist_war / family / society, 스토리 완성도·메시지 강화")
            st.mark론("- 편성: 시청 루틴 반영한 안정적 슬롯")

        if gap >= 20:
            st.info(f"배우 나이 {st.session_state.get('actor_age', 30)}세(≈{actor_decade}대) vs 타깃 {target_age_group} → **연령대 격차 큼**. 장르/편성/플랫폼을 타깃 성향에 맞춘 변경안의 우선순위를 높이세요.")
        else:
            st.caption(f"배우 나이 {st.session_state.get('actor_age', 30)}세(≈{actor_decade}대)와 타깃 {target_age_group}의 격차가 크지 않습니다.")

    def _build_user_base(df_raw: pd.DataFrame) -> pd.DataFrame:
        _user_mlb = colab_multilabel_transform(df_raw, cols=('genres','day','network'))
        _base = pd.concat([X_colab_base.iloc[:0].copy(), _user_mlb], ignore_index=True)
        _base = _base.drop(columns=[c for c in ['배우명','드라마명','genres','day','network','score','start airing'] if c in _base.columns], errors='ignore')
        for c in X_colab_base.columns:
            if c not in _base.columns:
                _base[c] = 0
        _base = _base[X_colab_base.columns].tail(1)
        num_cols_ = X_colab_base.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols_) > 0:
            _base[num_cols_] = _base[num_cols_].apply(pd.to_numeric, errors="coerce")
            _base[num_cols_] = _base[num_cols_].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return _base

    def _predict_from_raw(df_raw: pd.DataFrame) -> float:
        vb = _build_user_base(df_raw)
        return float(model_full.predict(vb)[0])

    current_pred = float(pred)

    def _classes_safe(key: str):
        return [s for s in (st.session_state.get(f"mlb_classes_{key}", []) or [])]

    genre_classes   = [g for g in _classes_safe("genres") if isinstance(g, str)]
    day_classes     = [d for d in _classes_safe("day") if isinstance(d, str)]
    network_classes = [n for n in _classes_safe("network") if isinstance(n, str)]

    # 액션 빌더
    def _add_genre(tag: str):
        def _fn(df):
            new = df.copy()
            cur = list(new.at[0, "genres"])
            if tag not in cur:
                cur = cur + [tag]
            new.at[0, "genres"] = cur
            return new
        return _fn

    def _set_days(days_list: List[str]):
        def _fn(df):
            new = df.copy()
            new.at[0, "day"] = days_list
            return new
        return _fn

    def _ensure_platform(p: str):
        def _fn(df):
            new = df.copy()
            cur = list(new.at[0, "network"])
            if p not in cur:
                cur = cur + [p]
            new.at[0, "network"] = cur
            return new
        return _fn

    def _set_role(val: str):
        def _fn(df):
            new = df.copy()
            new.at[0, "role"] = val
            return new
        return _fn

    def _set_married(val: str):
        def _fn(df):
            new = df.copy()
            new.at[0, "married"] = val
            return new
        return _fn

    # 후보 생성
    actions = []
    g_priority_by_target = {
        "young": ["romance", "comedy", "action", "thriller"],
        "adult": ["drama", "thriller", "hist_war", "romance", "comedy"],
        "senior": ["hist_war", "drama", "society", "thriller"]
    }
    if target_decade <= 30:
        glist = g_priority_by_target["young"]
    elif target_decade <= 40:
        glist = g_priority_by_target["adult"]
    else:
        glist = g_priority_by_target["senior"]
    for g in glist:
        if g in genre_classes:
            actions.append(("genre", f"장르 추가: {g}", _add_genre(g)))

    if "saturday" in day_classes:
        actions.append(("schedule", "편성 요일: 토요일 단일", _set_days(["saturday"])))
    if "friday" in day_classes:
        actions.append(("schedule", "편성 요일: 금요일 단일", _set_days(["friday"])))
    if "wednesday" in day_classes:
        actions.append(("schedule", "편성 요일: 수요일 단일", _set_days(["wednesday"])))

    if "NETFLIX" in network_classes:
        actions.append(("platform", "플랫폼 포함: NETFLIX", _ensure_platform("NETFLIX")))
    if "TVN" in network_classes:
        actions.append(("platform", "플랫폼 포함: TVN", _ensure_platform("TVN")))
    if "WAVVE" in network_classes:
        actions.append(("platform", "플랫폼 포함: WAVVE", _ensure_platform("WAVVE")))

    if "role" in user_raw.columns and str(user_raw.at[0, "role"]) != "주연":
        actions.append(("casting", "역할: 주연으로 변경", _set_role("주연")))
    if "married" in user_raw.columns and str(user_raw.at[0, "married"]) != "미혼":
        actions.append(("married", "결혼여부: 미혼으로 변경", _set_married("미혼")))

    # 액션 스코어링 (모델 재학습/초기화 없음 → 슬라이더만 바꿔도 예측 유지)
    scored = []
    for cat, desc, fn in actions:
        cand = fn(user_raw)
        p = _predict_from_raw(cand)
        scored.append({"카테고리": cat, "변경안": desc, "예측": p, "리프트": p - current_pred})

    if not scored:
        st.info("추천할 액션이 없습니다. (현실 제약/입력값으로 인해 후보가 없을 수 있어요)")
    else:
        df_scored = pd.DataFrame(scored)
        idx_best = df_scored.groupby("카테고리")["리프트"].idxmax()
        df_best_per_cat = df_scored.loc[idx_best].copy()

        # 🔻 여기 슬라이더를 바꿔도 current_pred/state는 유지되므로 초기화 없음
        top_n = st.slider("추천 개수", 3, 4, 3 if not SAFE_MODE else 3, key="rec_topn_slider")
        df_top = df_best_per_cat.sort_values(["리프트", "예측"], ascending=False).head(top_n).reset_index(drop=True)

        st.dataframe(df_top[["카테고리","변경안"]], use_container_width=True)

        st.markdown("#### 🔍 액션별 솔루션 요약")
        genre_reason = {
            "thriller": "긴장감·몰입도 상승 → 체류시간/평점 우호적",
            "hist_war": "작품성·완성도 포인트로 평점 상향",
            "sf": "신선한 소재/세계관으로 초반 흡입력↑",
            "action": "시각적 임팩트로 초반 만족도 상승",
            "romance": "대중성 높아 넓은 타깃 적합",
            "drama": "보편적 공감 서사로 안정적",
            "comedy": "가벼운 톤으로 폭넓은 수용층 확보",
            "society": "사회적 메시지·현실감으로 충성도↑",
        }
        day_reason = {
            "월요일": "한 주의 시작, 고정 시청층 확보와 입소문 형성 기회",
            "화요일": "주중 초반, 지속 시청 유도에 유리",
            "수요일": "주중 중앙부 집중 시청층 공략",
            "목요일": "주말 전환 직전, 긴장감 있는 전개로 시청률 상승 유도",
            "금요일": "주말 초입 노출로 회차 전환율 확보",
            "토요일": "시청 가용시간↑ → 몰입/구전 효과 기대",
            "일요일": "가족·전 연령대 타깃, 다음 주 시청 예고 효과 극대화"
        }
        platform_reason = {
            "ENT": "예능 중심 편성으로 대중적 화제성 확보 용이",
            "ETC_P": "기타 케이블 채널로 틈새 시청층 공략",
            "GPC": "전문 채널 특화 타깃 시청층 집중",
            "JTBC": "프라임 타임 드라마 강세, 화제성 높은 시리즈 제작 경험",
            "KBS": "전국 단위 지상파 커버리지, 전 세대 접근성 우수",
            "NETFLIX": "글로벌 노출/알고리즘 추천 → 화제성/리뷰 확보 용이",
            "SBS": "트렌디한 드라마·예능 라인업으로 젊은층 선호",
            "TVN": "프라임 편성·브랜딩 시너지",
            "MBC": "오랜 드라마 제작 전통, 안정적 시청층 확보"
        }
        etc_reason = {"주연": "캐릭터 공감/노출 극대화", "미혼": "로맨스/청춘물 톤 결합 시 몰입도↑"}

        def _explain(desc: str) -> str:
            why = []
            m = re.search(r"장르 추가:\s*([A-Za-z_]+)", desc)
            if m:
                g = m.group(1).lower()
                if g in genre_reason: why.append(f"장르 효과: {genre_reason[g]}")
                if target_decade <= 20 and g in {"romance","comedy","action"}: why.append("젊은 타깃과 톤 매칭 양호")
                if target_decade >= 40 and g in {"hist_war","drama","thriller","society"}: why.append("성숙 타깃 선호 주제와 부합")
            if "토요일" in desc or "saturday" in desc: why.append(f"편성 효과: {day_reason['토요일']}")
            if "금요일" in desc or "friday" in desc:   why.append(f"편성 효과: {day_reason['금요일']}")
            if "수요일" in desc or "wednesday" in desc:why.append(f"편성 효과: {day_reason['수요일']}")
            for k, v in platform_reason.items():
                if k in desc: why.append(f"플랫폼 효과: {v}")
            if "주연" in desc: why.append(f"캐스팅 효과: {etc_reason['주연']}")
            if "미혼" in desc: why.append(f"캐릭터 톤: {etc_reason['미혼']}")
            return " / ".join(why) if why else "데이터 기반 상 상승 요인"

        st.markdown("**📝 상위 변경안 솔루션**")
        for _, r in df_top.iterrows():
            st.markdown(f"- **{r['변경안']}** · {_explain(r['변경안'])}")

# ================== 사이드바 (네비 + 설정) ==================
NAV_ITEMS = [
    ("overview", "🏠", "개요",        page_overview),
    ("basic",    "📋", "기초통계",    page_basic),
    ("dist",     "📈", "분포/교차",   page_dist),
    ("filter",   "🛠️", "필터",        page_filter),
    ("all",      "🗂️", "전체보기",    page_all),
    ("tuning",   "🧪", "튜닝",        page_tuning),
    ("ml",       "🤖", "ML모델",      page_ml),
    ("predict",  "🎯", "예측",        page_predict),
]

if "nav" not in st.session_state:
    st.session_state["nav"] = _get_nav_from_query() or "overview"
current = st.session_state["nav"]

with st.sidebar:
    st.markdown('<div class="sb-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .sb-brand { display:flex; align-items:center; gap:8px; font-weight:900; }
    section[data-testid="stSidebar"] .sb-brand .logo { font-size:35px !important; line-height:1; }
    section[data-testid="stSidebar"] .sb-brand .name { font-size:26px !important; line-height:1.2; }
    </style>
    <div class="sb-brand">
        <span class="logo">💫</span>
        <span class="name">케미스코어</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-menu">', unsafe_allow_html=True)
    for slug, icon, label, _fn in NAV_ITEMS:
        active = (slug == current)
        st.markdown(f'<div class="sb-nav {"active" if active else ""}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{slug}", use_container_width=True):
            st.session_state["nav"] = slug
            _set_nav_query(slug)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    sm_text = "ON" if SAFE_MODE else "OFF"
    st.markdown(f'<div class="sb-safe"><h4>Safe Mode: {sm_text}</h4>', unsafe_allow_html=True)
    st.markdown(
    '<div class="sb-card sb-config">'
    '<div class="sb-card-title">모델 설정</div>'
    '<div class="meta">test_size = 0.2, random_state = 42</div>'
    '</div>',
    unsafe_allow_html=True
)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-footer">© Chemiscore • <span class="ver">v0.2</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ================== 라우팅 ==================
PAGES = {slug: fn for slug, _, _, fn in NAV_ITEMS}
PAGES.get(st.session_state["nav"], page_overview)()
