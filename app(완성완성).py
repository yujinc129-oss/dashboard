# =========================================
# app.py  —  K-드라마 케미스코어 (Sparrow UI 스킨 + 사이드 네비)
# =========================================

# ---- page config MUST be first ----
import streamlit as st
st.set_page_config(page_title="케미스코어 | K-드라마 분석/예측", page_icon="🎬", layout="wide")

# ---- dependency guard (optional) ----
import importlib.util
_missing = [m for m in ("numpy","scipy","sklearn","joblib","threadpoolctl","xgboost") if importlib.util.find_spec(m) is None]
if _missing:
    st.error(f"필수 라이브러리 미설치: {_missing}. requirements.txt / runtime.txt 버전을 고정해 재배포하세요.")
    st.stop()

# ---- imports ----
import os, ast, random, re, platform
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

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

# ====== Sparrow UI CSS ======
def _inject_sparrow_css():
    st.markdown("""
    <style>
      /* ---------- Layout / Typography ---------- */
      .block-container{padding-top:5rem; padding-bottom:5rem;}
      h1,h2,h3{font-weight:800;}
      /* 연도별 평균 케미스코어 그래프 위쪽 간격 */
    h3, .stPlotlyChart {
            margin-top: 1rem !important;
        }
      /* ---------- Topbar ---------- */
      .topbar{display:flex; align-items:flex-end; justify-content:space-between; margin:6px 0 14px;}
      .topbar .title{font-size:28px; letter-spacing:-.2px; display:flex; gap:10px; align-items:center;}
      .crumb{font-size:12px; color:#6b7280; margin-top:4px;}
      .top-right{display:flex; gap:8px; align-items:center;}
      .chip{background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:6px 10px; font-size:12px;}

     # ... 기존 CSS 함수 안에서 Sidebar 관련 부분만 아래로 교체 ...
      /* ---------- Sidebar ---------- */
      section[data-testid="stSidebar"]{
        width:220px !important; min-width:220px;
        background:#0b1220; color:#e5e7eb; border-right:1px solid #070c16;
      }
      .sb-wrap{display:flex; flex-direction:column; height:100%;}
      .sb-brand{display:flex; align-items:center; gap:10px; padding:14px 12px 10px;}
      .sb-brand .logo{font-size:20px}
      .sb-brand .name{font-size:16px; font-weight:800; letter-spacing:.2px}

      .sb-menu{padding:6px 8px 8px; display:flex; flex-direction:column;}
      .sb-nav{margin:2px 0;}             /* 버튼 간격 최소화 */
      .sb-nav .stButton>button{
        width:100% !important;
        display:flex; align-items:center; gap:10px; justify-content:flex-start;
        background:transparent !important;
        color:#e5e7eb !important;
        border:1px solid #162033 !important;
        border-radius:10px !important;
        padding:8px 10px !important;
        font-size:14px !important;
        box-shadow:none !important;
        opacity:1 !important;            /* 희미해 보이는 문제 방지 */
      }
      .sb-nav .stButton>button:hover{
        background:#111a2b !important;
        border-color:#25324a !important;
      }
      .sb-nav.active .stButton>button{
        background:#2563eb !important;
        border-color:#2563eb !important;
        color:#ffffff !important;
      }

      .sb-card{background:#0f172a; border:1px solid #1f2937; border-radius:12px; padding:10px; margin-top:8px;}
      .sb-card h4{margin:0 0 6px 0; font-size:12px; color:#cbd5e1; font-weight:800;}
      .sb-footer{margin-top:auto; padding:10px 12px; font-size:11px; color:#9ca3af; border-top:1px solid #070c16;}


      /* ---------- Cards ---------- */
      .kpi-row{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:8px 0 6px;}
      .kpi{
        background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px;
        box-shadow:0 6px 18px rgba(17,24,39,.04);
      }
      .kpi h6{margin:0 0 4px; font-size:12px; color:#6b7280; font-weight:800;}
      .kpi .v{font-size:22px; font-weight:800; line-height:1;}
      .kpi .d{font-size:12px; color:#10b981; font-weight:700;}

      /* Plot containers tighter top margin */
      div[data-testid="stPlotlyChart"], div.stPlot {margin-top:8px;}

      /* 메인 컨테이너 상단 여백 살짝 키워서 카드 잘림 방지 */
      .block-container{padding-top:1.4rem; padding-bottom:2.2rem;}
    
      /* KPI 줄과 다음 섹션 간 간격 */
      .kpi-row{ margin-bottom: 18px; }
    
      /* Plotly 차트 바깥쪽 여백 줄이기 + 기본 높이 */
      div[data-testid="stPlotlyChart"]{ margin-top:8px; }
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

@st.cache_data
def load_data():
    raw = pd.read_json('drama_d.json')
    if isinstance(raw, pd.Series):
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    else:
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
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
# 페이지 함수들은 아래에서 정의됩니다.
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
        st.subheader("최근 연도 상위 작품 (중복 제거)")
        _df = raw_df.copy()
        _df['start airing'] = pd.to_numeric(_df['start airing'], errors='coerce')
        _df['score'] = pd.to_numeric(_df['score'], errors='coerce')
        _df = _df.dropna(subset=['start airing', 'score'])
    
        if not _df.empty:
            last_year = int(_df['start airing'].max())
            # 최근 1년 또는 2년 범위 (원하면 범위 조정 가능)
            recent = _df[_df['start airing'].between(last_year-1, last_year)]
    
            name_col = '드라마명' if '드라마명' in recent.columns else (
                'title' if 'title' in recent.columns else recent.columns[0]
            )
    
            # 드라마명 기준 중복 제거 (가장 높은 점수만 남김)
            recent_unique = (
                recent.sort_values('score', ascending=False)
                      .drop_duplicates(subset=[name_col], keep='first')
            )
    
            top_recent = recent_unique.sort_values('score', ascending=False).head(10)
            fig_recent = px.bar(top_recent, x=name_col, y='score', text='score')
            fig_recent.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_recent.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=40))
            st.plotly_chart(fig_recent, use_container_width=True)
        else:
            st.info("최근 연도 데이터가 없습니다.")

    # (오른쪽) 플랫폼별 작품 수 TOP 10
    with c2:
        st.subheader("플랫폼별 작품 수 (TOP 10)")
        _p = raw_df.copy()
        _p['network'] = _p['network'].apply(clean_cell_colab)
        _p = _p.explode('network').dropna(subset=['network'])
        p_cnt = (
                raw_df.assign(network=raw_df["network"].apply(clean_cell_colab))
                          .explode("network")
                          .dropna(subset=["network"])
                          .groupby("network")
                          .size()
                          .reset_index(name="count")  # 중복 방지
                )
        p_cnt = p_cnt.loc[:, ~p_cnt.columns.duplicated()].copy()

        fig_p = px.bar(p_cnt, x='network', y='count')
        fig_p.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=40))
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
    st.subheader("연도별 주요 플랫폼 작품 수")
    ct = (
        pd.DataFrame({'start airing': raw_df['start airing'], 'network': raw_df['network'].apply(clean_cell_colab)})
        .explode('network').groupby(['start airing','network']).size().reset_index(name='count')
    )
    ct['NETWORK_UP'] = ct['network'].astype(str).str.upper()
    focus = ['KBS','MBC','TVN','NETFLIX','SBS']
    fig3 = px.line(ct[ct['NETWORK_UP'].isin(focus)], x='start airing', y='count', color='network',
                   log_y=True, title="연도별 주요 플랫폼 작품 수")
    st.plotly_chart(fig3, use_container_width=True)

    p = (ct.pivot_table(index='start airing', columns='NETWORK_UP', values='count', aggfunc='sum')
           .fillna(0).astype(int))
    years = sorted(p.index)
    insights = []
    if 'NETFLIX' in p.columns:
        s = p['NETFLIX']; nz = s[s > 0]
        if not nz.empty:
            first_year = int(nz.index.min())
            max_year, max_val = int(s.idxmax()), int(s.max())
            insights.append(f"- **넷플릭스(OTT)의 급성장**: {first_year}년 이후 증가, **{max_year}년 {max_val}편** 최고치.")
    down_ter = []
    for b in ['KBS','MBC','SBS']:
        if b in p.columns and len(years) >= 2:
            slope = np.polyfit(years, p[b].reindex(years, fill_value=0), 1)[0]
            if slope < 0: down_ter.append(b)
    if down_ter:
        insights.append(f"- **지상파 감소 추세**: {' / '.join(down_ter)} 전반적 하락.")
    st.markdown("**인사이트**\n" + "\n".join(insights))

    st.subheader("장르 개수별 평균 평점 (배우 단위)")
    actor_col = '배우명' if '배우명' in raw_df.columns else ('actor' if 'actor' in raw_df.columns else None)
    if actor_col is None:
        st.info("배우 식별 컬럼을 찾을 수 없어(배우명/actor) 이 섹션을 건너뜁니다.")
        return
    gdf = (
        pd.DataFrame({actor_col: raw_df[actor_col], 'genres': raw_df['genres'].apply(clean_cell_colab)})
        .explode('genres').dropna(subset=[actor_col,'genres'])
    )
    genre_cnt = gdf.groupby(actor_col)['genres'].nunique().rename('장르개수')
    actor_mean = (raw_df.groupby(actor_col, as_index=False)['score']
                  .mean().rename(columns={'score':'배우평균점수'}))
    df_actor = actor_mean.merge(genre_cnt.reset_index(), on=actor_col, how='left')
    df_actor['장르개수'] = df_actor['장르개수'].fillna(0).astype(int)
    df_actor = df_actor[df_actor['장르개수'] > 0].copy()
    def bucket(n: int) -> str:
        if n <= 2:  return '1~2개'
        if n <= 4:  return '3~4개'
        if n <= 6:  return '5~6개'
        return '7개 이상'
    df_actor['장르개수구간'] = pd.Categorical(
        df_actor['장르개수'].apply(bucket),
        categories=['1~2개','3~4개','5~6개','7개 이상'],
        ordered=True
    )
    fig_box = px.box(
        df_actor, x='장르개수구간', y='배우평균점수',
        category_orders={'장르개수구간': ['1~2개','3~4개','5~6개','7개 이상']},
        title="장르 개수별 배우 평균 점수 분포"
    )
    st.plotly_chart(fig_box, use_container_width=True)

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

    scoring = st.selectbox("스코어링", ["neg_root_mean_squared_error", "r2"], index=0)
    cv = st.number_input("CV 폴드 수", 3, 10, 5, 1)
    cv_shuffle = st.checkbox("CV 셔플(shuffle)", value=False)

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

    model_zoo = {
        "KNN": ("nonsparse", KNeighborsRegressor()),
        "Linear Regression (Poly)": ("nonsparse", LinearRegression()),
        "Ridge": ("nonsparse", Ridge()),
        "Lasso": ("nonsparse", Lasso()),
        "ElasticNet": ("nonsparse", ElasticNet(max_iter=10000)),
        "SGDRegressor": ("nonsparse", SGDRegressor(max_iter=10000, random_state=SEED)),
        "SVR": ("nonsparse", SVR()),
        "Decision Tree": ("tree", DecisionTreeRegressor(random_state=SEED)),
        "Random Forest": ("tree", RandomForestRegressor(random_state=SEED)),
    }
    if 'XGBRegressor' in globals() and XGB_AVAILABLE:
        model_zoo["XGBRegressor"] = ("tree", XGBRegressor(
            random_state=SEED, objective="reg:squarederror", n_jobs=-1, tree_method="hist"
        ))

    default_param_grids = {
        "KNN": {"poly__degree":[1,2,3], "knn__n_neighbors":[3,4,5,6,7,8,9,10]},
        "Linear Regression (Poly)": {"poly__degree":[1,2,3]},
        "Ridge": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000]},
        "Lasso": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000]},
        "ElasticNet": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000], "model__l1_ratio":[0.1,0.5,0.9]},
        "SGDRegressor": {"poly__degree":[1,2,3], "model__learning_rate":["constant","invscaling","adaptive"]},
        "SVR": {"model__kernel":["poly","rbf","sigmoid"], "model__degree":[1,2,3]},
        "Decision Tree": {"model__max_depth":[10,15,20,25,30], "model__min_samples_split":[5,6,7,8,9,10],
                          "model__min_samples_leaf":[2,3,4,5], "model__max_leaf_nodes":[None,10,20,30]},
        "Random Forest": {"model__n_estimators":[100,200,300], "model__min_samples_split":[5,6,7,8,9,10],
                          "model__max_depth":[5,10,15,20,25,30]},
    }
    if "XGBRegressor" in model_zoo:
        default_param_grids["XGBRegressor"] = {
            "model__n_estimators":[200,400],
            "model__max_depth":[3,5,7],
            "model__learning_rate":[0.03,0.1,0.3],
            "model__subsample":[0.8,1.0],
            "model__colsample_bytree":[0.8,1.0],
        }

    model_name = st.selectbox("튜닝할 모델 선택", list(model_zoo.keys()), index=0)
    kind, estimator = model_zoo[model_name]
    pipe = make_pipeline(model_name, kind, estimator)

    st.markdown("**하이퍼파라미터 선택**")
    base_grid = default_param_grids.get(model_name, {})
    user_grid = {k: render_param_selector(k, v) for k, v in base_grid.items()}

    with st.expander("선택한 파라미터 확인"):
        st.write(user_grid)

    if st.button("GridSearch 실행"):
        cv_obj = KFold(n_splits=int(cv), shuffle=True, random_state=SEED) if cv_shuffle else int(cv)
        gs = GridSearchCV(estimator=pipe, param_grid=user_grid, cv=cv_obj,
                          scoring=scoring, n_jobs=-1, refit=True, return_train_score=True)
        with st.spinner("GridSearchCV 실행 중..."):
            gs.fit(X_train, y_train)

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

    if model_name == "XGBRegressor" and not XGB_AVAILABLE:
        st.warning("xgboost가 설치되어 있지 않습니다. requirements.txt에 `xgboost`를 추가하고 재배포해 주세요.")

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
        target_age_group = st.selectbox("🎯 타깃 시청자 연령대",
                                        options=opts_age_group if opts_age_group else ["(데이터 없음)"],
                                        index=safe_index,
                                        key="target_age_group_main")
        st.session_state["target_age_group"] = target_age_group
        st.session_state["actor_age"] = int(input_age)
        predict_btn = st.button("예측 실행")

    if not predict_btn:
        return

    if "best_estimator" in st.session_state:
        model_full = clone(st.session_state["best_estimator"])
        st.caption(f"예측 모델: GridSearch 베스트 재학습 사용 ({st.session_state.get('best_name')})")
    else:
        model_full = Pipeline([('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(n_estimators=100, random_state=SEED))])
        st.caption("예측 모델: 기본 RandomForest (미튜닝)")
    model_full.fit(X_colab_base, y_all)

    user_raw = pd.DataFrame([{
        'age': int(input_age), 'gender': input_gender, 'role': input_role, 'married': input_married,
        'air_q': input_quarter, 'age_group': derived_age_group,
        'genres': input_genre, 'day': input_week, 'network': input_plat, '장르구분': genre_bucket,
    }])

    def _build_user_base(df_raw: pd.DataFrame) -> pd.DataFrame:
        _user_mlb = colab_multilabel_transform(df_raw, cols=('genres','day','network'))
        _base = pd.concat([X_colab_base.iloc[:0].copy(), _user_mlb], ignore_index=True)
        _base = _base.drop(columns=[c for c in drop_cols if c in _base.columns], errors='ignore')
        for c in X_colab_base.columns:
            if c not in _base.columns:
                _base[c] = 0
        _base = _base[X_colab_base.columns].tail(1)
        num_cols_ = X_colab_base.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols_) > 0:
            _base[num_cols_] = _base[num_cols_].apply(pd.to_numeric, errors="coerce")
            _base[num_cols_] = _base[num_cols_].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return _base

    user_base = _build_user_base(user_raw)
    pred = float(model_full.predict(user_base)[0])
    st.success(f"💡 예상 평점: {pred:.2f}")

# ✅ 카운터팩추얼에서 다시 쓰려고 세션에 보관
    st.session_state["cf_user_raw"] = user_raw.copy()
    st.session_state["cf_pred"] = float(pred)
    st.session_state["cf_model"] = model_full

                           # =========================
    # 🔎 What-if (독립 액션 Top N, 중복효과/합산/조합 금지)
    # =========================
    st.markdown("---")
    st.subheader("2) 케미스코어 평점 예측")

    # 🔹 기존에 만든 타깃 연령대 selectbox를 재사용 (중복 위젯 생성 금지)
    target_age_group = st.session_state.get("target_age_group")
    if not target_age_group:
        # 혹시 세션이 비어있다면, 상단 입력에서 기본 후보를 사용해 설정
        age_group_candidates = ["10대", "20대", "30대", "40대", "50대", "60대 이상"]
        target_age_group = age_group_candidates[1]
        st.session_state["target_age_group"] = target_age_group

    def _age_group_to_decade(s: str) -> int:
        m = re.search(r"(\d+)", str(s))
        if m:
            n = int(m.group(1))
            return 60 if "이상" in str(s) and n < 60 else n
        return 0

    actor_decade  = (int(input_age)//10)*10
    target_decade = _age_group_to_decade(target_age_group)
    gap = abs(actor_decade - target_decade)

    with st.container():
        st.markdown("**🎯 타깃-배우 연령대 정렬 가이드**")
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
            st.markdown("- 편성: 시청 루틴 반영한 안정적 슬롯")

        if gap >= 20:
            st.info(f"배우 나이 {input_age}세(≈{actor_decade}대) vs 타깃 {target_age_group} → **연령대 격차 큼**. "
                    "장르/편성/플랫폼을 타깃 성향에 맞춘 변경안의 우선순위를 높이세요.")
        else:
            st.caption(f"배우 나이 {input_age}세(≈{actor_decade}대)와 타깃 {target_age_group}의 격차가 크지 않습니다.")

    # ── 공통 유틸: user_raw → user_base(feature vector)
    def _build_user_base(df_raw: pd.DataFrame) -> pd.DataFrame:
        _user_mlb = colab_multilabel_transform(df_raw, cols=('genres','day','network'))
        _base = pd.concat([X_colab_base.iloc[:0].copy(), _user_mlb], ignore_index=True)
        _base = _base.drop(columns=[c for c in drop_cols if c in _base.columns], errors='ignore')
        for c in X_colab_base.columns:
            if c not in _base.columns:
                _base[c] = 0
        _base = _base[X_colab_base.columns].tail(1)
        # 숫자열만 숫자화/결측 보정
        num_cols_ = X_colab_base.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols_) > 0:
            _base[num_cols_] = _base[num_cols_].apply(pd.to_numeric, errors="coerce")
            _base[num_cols_] = _base[num_cols_].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return _base

    def _predict_from_raw(df_raw: pd.DataFrame) -> float:
        vb = _build_user_base(df_raw)
        return float(model_full.predict(vb)[0])

    current_pred = float(pred)  # 출발점

    # ── 후보 클래스 안전 추출
    def _classes_safe(key: str):
        return [s for s in (st.session_state.get(f"mlb_classes_{key}", []) or [])]

    genre_classes   = [g for g in _classes_safe("genres") if isinstance(g, str)]
    day_classes     = [d for d in _classes_safe("day") if isinstance(d, str)]
    network_classes = [n for n in _classes_safe("network") if isinstance(n, str)]

    # ── 액션 빌더 (카테고리: genre/schedule/platform/casting/married)
    def _add_genre(tag: str):
        def _fn(df):
            new = df.copy()
            cur = list(new.at[0, "genres"])
            if tag not in cur:
                cur = cur + [tag]
            new.at[0, "genres"] = cur
            return new
        return _fn

    def _set_days(days_list: list[str]):
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

    # ── 후보 생성 (현실 제약: 데이터에 없는 옵션은 자동 제외)
    actions = []

    # [장르] 타깃 연령대에 맞는 우선순위 큐 (데이터에 존재하는 것만)
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

    # [편성] 단일 요일 세팅 (있을 때만)
    if "saturday" in day_classes:
        actions.append(("schedule", "편성 요일: 토요일 단일", _set_days(["saturday"])))
    if "friday" in day_classes:
        actions.append(("schedule", "편성 요일: 금요일 단일", _set_days(["friday"])))
    if "wednesday" in day_classes:
        actions.append(("schedule", "편성 요일: 수요일 단일", _set_days(["wednesday"])))

    # [플랫폼] 존재하는 플랫폼만
    if "NETFLIX" in network_classes:
        actions.append(("platform", "플랫폼 포함: NETFLIX", _ensure_platform("NETFLIX")))
    if "TVN" in network_classes:
        actions.append(("platform", "플랫폼 포함: TVN", _ensure_platform("TVN")))
    if "WAVVE" in network_classes:
        actions.append(("platform", "플랫폼 포함: WAVVE", _ensure_platform("WAVVE")))

    # [캐스팅/결혼] 현재 값과 다를 때만 후보 추가
    if "role" in user_raw.columns and str(user_raw.at[0, "role"]) != "주연":
        actions.append(("casting", "역할: 주연으로 변경", _set_role("주연")))
    if "married" in user_raw.columns and str(user_raw.at[0, "married"]) != "미혼":
        actions.append(("married", "결혼여부: 미혼으로 변경", _set_married("미혼")))

    # ── 각 액션(단독 적용)의 리프트만 계산 → 카테고리별 최고 1개 선택
    scored = []
    for cat, desc, fn in actions:
        cand = fn(user_raw)
        p = _predict_from_raw(cand)
        scored.append({"카테고리": cat, "변경안": desc, "예측": p, "리프트": p - current_pred})

    if not scored:
        st.info("추천할 액션이 없습니다. (현실 제약/입력값으로 인해 후보가 없을 수 있어요)")
    else:
        df_scored = pd.DataFrame(scored)

        # 카테고리별 최고 리프트 1개만 남기기
        idx_best = df_scored.groupby("카테고리")["리프트"].idxmax()
        df_best_per_cat = df_scored.loc[idx_best].copy()

        # 최종 Top N (합산 금지, 독립 효과만)
        top_n = st.slider("추천 개수", 3, 7, 5, key="rec_topn_slider")
        df_top = df_best_per_cat.sort_values(["리프트", "예측"], ascending=False).head(top_n).reset_index(drop=True)

        # 테이블
        st.dataframe(
            df_top[["카테고리","변경안"]],
            use_container_width=True
        )

        # ── 액션별 솔루션 요약 (중복 설명은 카테고리 정의로 대체)
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
        
        etc_reason = {
            "주연": "캐릭터 공감/노출 극대화",
            "미혼": "로맨스/청춘물 톤 결합 시 몰입도↑"
        }


        def _explain(desc: str) -> str:
            why = []
            # 장르
            m = re.search(r"장르 추가:\s*([A-Za-z_]+)", desc)
            if m:
                g = m.group(1).lower()
                if g in genre_reason:
                    why.append(f"장르 효과: {genre_reason[g]}")
                if target_decade <= 20 and g in {"romance","comedy","action"}:
                    why.append("젊은 타깃과 톤 매칭 양호")
                if target_decade >= 40 and g in {"hist_war","drama","thriller","society"}:
                    why.append("성숙 타깃 선호 주제와 부합")
            # 요일
            if "토요일" in desc or "saturday" in desc:
                why.append(f"편성 효과: {day_reason['토요일']}")
            if "금요일" in desc or "friday" in desc:
                why.append(f"편성 효과: {day_reason['금요일']}")
            if "수요일" in desc or "wednesday" in desc:
                why.append(f"편성 효과: {day_reason['수요일']}")
            # 플랫폼
            for k, v in platform_reason.items():
                if k in desc:
                    why.append(f"플랫폼 효과: {v}")
            # 기타
            if "주연" in desc:
                why.append(f"캐스팅 효과: {etc_reason['주연']}")
            if "미혼" in desc:
                why.append(f"캐릭터 톤: {etc_reason['미혼']}")
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
    section[data-testid="stSidebar"] .sb-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 900;
    }
    section[data-testid="stSidebar"] .sb-brand .logo {
        font-size: 35px !important;  /* 이모티콘 크기 */
        line-height: 1;
    }
    section[data-testid="stSidebar"] .sb-brand .name {
        font-size: 26px !important;  /* 글자 크기 */
        line-height: 1.2;
    }
    </style>
    <div class="sb-brand">
        <span class="logo">💫</span>
        <span class="name">케미스코어</span>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sb-menu">', unsafe_allow_html=True)
    for slug, icon, label, _fn in NAV_ITEMS:
        active = (slug == current)
        st.markdown(f'<div class="sb-nav {"active" if active else ""}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{slug}", use_container_width=True):
            st.session_state["nav"] = slug
            _set_nav_query(slug)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    # Card: model config
    st.markdown('<div class="sb-card"><h4>모델 설정: \ntest_size=0.2, random_state=42</h4>', unsafe_allow_html=True)
    test_size = 0.2
    st.markdown('</div>', unsafe_allow_html=True)  # /sb-card
    st.markdown('</div>', unsafe_allow_html=True)  # /sb-menu
    st.markdown('<div class="sb-footer">© Chemiscore • <span class="ver">v0.1</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # /sb-wrap

# ================== 라우팅 ==================
PAGES = {slug: fn for slug, _, _, fn in NAV_ITEMS}
PAGES.get(st.session_state["nav"], page_overview)()
