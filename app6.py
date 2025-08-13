# app.py
# ---- dependency guard (optional) ----
import importlib.util, streamlit as st
_missing = [m for m in ("numpy","scipy","sklearn","joblib","threadpoolctl","xgboost") if importlib.util.find_spec(m) is None]
if _missing:
    st.error(f"필수 라이브러리 미설치: {_missing}. requirements.txt / runtime.txt 버전을 고정해 재배포하세요.")
    st.stop()

import os
import ast
import random
import numpy as np
import pandas as pd
from pathlib import Path
import platform
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.express as px
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import clone
import re
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

# XGB가 설치돼 있으면 쓰도록 안전하게 추가
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ===== 페이지 설정 =====
st.set_page_config(page_title="K-드라마 분석/예측", page_icon="🎬", layout="wide")

# ===== 전역 시드 고정 =====
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

# ===== 한글 폰트 부트스트랩 =====
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
    wanted = ("nanum","malgun","applegothic","notosanscjk","sourcehan","gulim","dotum","batang","pretendard","gowun","spoqa")
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
    if exact:
        return counts[exact].idxmax()

    loose = [g for g in vocab if str(decade) in g]
    if loose:
        return counts[loose].idxmax()

    if decade >= 60:
        over = [g for g in vocab if ('60' in g) or ('이상' in g)]
        if over:
            return counts[over].idxmax()

    with_num = []
    for g in vocab:
        m = re.search(r'(\d+)', g)
        if m:
            with_num.append((g, int(m.group(1))))
    if with_num:
        nearest_num = min(with_num, key=lambda t: abs(t[1]-decade))[1]
        candidates = [g for g,n in with_num if n==nearest_num]
        return counts[candidates].idxmax()

    return counts.idxmax()

# ===== 전처리 유틸 =====
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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

def expand_feature_cols_for_training(base: pd.DataFrame, selected: list):
    cols = []
    for c in selected:
        if c in ('genres','day','network'):
            prefix = f"{c}_"
            cols += [k for k in base.columns if k.startswith(prefix)]
        else:
            cols.append(c)
    return cols

def build_X_from_selected(base: pd.DataFrame, selected: list) -> pd.DataFrame:
    X = pd.DataFrame(index=base.index)
    use_cols = expand_feature_cols_for_training(base, selected)
    if use_cols:
        X = pd.concat([X, base[use_cols]], axis=1)
    singles = [c for c in selected if c not in ('genres','day','network')]
    for c in singles:
        if c in base.columns and base[c].dtype == 'object':
            X = pd.concat([X, pd.get_dummies(base[c], prefix=c)], axis=1)
        elif c in base.columns:
            X[c] = base[c]
    return X

# ===== 데이터 로드 =====
@st.cache_data
def load_data():
    # drama_d.json 은 {컬럼명: {row_index: value}} 형태
    raw = pd.read_json('drama_d.json')
    if isinstance(raw, pd.Series):
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    else:
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    # ★ 노트북과 동일: 인덱스/순서 유지 (reset_index 제거)
    return raw

raw_df = load_data()

# ===== 멀티라벨 인코딩 결과 생성 (genres / day / network) =====
df_mlb = colab_multilabel_fit_transform(raw_df, cols=('genres','day','network'))

# ===== Colab 스타일 X/y, 전처리 정의 =====
drop_cols = [c for c in ['배우명','드라마명','genres','day','network','score','start airing'] if c in df_mlb.columns]
X_colab_base = df_mlb.drop(columns=drop_cols, errors='ignore')
y_all = df_mlb['score']

categorical_features = [c for c in ['role','gender','air_q','married','age_group'] if c in X_colab_base.columns]

# ★ OHE는 밀집(dense)로 -> StandardScaler 기본 사용 가능 (노트북과 동일)
try:
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='(missing)')),
                ('ohe', OneHotEncoder(
                    drop='first',
                    handle_unknown='ignore',
                    sparse_output=False   # 폴리/스케일러와 호환성↑
                )),
            ]),
            categorical_features
        )
    ],
    remainder='passthrough'
)
# ===== EDA용 리스트 =====
genre_list = [g for sub in raw_df.get('genres', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for g in sub]
broadcaster_list = [b for sub in raw_df.get('network', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for b in sub]
week_list = [w for sub in raw_df.get('day', pd.Series(dtype=object)).dropna().apply(clean_cell_colab) for w in sub]
unique_genres = sorted(set(genre_list))

# ===== 사이드바 =====
with st.sidebar:
    st.header("🤖 모델 설정")
    # ★ 노트북 재현: test_size 고정
    test_size = 0.2
    st.caption("노트북 재현 모드: test_size=0.2, random_state=42")

# ===== 탭 구성 =====
tabs = st.tabs(["🗂개요","📊기초통계","📈분포/교차","💬워드클라우드","⚙️필터","🔍전체보기","🔧튜닝","🤖ML모델","🎯예측"])

# --- 4.1 데이터 개요 ---
with tabs[0]:
    st.header("데이터 개요")
    c1,c2,c3 = st.columns(3)
    c1.metric("샘플 수", raw_df.shape[0])
    c2.metric("컬럼 수", raw_df.shape[1])
    c3.metric("고유 장르", len(unique_genres))
    st.subheader("결측치 비율")
    st.dataframe(raw_df.isnull().mean())
    st.subheader("원본 샘플")
    st.dataframe(raw_df.head(), use_container_width=True)

# --- 4.2 기초통계 ---
with tabs[1]:
    st.header("기초 통계: score")
    st.write(pd.to_numeric(raw_df['score'], errors='coerce').describe())
    fig,ax=plt.subplots(figsize=(6,3))
    ax.hist(pd.to_numeric(raw_df['score'], errors='coerce'), bins=20)
    ax.set_title("전체 평점 분포")
    st.pyplot(fig)

# --- 4.3 분포/교차분석 ---
with tabs[2]:
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
            txt = f"- **넷플릭스(OTT)의 급성장**: {first_year}년 이후 빠르게 증가, **{max_year}년 {max_val}편**으로 최고치."
            if 2020 in p.index:
                comps = ", ".join([f"{b} {int(p.loc[2020,b])}편" for b in ['KBS','MBC','SBS'] if b in p.columns])
                txt += f" 2020년에는 넷플릭스 {int(p.loc[2020,'NETFLIX'])}편, 지상파({comps})와 유사한 수준."
            insights.append(txt)

    import numpy as np
    down_ter = []
    for b in ['KBS','MBC','SBS']:
        if b in p.columns and len(years) >= 2:
            slope = np.polyfit(years, p[b].reindex(years, fill_value=0), 1)[0]
            if slope < 0:
                down_ter.append(b)
    if down_ter:
        insights.append(f"- **지상파의 지속적 감소**: {' / '.join(down_ter)} 등 전통 3사의 작품 수가 전반적으로 하락 추세.")

    if 'TVN' in p.columns:
        tvn = p['TVN']
        peak_year, peak_val = int(tvn.idxmax()), int(tvn.max())
        tail = []
        for y in [y for y in [2020, 2021, 2022] if y in tvn.index]:
            tail.append(f"{y}년 {int(tvn.loc[y])}편")
        insights.append(f"- **tvN의 성장과 정체**: 최고 {peak_year}년 {peak_val}편. 최근 수년({', '.join(tail)})은 정체/소폭 감소 경향.")

    if 2021 in p.index and 2022 in p.index:
        downs = [c for c in p.columns if p.loc[2022, c] < p.loc[2021, c]]
        if downs:
            insights.append(f"- **2022년 전년 대비 감소**: {', '.join(downs)} 등 여러 플랫폼이 2021년보다 줄어듦.")

    st.markdown("**인사이트**\n" + "\n".join(insights) +
                "\n\n*해석 메모: OTT-방송사 동시방영, 제작환경(예산/시청률), 코로나19 등 외부 요인이 영향을 준 것으로 해석 가능.*")

    # 장르 '개수'별 배우 평균 평점
    st.subheader("장르 개수별 평균 평점 (배우 단위, 1 ~ 2 / 3 ~ 4 / 5 ~ 6 / 7+)")
    actor_col = '배우명' if '배우명' in raw_df.columns else ('actor' if 'actor' in raw_df.columns else None)
    if actor_col is None:
        st.info("배우 식별 컬럼을 찾을 수 없어(배우명/actor) 이 섹션을 건너뜁니다.")
    else:
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
            title="장르 개수별 배우 평균 점수 분포 (1 ~ 2 / 3 ~ 4 / 5 ~ 6 / 7+)"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        stats = (df_actor.groupby('장르개수구간')['배우평균점수']
                 .agg(평균='mean', 중앙값='median', 표본수='count')
                 .reindex(['1~2개','3~4개','5~6개','7개 이상']).dropna(how='all').round(3))

        if not stats.empty and stats['표본수'].sum() > 0:
            best_mean_grp   = stats['평균'].idxmax()
            best_median_grp = stats['중앙값'].idxmax()
            vals = stats['평균'].dropna().values
            diffs = pd.Series(vals).diff().dropna()
            if (diffs >= 0).all():
                trend = "장르 수가 많을수록 평균 평점이 **높아지는 경향**"
            elif (diffs <= 0).all():
                trend = "장르 수가 많을수록 평균 평점이 **낮아지는 경향**"
            else:
                trend = "장르 수와 평균 평점 간 **일관된 단조 경향은 약함**"

            comp_txt = ""
            if {'1~2개','7개 이상'}.issubset(stats.index):
                diff_mean = stats.loc['1~2개','평균'] - stats.loc['7개 이상','평균']
                diff_med  = stats.loc['1~2개','중앙값'] - stats.loc['7개 이상','중앙값']
                sign = "높음" if diff_mean >= 0 else "낮음"
                comp_txt = f"- **1~2개 vs 7개 이상**: 평균 {abs(diff_mean):.3f}p {sign}, 중앙값 차이 {abs(diff_med):.3f}p\n"

            st.markdown("**요약 통계(배우 단위)**")
            try:
                st.markdown(stats.to_markdown())
            except Exception:
                st.dataframe(stats.reset_index(), use_container_width=True)

            st.markdown(
                f"""
**인사이트**
- 평균 기준 최고 그룹: **{best_mean_grp}** / 중앙값 기준 최고 그룹: **{best_median_grp}**  
- {trend}  
{comp_txt if comp_txt else ""}
- 장르 다양성↑ → 평점↑ (단조 증가)  
-> 다장르 경험 증가가 작품 선택 품질 향상에 기여했을 가능성.
"""
            )
        else:
            st.info("장르 개수 구간별 통계를 계산할 데이터가 부족합니다.")

    # 결혼 상태별 평균 점수
    st.subheader("주연 배우 결혼 상태별 평균 점수 비교")
    main_roles = raw_df[raw_df['role']=='주연'].copy()
    main_roles['결혼상태'] = main_roles['married'].apply(lambda x: '미혼' if x=='미혼' else '미혼 외')
    avg_scores_by_marriage = main_roles.groupby('결혼상태')['score'].mean()
    fig, ax = plt.subplots(figsize=(3,3))
    bars = ax.bar(avg_scores_by_marriage.index, avg_scores_by_marriage.values, color=['mediumseagreen','gray'])
    for b in bars:
        yv = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, yv+0.005, f'{yv:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('주연 배우 결혼 상태별 평균 점수 비교'); ax.set_ylabel('평균 점수'); ax.set_xlabel('결혼 상태')
    ax.set_ylim(min(avg_scores_by_marriage.values)-0.05, max(avg_scores_by_marriage.values)+0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    m_single = avg_scores_by_marriage.get('미혼')
    m_else   = avg_scores_by_marriage.get('미혼 외')
    diff_txt = f"(차이 {m_single - m_else:+.3f}p)" if (m_single is not None and m_else is not None) else ""
    st.markdown(
        f"""
**요약**
- 미혼 평균: **{m_single:.3f}**, 미혼 외 평균: **{m_else:.3f}** {diff_txt}
"""
    )

    # 장르별 작품 수 및 평균 점수
    st.subheader("장르별 작품 수 및 평균 점수")
    dfg = raw_df.copy()
    dfg['genres'] = dfg['genres'].apply(clean_cell_colab)
    dfg = dfg.explode('genres').dropna(subset=['genres','score'])
    g_score = dfg.groupby('genres')['score'].mean().round(3)
    g_count = dfg['genres'].value_counts()
    gdf = pd.DataFrame({'평균 점수': g_score, '작품 수': g_count}).reset_index()
    name_col = 'index' if 'index' in gdf.columns else ('genres' if 'genres' in gdf.columns else gdf.columns[0])
    gdf = gdf.rename(columns={name_col: '장르'})
    gdf = gdf.sort_values('작품 수', ascending=False).reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=(12,6))
    bars = ax1.bar(range(len(gdf)), gdf['작품 수'], color='lightgray')
    ax1.set_ylabel('작품 수')
    ax1.set_xticks(range(len(gdf)))
    ax1.set_xticklabels(gdf['장르'], rotation=45, ha='right')
    for i, r in enumerate(bars):
        h = r.get_height()
        ax1.text(i, h+max(2, h*0.01), f'{int(h)}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='#444')
    ax2 = ax1.twinx()
    ax2.plot(range(len(gdf)), gdf['평균 점수'], marker='o', linewidth=2, color='tab:blue')
    ax2.set_ylabel('평균 점수', color='tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')
    ax2.set_ylim(gdf['평균 점수'].min()-0.1, gdf['평균 점수'].max()+0.1)
    for i, v in enumerate(gdf['평균 점수']):
        ax2.text(i, v+0.01, f'{v:.3f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='tab:blue')
    plt.title('장르별 작품 수 및 평균 점수')
    ax1.set_xlabel('장르')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    top_cnt = gdf.nlargest(3, '작품 수')
    low_cnt = gdf.nsmallest(3, '작품 수')
    top_score = gdf.nlargest(4, '평균 점수')
    def fmt_counts(df): return ", ".join([f"{r['장르']}({int(r['작품 수']):,}편)" for _, r in df.iterrows()])
    def fmt_scores(df): return ", ".join([f"{r['장르']}({r['평균 점수']:.3f})" for _, r in df.iterrows()])
    st.markdown(
        f"""
**요약(데이터 근거)**  
- 작품 수 상위: **{fmt_counts(top_cnt)}**  
- 작품 수 하위: **{fmt_counts(low_cnt)}**  
- 평균 평점 상위: **{fmt_scores(top_score)}**
"""
    )

    # 방영 요일별
    st.subheader("방영 요일별 작품 수 및 평균 점수 (월→일)")
    dfe = raw_df.copy(); dfe['day'] = dfe['day'].apply(clean_cell_colab)
    dfe = dfe.explode('day').dropna(subset=['day','score']).copy()
    dfe['day'] = dfe['day'].astype(str).str.strip().str.lower()
    ordered = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    day_ko = {'monday':'월','tuesday':'화','wednesday':'수','thursday':'목','friday':'금','saturday':'토','sunday':'일'}
    mean_by = dfe.groupby('day')['score'].mean().reindex(ordered)
    cnt_by = dfe['day'].value_counts().reindex(ordered).fillna(0).astype(int)
    fig, ax1 = plt.subplots(figsize=(12,6))
    bars = ax1.bar(ordered, cnt_by.values, alpha=0.3, color='tab:gray')
    ax1.set_ylabel('작품 수', color='tab:gray'); ax1.tick_params(axis='y', labelcolor='tab:gray')
    for b in bars:
        h = b.get_height(); ax1.text(b.get_x()+b.get_width()/2, h+0.5, f'{int(h)}', ha='center', va='bottom', fontsize=9, color='black')
    ax2 = ax1.twinx(); ax2.plot(ordered, mean_by.values, marker='o', color='tab:blue')
    ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', labelcolor='tab:blue')
    if mean_by.notna().any(): ax2.set_ylim(mean_by.min()-0.05, mean_by.max()+0.05)
    for x, yv in zip(ordered, mean_by.values):
        if pd.notna(yv): ax2.text(x, yv+0.005, f'{yv:.3f}', color='tab:blue', fontsize=9, ha='center')
    ax1.set_xticks(ordered); ax1.set_xticklabels([day_ko[d] for d in ordered])
    plt.title('방영 요일별 작품 수 및 평균 점수 (월요일 → 일요일 순)'); plt.tight_layout(); st.pyplot(fig)

    weekday = ['monday','tuesday','wednesday','thursday']
    weekend = ['friday','saturday','sunday']
    wk_avg = mean_by.loc[weekday].mean(skipna=True)
    we_avg = mean_by.loc[weekend].mean(skipna=True)
    top_day_en  = mean_by.idxmax() if mean_by.notna().any() else None
    top_day_ko  = day_ko.get(top_day_en, "N/A") if top_day_en else "N/A"
    top_mean    = float(mean_by.max()) if mean_by.notna().any() else float("nan")
    wk_cnt = int(cnt_by.loc[weekday].sum()); we_cnt = int(cnt_by.loc[weekend].sum())

    st.markdown(
        f"""
**요약(데이터 근거)**  
- 주중 평균 평점(월~목): **{wk_avg:.3f}** · 주중 작품 수 합계: **{wk_cnt}편**  
- 주말 평균 평점(금~일): **{we_avg:.3f}** · 주말 작품 수 합계: **{we_cnt}편**  
- 평균 평점 최고 요일: **{top_day_ko} {top_mean:.3f}점**
"""
    )

# --- 4.4 워드클라우드 ---
from wordcloud import WordCloud
with tabs[3]:
    st.header("워드클라우드")
    from collections import Counter

    def top_pairs(words, n=5, keyfn=lambda x: str(x).strip().lower()):
        vals = [keyfn(w) for w in words if pd.notna(w) and str(w).strip() != ""]
        return Counter(vals).most_common(n)

    def pairs_to_str(pairs, label_map=None):
        it = []
        for k, v in pairs:
            kk = label_map.get(k, k) if label_map else k
            it.append(f"{kk}({v:,})")
        return ", ".join(it) if it else "N/A"

    font_path = st.session_state.get("kfont_path")

    if genre_list:
        wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path)\
                .generate(' '.join(genre_list))
        fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)

    if broadcaster_list:
        wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path)\
                .generate(' '.join(broadcaster_list))
        fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)

    if week_list:
        wk = [str(w).strip().lower() for w in week_list if pd.notna(w)]
        wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path)\
                .generate(' '.join(wk))
        fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)

# --- 4.5 실시간 필터 ---
with tabs[4]:
    st.header("실시간 필터")
    smin,smax = float(pd.to_numeric(raw_df['score'], errors='coerce').min()), float(pd.to_numeric(raw_df['score'], errors='coerce').max())
    sfilter = st.slider("최소 평점", smin,smax,smin)
    y_min = int(pd.to_numeric(raw_df['start airing'], errors='coerce').min())
    y_max = int(pd.to_numeric(raw_df['start airing'], errors='coerce').max())
    yfilter = st.slider("방영년도 범위", y_min, y_max, (y_min, y_max))
    filt = raw_df[(pd.to_numeric(raw_df['score'], errors='coerce')>=sfilter) & pd.to_numeric(raw_df['start airing'], errors='coerce').between(*yfilter)]
    st.dataframe(filt.head(20))

# --- 4.6 전체 미리보기 ---
with tabs[5]:
    st.header("원본 전체보기")
    st.dataframe(raw_df, use_container_width=True)

# --- 공통 준비 ---
# 파이프라인 빌더 (노트북과 동일한 구성/스텝명)
def make_pipeline(model_name, kind, estimator):
    if kind == "tree":
        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', estimator)
        ])

    if model_name == "SVR":
        return Pipeline([
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])

    if model_name == "KNN":
        return Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('knn', estimator)
        ])

    if model_name == "Linear Regression (Poly)":
        return Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('linreg', estimator)
        ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])

from sklearn.ensemble import RandomForestRegressor

# --- 4.7 GridSearch 튜닝 ---
with tabs[6]:
    st.header("GridSearchCV 튜닝")

    # split 보장 (노트북과 동일: test_size=0.2, random_state=42, shuffle=True)
    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X_colab_base, y_all, test_size=test_size, random_state=SEED, shuffle=True
        )
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(test_size)
    X_train, X_test, y_train, y_test = st.session_state["split_colab"]

    scoring = st.selectbox("스코어링", ["neg_root_mean_squared_error", "r2"], index=0)
    cv = st.number_input("CV 폴드 수", 3, 10, 5, 1)
    cv_shuffle = st.checkbox("CV 셔플(shuffle)", value=False)

    def render_param_selector(label, options):
        display_options, to_py = [], {}
        for v in options:
            if v is None:
                s = "(None)"; to_py[s] = None
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
                if not t:
                    continue
                if t.lower() == "none":
                    val = None
                else:
                    try:
                        val = int(t)
                    except:
                        try:
                            val = float(t)
                        except:
                            val = t
                chosen.append(val)

        uniq = []
        for v in chosen:
            if v not in uniq:
                uniq.append(v)
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

    # 그리드 (노트북과 동일 키)
    default_param_grids = {
        "KNN": {"poly__degree":[1,2,3], "knn__n_neighbors":[3,4,5,6,7,8,9,10]},
        "Linear Regression (Poly)": {"poly__degree":[1,2,3]},
        "Ridge": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000]},
        "Lasso": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000]},
        "ElasticNet": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000], "model__l1_ratio":[0.1,0.5,0.9]},
        "SGDRegressor": {"poly__degree":[1,2,3], "model__learning_rate":["constant","invscaling","adaptive"]},
        # SVR은 Poly 제거
        "SVR": {"model__kernel":["poly","rbf","sigmoid"], "model__degree":[1,2,3]},
        "Decision Tree": {"model__max_depth":[10,15,20,25,30], "model__min_samples_split":[5,6,7,8,9,10], "model__min_samples_leaf":[2,3,4,5], "model__max_leaf_nodes":[None,10,20,30]},
        "Random Forest": {"model__n_estimators":[100,200,300], "model__min_samples_split":[5,6,7,8,9,10], "model__max_depth":[5,10,15,20,25,30]},
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
    user_grid = {}
    for param_key, default_vals in base_grid.items():
        user_vals = render_param_selector(param_key, default_vals)
        user_grid[param_key] = user_vals if len(user_vals) > 0 else default_vals

    with st.expander("선택한 파라미터 확인"):
        st.write(user_grid)

    if st.button("GridSearch 실행"):
        # ★ 노트북과 동일: 기본은 shuffle=False (정수 전달)
        if cv_shuffle:
            cv_obj = KFold(n_splits=int(cv), shuffle=True, random_state=SEED)
        else:
            cv_obj = int(cv)  # 예: 5

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=user_grid,
            cv=cv_obj,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )
        with st.spinner("GridSearchCV 실행 중..."):
            gs.fit(X_train, y_train)

        st.subheader("베스트 결과")
        st.write("Best Params:", gs.best_params_)
        if scoring == "neg_root_mean_squared_error":
            st.write("Best CV RMSE (음수):", gs.best_score_)  # 부호 유지
        else:
            st.write(f"Best CV {scoring}:", gs.best_score_)

        y_pred_tr = gs.predict(X_train)
        y_pred_te = gs.predict(X_test)
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
        cols = ["rank_test_score","mean_test_score","std_test_score","mean_train_score","std_train_score","params"]
        safe_cols = [c for c in ["rank_test_score","mean_test_score","std_test_score","mean_train_score","std_train_score","params"] if c in cvres.columns]
        sorted_cvres = cvres.loc[:, safe_cols].sort_values("rank_test_score").reset_index(drop=True)
        st.dataframe(sorted_cvres)

    if model_name == "XGBRegressor" and not XGB_AVAILABLE:
        st.warning("xgboost가 설치되어 있지 않습니다. requirements.txt에 `xgboost`를 추가하고 재배포해 주세요.")

# --- 4.8 머신러닝 모델링 ---
with tabs[7]:
    st.header("머신러닝 모델링")

    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X_colab_base, y_all, test_size=test_size, random_state=SEED, shuffle=True
        )
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(test_size)
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

    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    st.metric("Train R²", f"{r2_score(y_train, y_pred_tr):.3f}")
    st.metric("Test  R²", f"{r2_score(y_test,  y_pred_te):.3f}")
    st.metric("Train RMSE", f"{rmse(y_train, y_pred_tr):.3f}")
    st.metric("Test  RMSE", f"{rmse(y_test,  y_pred_te):.3f}")

    if "best_params" in st.session_state:
        with st.expander("베스트 하이퍼파라미터 보기"):
            st.json(st.session_state["best_params"])

# --- 4.9 예측 실행 ---
with tabs[8]:
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
        if n_genre == 0:
            genre_bucket = "장르없음"
        elif n_genre <= 2:
            genre_bucket = "1~2개"
        elif n_genre <= 4:
            genre_bucket = "3~4개"
        elif n_genre <= 6:
            genre_bucket = "5~6개"
        else:
            genre_bucket = "7개 이상"

        st.caption(f"자동 연령대: **{derived_age_group}**  |  장르 개수: **{genre_bucket}**")

    with col_right:
        st.markdown("**② 편성 특성**")
        input_quarter = st.selectbox("방영분기", quarter_opts) if quarter_opts else st.text_input("방영분기 입력", "")
        input_week    = st.multiselect("방영요일 (멀티 선택)", week_opts, default=week_opts[:1] if week_opts else [])
        input_plat    = st.multiselect("플랫폼 (멀티 선택)", plat_opts, default=plat_opts[:1] if plat_opts else [])

    predict_btn = st.button("예측 실행")

    if predict_btn:
        if "best_estimator" in st.session_state:
            model_full = clone(st.session_state["best_estimator"])
            st.caption(f"예측 모델: GridSearch 베스트 재학습 사용 ({st.session_state.get('best_name')})")
        else:
            model_full = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=SEED))
            ])
            st.caption("예측 모델: 기본 RandomForest (미튜닝)")

        model_full.fit(X_colab_base, y_all)

        user_raw = pd.DataFrame([{
            'age'      : input_age,
            'gender'   : input_gender,
            'role'     : input_role,
            'married'  : input_married,
            'air_q'    : input_quarter,
            'age_group': derived_age_group,
            'genres'   : input_genre,
            'day'      : input_week,
            'network'  : input_plat,
            '장르구분'    : genre_bucket,
        }])

        user_mlb = colab_multilabel_transform(user_raw, cols=('genres','day','network'))

        user_base = pd.concat([X_colab_base.iloc[:0].copy(), user_mlb], ignore_index=True)
        user_base = user_base.drop(columns=[c for c in drop_cols if c in user_base.columns], errors='ignore')
        for c in X_colab_base.columns:
            if c not in user_base.columns:
                user_base[c] = 0
        user_base = user_base[X_colab_base.columns].tail(1)
        user_base = user_base.replace([np.inf, -np.inf], np.nan).fillna(0)

        pred = model_full.predict(user_base)[0]

        pred = model_full.predict(user_base)[0]
        st.success(f"💡 예상 평점: {pred:.2f}")
                # =========================
        # 🔎 Counterfactual What-if
        # =========================
        st.markdown("---")
        st.subheader("🧪 What-if(카운터팩추얼) 탐색")

        # ── 공통 유틸: user_raw → user_base(feature vector)
        def _build_user_base(df_raw: pd.DataFrame) -> pd.DataFrame:
            _user_mlb = colab_multilabel_transform(df_raw, cols=('genres','day','network'))
            _base = pd.concat([X_colab_base.iloc[:0].copy(), _user_mlb], ignore_index=True)
            _base = _base.drop(columns=[c for c in drop_cols if c in _base.columns], errors='ignore')
            for c in X_colab_base.columns:
                if c not in _base.columns:
                    _base[c] = 0
            _base = _base[X_colab_base.columns].tail(1)
            # 폴리노미얼/스케일러 안정화: 숫자화 + 결측 0
            _base = _base.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            return _base

        def _predict_from_raw(df_raw: pd.DataFrame) -> float:
            vb = _build_user_base(df_raw)
            return float(model_full.predict(vb)[0])

        # 현재 입력 저장(What-if의 출발점)
        st.session_state["cf_user_raw"] = user_raw.copy()
        current_pred = float(pred)

        # ── 변경 가능한 액션(필요시 자유롭게 추가/수정)
        # 후보 라벨은 실제 학습에 존재하는 값만 사용(mlb classes 참조)
        def _classes_safe(key: str):
            return [s for s in (st.session_state.get(f"mlb_classes_{key}", []) or [])]

        genre_classes   = [g for g in _classes_safe("genres") if isinstance(g, str)]
        day_classes     = [d for d in _classes_safe("day") if isinstance(d, str)]
        network_classes = [n for n in _classes_safe("network") if isinstance(n, str)]

        # 우선순위 장르(데이터에 있는 것만 남김)
        priority_genres = [g for g in ["thriller","hist_war","sf","action","romance","drama","comedy"] if g in genre_classes]
        # 요일 후보
        saturday_only   = ["saturday"] if "saturday" in day_classes else (day_classes[:1] if day_classes else [])
        friday_only     = ["friday"]   if "friday"   in day_classes else []
        wednesday_only  = ["wednesday"]if "wednesday"in day_classes else []
        # 플랫폼 후보
        netflix         = "NETFLIX" if "NETFLIX" in network_classes else (network_classes[0] if network_classes else None)
        tvn             = "TVN" if "TVN" in network_classes else None

        # 액션 정의: (id, 설명, 적용함수)
        # 적용함수는 user_raw(DataFrame) 하나를 받아서 변경된 DataFrame을 반환
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

        # 단일 액션 후보들
        actions = []
        for g in priority_genres:
            actions.append((f"add_genre_{g}", f"장르 추가: {g}", _add_genre(g)))
        if saturday_only:
            actions.append(("set_sat_only", "편성 요일: 토요일 단일", _set_days(saturday_only)))
        if friday_only:
            actions.append(("set_fri_only", "편성 요일: 금요일 단일", _set_days(friday_only)))
        if wednesday_only:
            actions.append(("set_wed_only", "편성 요일: 수요일 단일", _set_days(wednesday_only)))
        if netflix:
            actions.append(("ensure_netflix", "플랫폼 포함: NETFLIX", _ensure_platform(netflix)))
        if tvn:
            actions.append(("ensure_tvn", "플랫폼 포함: TVN", _ensure_platform(tvn)))

        # role / married가 데이터에 있다면 액션 추가
        if "role" in user_raw.columns:
            if str(user_raw.at[0,"role"]) != "주연":
                actions.append(("set_lead", "역할: 주연으로 변경", _set_role("주연")))
        if "married" in user_raw.columns and str(user_raw.at[0,"married"]) != "미혼":
            actions.append(("set_single", "결혼여부: 미혼으로 변경", _set_married("미혼")))

        # ── 단일 액션 평가
        rows = []
        for aid, desc, fn in actions:
            cand = fn(user_raw)
            p = _predict_from_raw(cand)
            rows.append({"종류":"단일","아이디":aid,"설명":desc,"예측":p,"리프트":p - current_pred,"편집수":1,"적용":fn})

        # ── 2개 조합(연산량 제한: 단일 리프트 상위 6개만 조합)
        rows_sorted_single = sorted(rows, key=lambda d: d["리프트"], reverse=True)[:6]
        from itertools import combinations
        for (a1, a2) in combinations(rows_sorted_single, 2):
            fn_combo = lambda df, f1=a1["적용"], f2=a2["적용"]: f2(f1(df))
            p = _predict_from_raw(fn_combo(user_raw))
            rows.append({
                "종류":"조합2","아이디":f'{a1["아이디"]}+{a2["아이디"]}',
                "설명":f'{a1["설명"]} + {a2["설명"]}',
                "예측":p,"리프트":p - current_pred,"편집수":2,"적용":fn_combo
            })

        # 표 출력
        import math
        import pandas as _pd
        df_cf = _pd.DataFrame(rows)
        if not df_cf.empty:
            df_view = (df_cf
                .sort_values(["예측","리프트","편집수"], ascending=[False, False, True])
                [["종류","설명","예측","리프트","편집수"]]
                .head(12)
                .reset_index(drop=True))
            st.dataframe(df_view.style.format({"예측":"{:.3f}","리프트":"{:+.3f}"}), use_container_width=True)

        # 목표 점수 도달 추천
        target = st.number_input("목표 점수", min_value=0.0, max_value=10.0, value=min(8.2, max(7.0, round(current_pred+0.2,2))), step=0.05)
        rec = None
        if not df_cf.empty:
            above = df_cf[df_cf["예측"] >= float(target)]
            if not above.empty:
                # 편집 수 최소 → 리프트 최대 순
                rec = (above.sort_values(["편집수","예측"], ascending=[True, False]).iloc[0]).to_dict()
            else:
                rec = (df_cf.sort_values(["예측"], ascending=False).iloc[0]).to_dict()

        if rec is not None:
            st.success(f"추천 변경안 ▶ {rec['설명']}  |  예상 {rec['예측']:.3f}점 ({rec['리프트']:+.3f})")

            # 적용 버튼: 현재 입력에 반영해서 즉시 재예측
            if st.button("이 변경안 적용해서 다시 예측"):
                applied_raw = rec["적용"](user_raw)
                # 재예측 및 화면 갱신
                new_pred = _predict_from_raw(applied_raw)
                st.session_state["cf_user_raw"] = applied_raw
                st.info(f"재예측 결과: {new_pred:.2f}  (기존 {current_pred:.2f} → {new_pred - current_pred:+.2f})")

