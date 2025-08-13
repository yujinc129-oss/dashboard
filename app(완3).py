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
st.header("💫케미스코어")
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
# start airing 은 입력에서 제외
drop_cols = [c for c in ['배우명','드라마명','genres','day','network','score','start airing'] if c in df_mlb.columns]

# y는 확실히 수치형으로
if 'score' in df_mlb.columns:
    df_mlb['score'] = pd.to_numeric(df_mlb['score'], errors='coerce')

X_colab_base = df_mlb.drop(columns=drop_cols, errors='ignore')
y_all = df_mlb['score']

categorical_features = [c for c in ['role','gender','air_q','married','age_group'] if c in X_colab_base.columns]

# ★ OHE는 dense 로 → StandardScaler와 폴리 호환성 ↑
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
                ('ohe', ohe),
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
tabs = st.tabs(["🗂개요","📊기초통계","📈분포/교차","⚙️필터","🔍전체보기","🔧튜닝","🤖ML모델","🎯예측"])

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

    # 결혼 상태별 평균 점수 (간단 출력)
    st.subheader("주연 배우 결혼 상태별 평균 점수 비교")
    main_roles = raw_df[raw_df['role']=='주연'].copy()
    main_roles['결혼상태'] = main_roles['married'].apply(lambda x: '미혼' if x=='미혼' else '미혼 외')
    avg_scores_by_marriage = main_roles.groupby('결혼상태')['score'].mean()
    fig, ax = plt.subplots(figsize=(6, 6))  # 크기 줄임
    bars = ax.bar(avg_scores_by_marriage.index, avg_scores_by_marriage.values, 
                  color=['mediumseagreen', 'gray'])
    
    for b in bars:
        yv = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, yv+0.005, f'{yv:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('주연 배우 결혼 상태별 평균 점수 비교', fontsize=10)
    ax.set_ylabel('평균 점수', fontsize=9)
    ax.set_xlabel('결혼 상태', fontsize=9)
    ax.set_ylim(min(avg_scores_by_marriage.values)-0.05, max(avg_scores_by_marriage.values)+0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    st.pyplot(fig, use_container_width=False)
    m_single = avg_scores_by_marriage.get('미혼')
    m_else   = avg_scores_by_marriage.get('미혼 외')
    diff_txt = f"(차이 {m_single - m_else:+.3f}p)" if (m_single is not None and m_else is not None) else ""
    st.markdown(
        f"""
**요약**
- 미혼 평균: **{m_single:.3f}**, 미혼 외 평균: **{m_else:.3f}** {diff_txt}

**인사이트**
- 미혼 배우는 상대적으로 **청춘물·로맨틱 코미디·성장형 서사**에 자주 등장하며, 이런 장르는 시청자 선호도와 감정 이입률이 높아 **평점이 우호적으로 형성**되는 경향이 있습니다.
- 반면, 기혼 배우는 **가족극·사회극·정치물**에 출연하는 비중이 높고, 이들 장르는 주제의 무게감/몰입 장벽으로 **평가가 갈릴 가능성**이 큽니다.
- 시청자 인식에서 ‘**싱글**’ 이미지는 보다 자유롭고 다양한 캐릭터 소비로 이어지기 쉬워, **연애 서사 몰입도**나 **대중적 판타지 자극** 역할이 미혼 배우에게 더 자주 부여되는 편입니다.
"""
    )

    # --- 장르별 작품 수 및 평균 점수 (FIX) ---
    st.subheader("장르별 작품 수 및 평균 점수")
    
    dfg = raw_df.copy()
    dfg['genres'] = dfg['genres'].apply(clean_cell_colab)
    dfg = dfg.explode('genres').dropna(subset=['genres','score'])
    
    g_score = dfg.groupby('genres')['score'].mean().round(3)
    g_count = dfg['genres'].value_counts()
    
    # 인덱스->열 변환 후, 이름 컬럼을 '장르'로 통일
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
    st.pyplot(fig, use_container_width=False)


    # 요약 + 인사이트
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

**인사이트(생산량)**  
- **romance / drama / comedy**는 보편적 감정선과 일상 배경으로 **비용 대비 효율**이 높고, 폭넓은 시청층을 확보하기 좋아 **반복 제작**이 이루어집니다.  
- **action / sf / hist_war**는 CG·무술·대규모 세트·역사 고증 등으로 **제작비·제작기간 부담**이 커 상대적으로 **물량이 적은** 편입니다.

**인사이트(평점)**  
- **hist_war, thriller, sf**는 마니아층 중심으로 **완성도와 개성**이 평가 포인트가 되며 **평점이 높게 형성**되는 경향이 있습니다.  
- 반면 **romance, society**는 감정 서사의 반복으로 중후반 전개에 따라 **호불호**가 커져 평균 점수가 상대적으로 낮아질 수 있습니다.  
- **action / sf**는 시각적 임팩트가 커 OTT/온라인 시청 환경에서 **초기 만족도(첫인상 효과)**가 높게 나타나 평균 점수를 끌어올리기도 합니다.
"""
    )

    # 방영 요일별 작품 수 및 평균 점수 (day / score)
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
    plt.title('방영 요일별 작품 수 및 평균 점수 (월요일 → 일요일 순)'); plt.tight_layout(); st.pyplot(fig,use_container_width=False)

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

**인사이트(요약)**  
- **주중(월~목)**: 일상 편성 비중이 높고, 다양한 연령/취향을 겨냥한 **보편적 콘텐츠**가 많음. 제작 속도·양산성, **시청률 지향** 편성이 상대적으로 두드러짐.  
- **금요일**: 한 주 피로/외부 일정 영향으로 **실시간 시청 집중도 낮음** → 예능·뉴스·영화 대체 편성 빈도 높음.  
- **일요일**: 다음 날 출근 부담으로 **가벼운 콘텐츠 선호**, 전통적으로 예능이 프라임을 장악 → 드라마 **편성 수요 낮음**.  
- **토요일**: 시간적 여유 + 다음 날 휴식으로 **시청률·몰입·광고 효과 최대**. 고품질 작품을 집중 투입하는 **황금 슬롯**.
"""
    )


    # --- 방영년도별 작품 수 및 평균 점수 ---
    st.subheader("방영년도별 작품 수 및 평균 점수")
    dfe = raw_df.copy()
    dfe['start airing'] = pd.to_numeric(dfe['start airing'], errors='coerce')
    dfe['score'] = pd.to_numeric(dfe['score'], errors='coerce')
    dfe = dfe.dropna(subset=['start airing','score']).copy()
    dfe['start airing'] = dfe['start airing'].astype(int)

    mean_score_by_year = dfe.groupby('start airing')['score'].mean().round(3)
    count_by_year      = dfe['start airing'].value_counts()
    years = sorted(set(mean_score_by_year.index) | set(count_by_year.index))
    mean_s = mean_score_by_year.reindex(years)
    count_s = count_by_year.reindex(years, fill_value=0)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_bar = 'tab:gray'
    ax1.set_xlabel('방영년도')
    ax1.set_ylabel('작품 수', color=color_bar)
    bars = ax1.bar(years, count_s.values, alpha=0.3, color=color_bar, width=0.6)
    ax1.tick_params(axis='y', labelcolor=color_bar)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + max(0.5, h*0.02),
                 f'{int(h)}', ha='center', va='bottom', fontsize=9, color='black')

    ax2 = ax1.twinx()
    color_line = 'tab:blue'
    ax2.set_ylabel('평균 점수', color=color_line)
    ax2.plot(years, mean_s.values, marker='o', color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)
    if mean_s.notna().any():
        ax2.set_ylim(mean_s.min() - 0.05, mean_s.max() + 0.05)
    for x, y in zip(years, mean_s.values):
        if pd.notna(y):
            ax2.text(x, y + 0.01, f'{y:.3f}', color=color_line, fontsize=9, ha='center')

    plt.title('방영년도별 작품 수 및 평균 점수')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    # 상관/성장 요약
    valid_mask = mean_s.notna() & count_s.notna()
    yrs   = pd.Index(years)[valid_mask]
    cnt   = count_s[valid_mask].astype(float)
    meanv = mean_s[valid_mask].astype(float)
    r_all = float(np.corrcoef(cnt.values, meanv.values)[0, 1]) if len(yrs) >= 2 else np.nan
    mask_2017 = yrs >= 2017
    if mask_2017.any() and mask_2017.sum() >= 2:
        r_2017 = float(np.corrcoef(cnt[mask_2017].values, meanv[mask_2017].values)[0, 1])
    else:
        r_2017 = np.nan

    def calc_cagr(series: pd.Series, start_year: int, end_year: int):
        if start_year in series.index and end_year in series.index:
            start, end = float(series.loc[start_year]), float(series.loc[end_year])
            n = int(end_year - start_year)
            if start > 0 and n > 0:
                return (end / start) ** (1 / n) - 1
        return np.nan

    first_year, last_year = int(yrs.min()), int(yrs.max())
    cagr_all   = calc_cagr(count_s, first_year, last_year)
    cagr_2017  = calc_cagr(count_s, max(2017, first_year), last_year)

    st.markdown("**추가 통계 요약**")
    st.markdown(
        f"""
- 전체 기간 상관계수 r(작품 수 vs 평균 점수): **{r_all:.3f}**  
- 2017년 이후 상관계수 r: **{r_2017:.3f}**  
- 작품 수 CAGR(전체 {first_year}→{last_year}): **{(cagr_all*100):.2f}%/년**  
- 작품 수 CAGR(2017→{last_year}): **{(cagr_2017*100):.2f}%/년**
"""
    )
    if not np.isnan(r_2017):
        trend = "음(-)의" if r_2017 < 0 else "양(+)의"
        st.caption(f"메모: 2017년 이후 구간에서 작품 수와 평균 점수는 **{trend} 상관**을 보입니다.")

    # 연령대별 작품 수 & 성별 평균 점수 (age_group / gender / score)
    st.subheader("연령대별 작품 수 및 성별 평균 점수 (주연 배우 기준)")
    main_roles = raw_df.copy()
    main_roles = main_roles[main_roles['role'] == '주연']
    main_roles = main_roles.dropna(subset=['age_group','gender','score']).copy()
    main_roles['score'] = pd.to_numeric(main_roles['score'], errors='coerce')
    main_roles = main_roles.dropna(subset=['score'])

    def age_key(s: str):
        m = re.search(r'(\d+)', str(s))
        return int(m.group(1)) if m else 999

    age_order = sorted(main_roles['age_group'].astype(str).unique(), key=age_key)
    age_counts = (main_roles['age_group'].value_counts().reindex(age_order).fillna(0).astype(int))
    ga = (main_roles.groupby(['gender','age_group'])['score'].mean().round(3).reset_index())
    male_vals   = ga[ga['gender']=='남자'].set_index('age_group').reindex(age_order)['score']
    female_vals = ga[ga['gender']=='여자'].set_index('age_group').reindex(age_order)['score']

    fig, ax1 = plt.subplots(figsize=(6, 6))
    bars = ax1.bar(age_order, age_counts.values, color='lightgray', label='작품 수')
    ax1.set_ylabel('작품 수', fontsize=12)
    ax1.set_ylim(0, max(age_counts.max()*1.2, age_counts.max()+2))
    for rect in bars:
        h = rect.get_height()
        ax1.text(rect.get_x()+rect.get_width()/2, h + max(2, h*0.02),
                 f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2 = ax1.twinx()
    line1, = ax2.plot(age_order, male_vals.values, marker='o', linewidth=2, label='남자')
    line2, = ax2.plot(age_order, female_vals.values, marker='o', linewidth=2, label='여자')
    ax2.set_ylabel('평균 점수', fontsize=12)

    all_means = pd.concat([male_vals, female_vals]).dropna()
    if not all_means.empty:
        ymin = float(all_means.min()) - 0.05
        ymax = float(all_means.max()) + 0.05
        if ymin == ymax:
            ymin, ymax = ymin-0.05, ymax+0.05
        ax2.set_ylim(ymin, ymax)

    for x, y in zip(age_order, male_vals.values):
        if not np.isnan(y):
            ax2.text(x, y + 0.004, f'{y:.3f}', color=line1.get_color(),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    for x, y in zip(age_order, female_vals.values):
        if not np.isnan(y):
            ax2.text(x, y + 0.004, f'{y:.3f}', color=line2.get_color(),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('연령대별 작품 수 및 성별 평균 점수 (주연 배우 기준)', fontsize=14)
    ax1.set_xlabel('연령대', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend([line1, line2], ['남자','여자'], loc='upper left')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

# --- 4.5 실시간 필터 ---
with tabs[3]:
    st.header("실시간 필터")
    smin,smax = float(pd.to_numeric(raw_df['score'], errors='coerce').min()), float(pd.to_numeric(raw_df['score'], errors='coerce').max())
    sfilter = st.slider("최소 평점", smin,smax,smin)
    y_min = int(pd.to_numeric(raw_df['start airing'], errors='coerce').min())
    y_max = int(pd.to_numeric(raw_df['start airing'], errors='coerce').max())
    yfilter = st.slider("방영년도 범위", y_min, y_max, (y_min, y_max))
    filt = raw_df[(pd.to_numeric(raw_df['score'], errors='coerce')>=sfilter) & pd.to_numeric(raw_df['start airing'], errors='coerce').between(*yfilter)]
    st.dataframe(filt.head(20))

# --- 4.6 전체 미리보기 ---
with tabs[4]:
    st.header("원본 전체보기")
    st.dataframe(raw_df, use_container_width=True)

# --- 공통 준비 ---
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

# --- 4.7 GridSearch 튜닝 ---
with tabs[5]:
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
                if not t: continue
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
        safe_cols = [c for c in ["rank_test_score","mean_test_score","std_test_score","mean_train_score","std_train_score","params"] if c in cvres.columns]
        sorted_cvres = cvres.loc[:, safe_cols].sort_values("rank_test_score").reset_index(drop=True)
        st.dataframe(sorted_cvres, use_container_width=True)

    if model_name == "XGBRegressor" and not XGB_AVAILABLE:
        st.warning("xgboost가 설치되어 있지 않습니다. requirements.txt에 `xgboost`를 추가하고 재배포해 주세요.")

# --- 4.8 머신러닝 모델링 ---
with tabs[6]:
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

# --- 4.9 예측 실행 + 카운터팩추얼 ---
with tabs[7]:
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

        # 🎯 타깃 시청자 연령대 (여기로 이동)
        age_group_candidates = ["10대", "20대", "30대", "40대", "50대", "60대 이상"]
        data_age_groups = sorted(set(str(x) for x in raw_df.get("age_group", pd.Series([], dtype=object)).dropna().unique()))
        opts_age_group = data_age_groups if data_age_groups else age_group_candidates
        safe_index = 0 if not opts_age_group else min(1, len(opts_age_group)-1)

        target_age_group = st.selectbox(
            "🎯 타깃 시청자 연령대",
            options=opts_age_group if opts_age_group else ["(데이터 없음)"],
            index=safe_index,
            key="target_age_group_main"   # ✅ 고유 키 추가
        )
        # What-if에서 쓰려고 세션에 저장
        st.session_state["target_age_group"] = target_age_group
        st.session_state["actor_age"] = int(input_age)
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

        # 🔧 숫자열만 안전 보정 (카테고리 열은 그대로 문자열 유지)
        num_cols = X_colab_base.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            user_base[num_cols] = user_base[num_cols].apply(pd.to_numeric, errors="coerce")
            user_base[num_cols] = user_base[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        pred = model_full.predict(user_base)[0]
        st.success(f"💡 예상 평점: {pred:.2f}")
        # ✅ 카운터팩추얼에서 다시 쓰려고 세션에 보관
        st.session_state["cf_user_raw"] = user_raw.copy()
        st.session_state["cf_pred"] = float(pred)
        st.session_state["cf_model"] = model_full

                               # =========================
        # 🔎 What-if (독립 액션 Top N, 중복효과/합산/조합 금지)
        # =========================
        st.markdown("---")
        st.subheader("2) 케미스코어 평점 예측측")

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

            st.markdown("**📝 상위 변경안 솔루션 (독립 효과 기준)**")
            for _, r in df_top.iterrows():
                st.markdown(f"- **{r['변경안']}** · {_explain(r['변경안'])}")



        

        
    
      
