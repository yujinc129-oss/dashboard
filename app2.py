# app.py
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

#XGB가 설치돼 있으면 쓰도록 안전하게 추가
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def rmse(y_true, y_pred):
    # squared=False 미지원 환경에서도 동작
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
    # 데이터셋에 있는 연령대 라벨들
    s = raw_df.get('연령대')
    if s is None or s.dropna().empty:
        # 폴백: 기본 구간
        if age < 20: return "10대"
        if age < 30: return "20대"
        if age < 40: return "30대"
        if age < 50: return "40대"
        return "50대 이상"

    series = s.dropna().astype(str)
    vocab = series.unique().tolist()
    counts = series.value_counts()

    decade = (int(age)//10)*10  # 27→20, 41→40 ...

    # 1) '20대'처럼 정확한 패턴 우선
    exact = [g for g in vocab if re.search(rf"{decade}\s*대", g)]
    if exact:
        return counts[exact].idxmax()  # 가장 흔한 라벨

    # 2) 숫자만 포함돼도 허용 (예: '20대 후반')
    loose = [g for g in vocab if str(decade) in g]
    if loose:
        return counts[loose].idxmax()

    # 3) 50대 이상 폴백
    if decade >= 50:
        over = [g for g in vocab if ('50' in g) or ('이상' in g)]
        if over:
            return counts[over].idxmax()

    # 4) 가장 가까운 십대 라벨로 매칭
    with_num = []
    for g in vocab:
        m = re.search(r'(\d+)', g)
        if m:
            with_num.append((g, int(m.group(1))))
    if with_num:
        nearest_num = min(with_num, key=lambda t: abs(t[1]-decade))[1]
        candidates = [g for g,n in with_num if n==nearest_num]
        return counts[candidates].idxmax()

    # 5) 최빈값
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

def colab_multilabel_fit_transform(df: pd.DataFrame, cols=('장르','방영요일','플랫폼')) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)
        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(out[col])
        new_cols = [f"{col}_{c.strip().upper()}" for c in mlb.classes_]
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors='ignore')
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)
         # 세션에 '학습된' mlb와 클래스 둘 다 저장
        st.session_state[f"mlb_{col}"] = mlb
        st.session_state[f"mlb_classes_{col}"] = mlb.classes_.tolist()
    return out

def colab_multilabel_transform(df: pd.DataFrame, cols=('장르','방영요일','플랫폼')) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)

        # 1) 세션에 '학습된' mlb가 있으면 그대로 사용
        mlb = st.session_state.get(f"mlb_{col}", None)

        # 2) 없으면 classes로부터 복구 (classes_ 직접 세팅)
        if mlb is None:
            classes = st.session_state.get(f"mlb_classes_{col}", [])
            mlb = MultiLabelBinarizer()
            if classes:
                mlb.classes_ = np.array(classes)  # ← transform이 바로 가능
            else:
                # 3) 그래도 없으면 df_mlb 컬럼에서 유추 (prefix 제거)
                try:
                    prefix = f"{col}_"
                    labels = [c[len(prefix):] for c in df_mlb.columns if c.startswith(prefix)]
                    if labels:
                        mlb.classes_ = np.array(labels)
                    else:
                        # 마지막 폴백: 현재 입력으로 fit (이 경우 훈련 스키마와 어긋날 수 있음)
                        mlb.fit(out[col])
                except Exception:
                    mlb.fit(out[col])

        arr = mlb.transform(out[col])  # 이제 NotFittedError 안 남
        new_cols = [f"{col}_{c}" for c in mlb.classes_]  # classes_ 이미 UPPER 처리됨
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors='ignore')
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)

    return out

# (선택) 사용자 선택 기반 X 생성 유틸 — EDA/예측 탭에서 사용
def expand_feature_cols_for_training(base: pd.DataFrame, selected: list):
    cols = []
    for c in selected:
        if c in ('장르','방영요일','플랫폼'):
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
    singles = [c for c in selected if c not in ('장르','방영요일','플랫폼')]
    for c in singles:
        if c in base.columns and base[c].dtype == 'object':
            X = pd.concat([X, pd.get_dummies(base[c], prefix=c)], axis=1)
        elif c in base.columns:
            X[c] = base[c]
    return X

# ===== 데이터 로드 =====
@st.cache_data
def load_data():
    raw = pd.read_json('drama_data.json')
    return pd.DataFrame({c: pd.Series(v) for c,v in raw.items()})

raw_df = load_data()

# ===== Colab과 동일 멀티라벨 인코딩 결과 생성 =====
df_mlb = colab_multilabel_fit_transform(raw_df, cols=('장르','방영요일','플랫폼'))
df_mlb['점수'] = pd.to_numeric(df_mlb['점수'], errors='coerce')

# ===== Colab 스타일 X/y, 전처리 정의 =====
drop_cols = [c for c in ['배우명','드라마명','장르','방영요일','플랫폼','점수','방영년도'] if c in df_mlb.columns]
X_colab_base = df_mlb.drop(columns=drop_cols)
y_all = df_mlb['점수']
categorical_features = [c for c in ['역할','성별','방영분기','결혼여부','연령대'] if c in X_colab_base.columns]
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# ===== EDA용 리스트 =====
genre_list = [g for sub in raw_df['장르'].dropna().apply(clean_cell_colab) for g in sub]
broadcaster_list = [b for sub in raw_df['플랫폼'].dropna().apply(clean_cell_colab) for b in sub]
week_list = [w for sub in raw_df['방영요일'].dropna().apply(clean_cell_colab) for w in sub]
unique_genres = sorted(set(genre_list))

# ===== 사이드바 =====
with st.sidebar:
    st.header("🤖 모델 설정")
    # 노트북과 동일하게 기본값 0.3으로 설정
    test_size = st.slider('테스트셋 비율', 0.1, 0.5, 0.3, 0.05)
    feature_cols = st.multiselect(
        '특성 선택(예측 탭용)',
        ['나이','방영년도','성별','장르','배우명','플랫폼','결혼여부'],
        default=['나이','방영년도','장르']
    )

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
    st.header("기초 통계: 점수")
    st.write(raw_df['점수'].astype(float).describe())
    fig,ax=plt.subplots(figsize=(6,3))
    ax.hist(raw_df['점수'].astype(float), bins=20)
    ax.set_title("전체 평점 분포")
    st.pyplot(fig)

# --- 4.3 분포/교차분석 ---
with tabs[2]:
    st.header("분포 및 교차분석")
    st.subheader("연도별 주요 플랫폼 작품 수")
    ct = (
        pd.DataFrame({'방영년도': raw_df['방영년도'], '플랫폼': raw_df['플랫폼'].apply(clean_cell_colab)})
        .explode('플랫폼').groupby(['방영년도','플랫폼']).size().reset_index(name='count')
    )
    ct['플랫폼_up'] = ct['플랫폼'].str.upper()
    focus = ['KBS','MBC','TVN','NETFLIX','SBS']
    fig3 = px.line(ct[ct['플랫폼_up'].isin(focus)], x='방영년도', y='count', color='플랫폼',
                   log_y=True, title="연도별 주요 플랫폼 작품 수")
    st.plotly_chart(fig3, use_container_width=True)

    # ---------- 자동 인사이트 ----------
    import numpy as np
    
    # 연-플랫폼 피벗
    p = (ct.pivot_table(index='방영년도', columns='플랫폼_up', values='count', aggfunc='sum')
           .fillna(0).astype(int))
    years = sorted(p.index)
    
    insights = []
    
    # 1) Netflix 급성장
    if 'NETFLIX' in p.columns:
        s = p['NETFLIX']
        nz = s[s > 0]
        if not nz.empty:
            first_year = int(nz.index.min())
            max_year, max_val = int(s.idxmax()), int(s.max())
            txt = f"- **넷플릭스(OTT)의 급성장**: {first_year}년 이후 빠르게 증가, **{max_year}년 {max_val}편**으로 최고치."
            # 2020년 비교
            if 2020 in p.index:
                comps = ", ".join([f"{b} {int(p.loc[2020,b])}편"
                                   for b in ['KBS','MBC','SBS'] if b in p.columns])
                txt += f" 2020년에는 넷플릭스 {int(p.loc[2020,'NETFLIX'])}편, 지상파({comps})와 유사한 수준."
            insights.append(txt)
    
    # 2) 지상파 감소 추세
    down_ter = []
    for b in ['KBS','MBC','SBS']:
        if b in p.columns and len(years) >= 2:
            slope = np.polyfit(years, p[b].reindex(years, fill_value=0), 1)[0]
            if slope < 0:
                down_ter.append(b)
    if down_ter:
        insights.append(f"- **지상파의 지속적 감소**: {' / '.join(down_ter)} 등 전통 3사의 작품 수가 전반적으로 하락 추세.")
    
    # 3) tvN 성장과 정체
    if 'TVN' in p.columns:
        tvn = p['TVN']
        peak_year, peak_val = int(tvn.idxmax()), int(tvn.max())
        tail = []
        for y in [y for y in [2020, 2021, 2022] if y in tvn.index]:
            tail.append(f"{y}년 {int(tvn.loc[y])}편")
        insights.append(f"- **tvN의 성장과 정체**: 최고 {peak_year}년 {peak_val}편. 최근 수년({', '.join(tail)})은 정체/소폭 감소 경향.")
    
    # 4) 2022년 전년 대비 감소
    if 2021 in p.index and 2022 in p.index:
        downs = [c for c in p.columns if p.loc[2022, c] < p.loc[2021, c]]
        if downs:
            insights.append(f"- **2022년 전년 대비 감소**: {', '.join(downs)} 등 여러 플랫폼이 2021년보다 줄어듦.")
    
    # 출력
    st.markdown("**인사이트**\n" + "\n".join(insights) +
                "\n\n*해석 메모: OTT-방송사 동시방영, 제작환경(예산/시청률), 코로나19 등 외부 요인이 영향을 준 것으로 해석 가능.*")

    # --- 장르 '개수'별 배우 평균 평점 (1~2 / 3~4 / 5~6 / 7+) ---
    st.subheader("장르 개수별 평균 평점 (배우 단위, 1 ~ 2 / 3 ~ 4 / 5 ~ 6 / 7+)")
    
    # 1) 배우별 고유 장르 개수
    gdf = (
        pd.DataFrame({
            '배우명': raw_df['배우명'],
            '장르' :  raw_df['장르'].apply(clean_cell_colab)
        })
        .explode('장르')
        .dropna(subset=['배우명','장르'])
    )
    genre_cnt = gdf.groupby('배우명')['장르'].nunique().rename('장르개수')
    
    # 2) 배우별 평균 점수
    actor_mean = (raw_df.groupby('배우명', as_index=False)['점수']
                  .mean()
                  .rename(columns={'점수':'배우평균점수'}))
    
    # 3) 병합 + 구간화(1~2, 3~4, 5~6, 7+)
    df_actor = actor_mean.merge(genre_cnt.reset_index(), on='배우명', how='left')
    df_actor['장르개수'] = df_actor['장르개수'].fillna(0).astype(int)
    df_actor = df_actor[df_actor['장르개수'] > 0].copy()  # 장르정보 없는 배우 제외
    
    def bucket(n: int) -> str:
        if n <= 2:  return '1~2개'
        if n <= 4:  return '3~4개'
        if n <= 6:  return '5~6개'
        return '7개 이상'
    
    df_actor['장르개수구간'] = df_actor['장르개수'].apply(bucket)
    order_bins = ['1~2개','3~4개','5~6개','7개 이상']
    df_actor['장르개수구간'] = pd.Categorical(df_actor['장르개수구간'],
                                          categories=order_bins, ordered=True)
    
    # 4) 박스플롯
    fig_box = px.box(
        df_actor, x='장르개수구간', y='배우평균점수',
        category_orders={'장르개수구간': order_bins},
        title="장르 개수별 배우 평균 점수 분포 (1 ~ 2 / 3 ~ 4 / 5 ~ 6 / 7+)"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # 5) 그래프 아래 인사이트 자동 생성
    stats = (df_actor.groupby('장르개수구간')['배우평균점수']
             .agg(평균='mean', 중앙값='median', 표본수='count')
             .reindex(order_bins)
             .dropna(how='all')
             .round(3))
    
    if not stats.empty and stats['표본수'].sum() > 0:
        # 최고 그룹
        best_mean_grp   = stats['평균'].idxmax()
        best_median_grp = stats['중앙값'].idxmax()
    
        # 단조 경향(평균 기준)
        vals = stats['평균'].dropna().values
        diffs = pd.Series(vals).diff().dropna()
        if (diffs >= 0).all():
            trend = "장르 수가 많을수록 평균 평점이 **높아지는 경향**"
        elif (diffs <= 0).all():
            trend = "장르 수가 많을수록 평균 평점이 **낮아지는 경향**"
        else:
            trend = "장르 수와 평균 평점 간 **일관된 단조 경향은 약함**"
    
        # 1~2개 vs 7개 이상 비교(둘 다 있을 때만)
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
평균이 7.774 → 7.802 → 7.861 → 7.911로 계단식 상승합니다.
중앙값도 7.700 → 7.715 → 7.810 → 7.901로 동일하게 증가.

-> 다장르 경험이 많을수록 연기 적응력/인지도/캐스팅 파워가 높아 작품 선택 품질이 좋아졌을 가능성.
            """
        )
    else:
        st.info("장르 개수 구간별 통계를 계산할 데이터가 부족합니다.")


    st.subheader("주연 배우 결혼 상태별 평균 점수 비교")
    main_roles = raw_df[raw_df['역할']=='주연'].copy()
    main_roles['결혼상태'] = main_roles['결혼여부'].apply(lambda x: '미혼' if x=='미혼' else '미혼 외')
    avg_scores_by_marriage = main_roles.groupby('결혼상태')['점수'].mean()
    fig, ax = plt.subplots(figsize=(3,3))
    bars = ax.bar(avg_scores_by_marriage.index, avg_scores_by_marriage.values, color=['mediumseagreen','gray'])
    for b in bars:
        yv = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, yv+0.005, f'{yv:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('주연 배우 결혼 상태별 평균 점수 비교'); ax.set_ylabel('평균 점수'); ax.set_xlabel('결혼 상태')
    ax.set_ylim(min(avg_scores_by_marriage.values)-0.05, max(avg_scores_by_marriage.values)+0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    # --- 그래프 아래 인사이트 ---
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

    st.subheader("장르별 작품 수 및 평균 점수")
    dfg = raw_df.copy(); dfg['장르'] = dfg['장르'].apply(clean_cell_colab)
    dfg = dfg.explode('장르').dropna(subset=['장르','점수'])
    g_score = dfg.groupby('장르')['점수'].mean().round(3)
    g_count = dfg['장르'].value_counts()
    gdf = (pd.DataFrame({'평균 점수': g_score, '작품 수': g_count}).reset_index().rename(columns={'index':'장르'}))
    gdf = gdf.sort_values('작품 수', ascending=False).reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=(12,6))
    bars = ax1.bar(range(len(gdf)), gdf['작품 수'], color='lightgray')
    ax1.set_ylabel('작품 수'); ax1.set_xticks(range(len(gdf))); ax1.set_xticklabels(gdf['장르'], rotation=45, ha='right')
    for i, r in enumerate(bars):
        h = r.get_height(); ax1.text(i, h+max(2, h*0.01), f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444')
    ax2 = ax1.twinx(); ax2.plot(range(len(gdf)), gdf['평균 점수'], marker='o', linewidth=2, color='tab:blue')
    ax2.set_ylabel('평균 점수', color='tab:blue'); ax2.tick_params(axis='y', colors='tab:blue')
    ax2.set_ylim(gdf['평균 점수'].min()-0.1, gdf['평균 점수'].max()+0.1)
    for i, v in enumerate(gdf['평균 점수']):
        ax2.text(i, v+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='tab:blue')
    plt.title('장르별 작품 수 및 평균 점수'); ax1.set_xlabel('장르'); ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(); st.pyplot(fig)

    # ====== 요약(데이터) + 인사이트 출력 ======
    # 상/하위 장르 정리
    top_cnt = gdf.nlargest(3, '작품 수')
    low_cnt = gdf.nsmallest(3, '작품 수')
    top_score = gdf.nlargest(4, '평균 점수')  # 상위 4개 정도
    
    def fmt_counts(df):
        return ", ".join([f"{r['장르']}({int(r['작품 수']):,}편)" for _, r in df.iterrows()])
    
    def fmt_scores(df):
        return ", ".join([f"{r['장르']}({r['평균 점수']:.3f})" for _, r in df.iterrows()])
    
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
    - 반면 **romance, society**는 삼각관계·출생의 비밀 등 **감정 서사의 반복**으로 중후반 전개에 따라 **호불호**가 커져 평균 점수가 상대적으로 낮아질 수 있습니다.  
    - **action / sf**는 시각적 임팩트가 커 OTT/온라인 시청 환경에서 **초기 만족도(첫인상 효과)**가 높게 나타나 평균 점수를 끌어올리기도 합니다.
    """
    )


    st.subheader("방영 요일별 작품 수 및 평균 점수 (월→일)")
    dfe = raw_df.copy(); dfe['방영요일'] = dfe['방영요일'].apply(clean_cell_colab)
    dfe = dfe.explode('방영요일').dropna(subset=['방영요일','점수']).copy()
    dfe['방영요일'] = dfe['방영요일'].astype(str).str.strip().str.lower()
    ordered = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    day_ko = {'monday':'월','tuesday':'화','wednesday':'수','thursday':'목','friday':'금','saturday':'토','sunday':'일'}
    mean_by = dfe.groupby('방영요일')['점수'].mean().reindex(ordered)
    cnt_by = dfe['방영요일'].value_counts().reindex(ordered).fillna(0).astype(int)
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

    # ====== 요약(데이터) + 인사이트 출력 ======
    weekday = ['monday','tuesday','wednesday','thursday']
    weekend = ['friday','saturday','sunday']
    
    wk_avg = mean_by.loc[weekday].mean(skipna=True)
    we_avg = mean_by.loc[weekend].mean(skipna=True)
    
    top_day_en  = mean_by.idxmax() if mean_by.notna().any() else None
    top_day_ko  = day_ko.get(top_day_en, "N/A") if top_day_en else "N/A"
    top_mean    = float(mean_by.max()) if mean_by.notna().any() else float("nan")
    
    wk_cnt = int(cnt_by.loc[weekday].sum())
    we_cnt = int(cnt_by.loc[weekend].sum())
    
    st.markdown(
        f"""
    **요약(데이터 근거)**  
    - 주중 평균 평점(월~목): **{wk_avg:.3f}** · 주중 작품 수 합계: **{wk_cnt}편**  
    - 주말 평균 평점(금~일): **{we_avg:.3f}** · 주말 작품 수 합계: **{we_cnt}편**  
    - 평균 평점 최고 요일: **{top_day_ko} {top_mean:.3f}점**
    
    **인사이트(요약)**  
    - **주중(월~목)**: 일상 편성 비중이 높고, 다양한 연령/취향을 겨냥한 **보편적 콘텐츠**가 많음.  
      제작 속도·양산성, **시청률 지향** 편성이 상대적으로 두드러짐.  
    - **금요일**: 한 주 피로와 외부 일정(회식·외출·주말 준비)로 **실시간 시청 집중도 낮음** →  
      방송사는 **예능·뉴스·영화 대체 편성**을 택하는 경우가 많아 드라마 편성/성과가 약함.  
    - **일요일**: 다음 날이 월요일이라 **가벼운 콘텐츠 선호**. 전통적으로 예능이 프라임을 장악 →  
      드라마 **편성 수요 자체가 낮은 구조**.  
    - **토요일**: 시간적 여유 + 다음 날 휴식으로 **시청률·몰입·광고 효과가 최대**.  
      고품질 작품을 집중 투입하며 **경쟁이 가장 치열**한 요일.  
    
    **해석 메모**  
    - 주중은 피로·시간 제약 탓에 **완주율/몰입도가 낮아** 호불호가 강하게 나타나기 쉬움.  
    - 주말(특히 토요일) 편성은 기획 단계부터 **타깃·예산·완성도**가 높은 전략 편성이 많아,  
      **감정 몰입**이 잘 일어나고 좋은 작품일수록 **우호적 평가**가 형성되는 경향이 있습니다.
    """
    )


    # --- 주연 배우 성별 인원수 및 비율 ---
    st.subheader("주연 배우 성별 인원수 및 비율")

    # '주연'만 필터 + 성별 결측 제거
    main_roles = raw_df[raw_df['역할'] == '주연'].dropna(subset=['성별']).copy()
    
    # 성별별 인원수 / 확률
    gender_counts = main_roles['성별'].value_counts()
    total_main_roles = int(gender_counts.sum())
    gender_probs = (gender_counts / total_main_roles).reindex(gender_counts.index)
    
    # 색상(성별 개수에 맞춰 반복)
    palette = ['skyblue', 'lightpink', 'lightgreen', 'lightgray', 'orange', 'violet']
    colors = [palette[i % len(palette)] for i in range(len(gender_counts))]
    
    # 그래프
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(gender_counts.index.astype(str), gender_counts.values, color=colors)
    
    # 라벨: 인원수 + 확률(%) 표기
    for bar, prob in zip(bars, gender_probs.values):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, yval + max(2, yval*0.02),
            f'{int(yval)}명\n({prob*100:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax.set_title('주연 배우 성별 인원수 및 비율', fontsize=14)
    ax.set_ylabel('인원수'); ax.set_xlabel('성별')
    
    # 여유 공백
    ymax = gender_counts.max()
    ax.set_ylim(0, ymax + max(10, int(ymax*0.15)))
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # --- 방영년도별 작품 수 및 평균 점수 ---
    st.subheader("방영년도별 작품 수 및 평균 점수")
    
    # 숫자형 변환 & 결측 제거
    dfe = raw_df.copy()
    dfe['방영년도'] = pd.to_numeric(dfe['방영년도'], errors='coerce')
    dfe['점수']    = pd.to_numeric(dfe['점수'], errors='coerce')
    dfe = dfe.dropna(subset=['방영년도','점수']).copy()
    dfe['방영년도'] = dfe['방영년도'].astype(int)
    
    # 집계
    mean_score_by_year = dfe.groupby('방영년도')['점수'].mean().round(3)
    count_by_year      = dfe['방영년도'].value_counts()
    
    # x축 연도(둘의 합집합, 오름차순)
    years = sorted(set(mean_score_by_year.index) | set(count_by_year.index))
    mean_s = mean_score_by_year.reindex(years)
    count_s = count_by_year.reindex(years, fill_value=0)
    
    # 시각화
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 왼쪽 Y축: 작품 수 (막대)
    color_bar = 'tab:gray'
    ax1.set_xlabel('방영년도')
    ax1.set_ylabel('작품 수', color=color_bar)
    bars = ax1.bar(years, count_s.values, alpha=0.3, color=color_bar, width=0.6)
    ax1.tick_params(axis='y', labelcolor=color_bar)
    
    # 막대 위 수치
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + max(0.5, h*0.02),
                 f'{int(h)}', ha='center', va='bottom', fontsize=9, color='black')
    
    # 오른쪽 Y축: 평균 점수 (선)
    ax2 = ax1.twinx()
    color_line = 'tab:blue'
    ax2.set_ylabel('평균 점수', color=color_line)
    ax2.plot(years, mean_s.values, marker='o', color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)
    if mean_s.notna().any():
        ax2.set_ylim(mean_s.min() - 0.05, mean_s.max() + 0.05)
    
    # 점 위 수치
    for x, y in zip(years, mean_s.values):
        if pd.notna(y):
            ax2.text(x, y + 0.01, f'{y:.3f}', color=color_line, fontsize=9, ha='center')
    
    plt.title('방영년도별 작품 수 및 평균 점수')
    plt.tight_layout()
    st.pyplot(fig)

    # 유효 연도만 선택
    valid_mask = mean_s.notna() & count_s.notna()
    yrs   = pd.Index(years)[valid_mask]
    cnt   = count_s[valid_mask].astype(float)
    meanv = mean_s[valid_mask].astype(float)
    
    # (1) 전체 구간 상관계수
    r_all = float(np.corrcoef(cnt.values, meanv.values)[0, 1]) if len(yrs) >= 2 else np.nan
    
    # (2) 2017년 이후 상관계수 (데이터 있으면)
    mask_2017 = yrs >= 2017
    if mask_2017.any() and mask_2017.sum() >= 2:
        r_2017 = float(np.corrcoef(cnt[mask_2017].values, meanv[mask_2017].values)[0, 1])
    else:
        r_2017 = np.nan
    
    # (3) CAGR(작품 수) - 전체 / 2017→마지막
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
    
    # (4) 요약 표시
    st.markdown("**추가 통계 요약**")
    st.markdown(
        f"""
    - 전체 기간 상관계수 r(작품 수 vs 평균 점수): **{r_all:.3f}**  
    - 2017년 이후 상관계수 r: **{r_2017:.3f}**  
    - 작품 수 CAGR(전체 {first_year}→{last_year}): **{(cagr_all*100):.2f}%/년**  
    - 작품 수 CAGR(2017→{last_year}): **{(cagr_2017*100):.2f}%/년**
    """
    )
    
    # (선택) 해석 한 줄
    if not np.isnan(r_2017):
        trend = "음(-)의" if r_2017 < 0 else "양(+)의"
        st.caption(f"메모: 2017년 이후 구간에서 작품 수와 평균 점수는 **{trend} 상관**을 보입니다.")

    # --- 연령대별 작품 수 & 성별 평균 점수 (주연 배우 기준) ---
    st.subheader("연령대별 작품 수 및 성별 평균 점수 (주연 배우 기준)")
    
    import re
    import numpy as np
    
    # 1) 데이터 준비: 주연만, 필요한 컬럼 결측 제거
    main_roles = raw_df.copy()
    main_roles = main_roles[main_roles['역할'] == '주연']
    main_roles = main_roles.dropna(subset=['연령대','성별','점수']).copy()
    main_roles['점수'] = pd.to_numeric(main_roles['점수'], errors='coerce')
    main_roles = main_roles.dropna(subset=['점수'])
    
    # 2) 연령대 정렬 키 (예: '20대 후반'도 20으로 인식, '50대 이상'은 50)
    def age_key(s: str):
        m = re.search(r'(\d+)', str(s))
        return int(m.group(1)) if m else 999
    
    age_order = sorted(main_roles['연령대'].astype(str).unique(), key=age_key)
    
    # 3) 연령대별 작품 수
    age_counts = (main_roles['연령대']
                  .value_counts()
                  .reindex(age_order)
                  .fillna(0)
                  .astype(int))
    
    # 4) 성별+연령대별 평균 점수
    ga = (main_roles.groupby(['성별','연령대'])['점수']
          .mean()
          .round(3)
          .reset_index())
    
    male_vals   = ga[ga['성별']=='남자'].set_index('연령대').reindex(age_order)['점수']
    female_vals = ga[ga['성별']=='여자'].set_index('연령대').reindex(age_order)['점수']
    
    # 5) 시각화
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 막대: 작품 수
    bars = ax1.bar(age_order, age_counts.values, color='lightgray', label='작품 수')
    ax1.set_ylabel('작품 수', fontsize=12)
    ax1.set_ylim(0, max(age_counts.max()*1.2, age_counts.max()+2))
    
    # 막대 위 수치
    for rect in bars:
        h = rect.get_height()
        ax1.text(rect.get_x()+rect.get_width()/2, h + max(2, h*0.02),
                 f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 선: 평균 점수(이중축)
    ax2 = ax1.twinx()
    line1, = ax2.plot(age_order, male_vals.values, marker='o', linewidth=2, label='남자')
    line2, = ax2.plot(age_order, female_vals.values, marker='o', linewidth=2, label='여자')
    ax2.set_ylabel('평균 점수', fontsize=12)
    
    # y축 범위(데이터 기반)
    all_means = pd.concat([male_vals, female_vals]).dropna()
    if not all_means.empty:
        ymin = float(all_means.min()) - 0.05
        ymax = float(all_means.max()) + 0.05
        if ymin == ymax:  # 동일값 보호
            ymin, ymax = ymin-0.05, ymax+0.05
        ax2.set_ylim(ymin, ymax)
    
    # 점 위 수치
    for x, y in zip(age_order, male_vals.values):
        if not np.isnan(y):
            ax2.text(x, y + 0.004, f'{y:.3f}', color=line1.get_color(),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    for x, y in zip(age_order, female_vals.values):
        if not np.isnan(y):
            ax2.text(x, y + 0.004, f'{y:.3f}', color=line2.get_color(),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 제목/격자/범례
    plt.title('연령대별 작품 수 및 성별 평균 점수 (주연 배우 기준)', fontsize=14)
    ax1.set_xlabel('연령대', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    lines, labels = [bars, line1, line2], ['작품 수', '남자', '여자']
    ax1.legend(lines[1:], labels[1:], loc='upper left')  # 선만 범례로 표시
    
    plt.tight_layout()
    st.pyplot(fig)

    # ====== 그래프 아래 요약 & 인사이트 ======
    # 1) 작품 수 톱3
    age_top = age_counts.sort_values(ascending=False)
    top_items = [f"{idx} {int(val)}편" for idx, val in age_top.head(3).items()]
    top_txt = " · ".join(top_items) if len(top_items) else "N/A"
    
    # 2) 성별별 최고/최저 평균
    def safe_max(s):
        s = s.dropna()
        return (s.idxmax(), float(s.max())) if not s.empty else (None, np.nan)
    
    def safe_min(s):
        s = s.dropna()
        return (s.idxmin(), float(s.min())) if not s.empty else (None, np.nan)
    
    m_best_age, m_best = safe_max(male_vals)
    f_best_age, f_best = safe_max(female_vals)
    f_worst_age, f_worst = safe_min(female_vals)
    
    def fmt(v): 
        return f"{v:.3f}" if (v is not None and not np.isnan(v)) else "N/A"
    def nz(s): 
        return s if s is not None else "N/A"
    
    st.markdown(
        f"""
    **요약(데이터 근거)**  
    - 작품 수 상위: **{top_txt}**  
    - 남성 최고 평균: **{nz(m_best_age)} {fmt(m_best)}**  
    - 여성 최고 평균: **{nz(f_best_age)} {fmt(f_best)}**  
    - 여성 최저 평균: **{nz(f_worst_age)} {fmt(f_worst)}**
    
    **인사이트(요약)**  
    - **캐스팅 집중**: 로맨스·청춘·판타지·직장물·성장 서사 등 주류 장르의 주인공 설정이 **20–30대**에 맞춰져, 해당 연령대에 **주연 기회가 집중**됩니다.  
    - **50대+ 수량이 적은 이유**: 캐릭터가 부모/상사/조력자·악역 등 **조연 축**에 배치되기 쉬워 주연 편수가 적고, 신진 배우 유입도 제한적이라 **배우 풀 자체가 작음**.  
    - **성별 격차 패턴**:  
      - 남성은 **40–50대**에도 CEO/검사/형사/변호사 등 **중심축 역할**을 맡으며 평균 점수가 비교적 **안정적**입니다.  
      - 여성은 연령이 높아질수록 **중심 서사에서 비중이 줄어**(엄마·장모·할머니 등 주변 인물), **평균 점수가 하락하는 경향**이 관찰됩니다.  
    - **실무 시사점**: 연령·성별 편중을 완화하려면 **장르·캐릭터 설계 단계**에서 다양한 연령대의 **주도적 역할**을 의도적으로 기획하는 접근이 필요합니다.
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
    
        # 1) 장르 워드클라우드 + 인사이트
        if genre_list:
            wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path)\
                    .generate(' '.join(genre_list))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            st.pyplot(fig)
    
            top_g = top_pairs(genre_list, n=5)
            st.markdown(
                f"""
    **요약(데이터 근거)**  
    - 상위 장르: **{pairs_to_str(top_g)}**
    
    **인사이트**  
    - **romance / comedy / drama** 중심으로 물량이 쏠림 → 대중성·제작 효율이 높아 **반복 생산**되는 구조.  
    - **thriller / sf / hist_war** 등은 상대적으로 적지만 **팬덤 충성도/완성도 승부**로 평점이 높게 형성되는 경향.  
    - 혼합 표기(예: *romance thriller*)가 자주 보임 → **멀티 장르 트렌드**가 활발.
    """
            )
    
        # 2) 플랫폼 워드클라우드 + 인사이트
        if broadcaster_list:
            wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path)\
                    .generate(' '.join(broadcaster_list))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            st.pyplot(fig)
    
            top_p = top_pairs(broadcaster_list, n=6)
            st.markdown(
                f"""
    **요약(데이터 근거)**  
    - 상위 플랫폼: **{pairs_to_str(top_p)}**
    
    **인사이트**  
    - **KBS·MBC·SBS** 등 전통 채널이 여전히 주류, **TVN·NETFLIX**가 뒤를 추격 → **방송사 주도 + OTT 가세** 구도.  
    - 기타/공동 표기(ETC 등)는 **동시 방영·공동 유통**이 흔하다는 신호.  
    - 전략: 전통 채널 편성 확보와 함께 **OTT 협업/동시 공개**를 병행하는 하이브리드 유통이 유리.
    """
            )
    
        # 3) 방영요일 워드클라우드 + 인사이트
        if week_list:
            # 영문 소문자 통일
            wk = [str(w).strip().lower() for w in week_list if pd.notna(w)]
            wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path)\
                    .generate(' '.join(wk))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            st.pyplot(fig)
    
            # 요일 한글 매핑
            day_ko = {'monday':'월', 'tuesday':'화', 'wednesday':'수', 'thursday':'목',
                      'friday':'금', 'saturday':'토', 'sunday':'일'}
            top_w = top_pairs(wk, n=7)
            st.markdown(
                f"""
    **요약(데이터 근거)**  
    - 요일 빈도: **{pairs_to_str(top_w, label_map=day_ko)}**
    
    **인사이트**  
    - **주중(월~목)** 물량이 크고, **토요일**이 강세 → 토요일은 시청 여건·몰입·광고 효과가 높은 **황금 슬롯**.  
    - **금·일요일**은 예능/가벼운 콘텐츠와의 경쟁으로 드라마 편성 비중이 상대적으로 낮은 편.  
    - 전략: 주중은 **보편 장르**로 안정적 편성, 토요일은 **고품질·화제성** 승부, 금·일은 장르 선택과 마케팅을 차별화.
    """
            )

# --- 4.5 실시간 필터 ---
with tabs[4]:
    st.header("실시간 필터")
    smin,smax = float(raw_df['점수'].min()), float(raw_df['점수'].max())
    sfilter = st.slider("최소 평점", smin,smax,smin)
    yfilter = st.slider("방영년도 범위", int(raw_df['방영년도'].min()), int(raw_df['방영년도'].max()), (2000,2025))
    filt = raw_df[(raw_df['점수']>=sfilter) & raw_df['방영년도'].between(*yfilter)]
    st.dataframe(filt.head(20))

# --- 4.6 전체 미리보기 ---
with tabs[5]:
    st.header("원본 전체보기")
    st.dataframe(raw_df, use_container_width=True)

# --- 4.7 머신러닝 모델링 (Colab 설정 그대로) ---
# --- 4.8 GridSearch 튜닝 (모든 모델) ---
with tabs[6]:
    st.header("GridSearchCV 튜닝")

    # split 보장 (튜닝을 먼저 들어와도 동작하도록)
    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X_colab_base, y_all, test_size=test_size, random_state=SEED, shuffle=True
        )
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(test_size)
    X_train, X_test, y_train, y_test = st.session_state["split_colab"]

    scoring = st.selectbox("스코어링", ["neg_root_mean_squared_error", "r2"], index=0)
    cv = st.number_input("CV 폴드 수", 3, 10, 5, 1)

    model_zoo = {
        "KNN": ("nonsparse", KNeighborsRegressor()),
        "Linear Regression (Poly)": ("nonsparse", LinearRegression()),
        "Ridge": ("nonsparse", Ridge()),
        "Lasso": ("nonsparse", Lasso()),
        "ElasticNet": ("nonsparse", ElasticNet(max_iter=10000)),
        "SGDRegressor": ("nonsparse", SGDRegressor(max_iter=10000)),
        "SVR": ("nonsparse", SVR()),
        "Decision Tree": ("tree", DecisionTreeRegressor(random_state=SEED)),
        "Random Forest": ("tree", RandomForestRegressor(random_state=SEED)),
    }
    if 'XGBRegressor' in globals() and XGB_AVAILABLE:
        model_zoo["XGBRegressor"] = ("tree", XGBRegressor(
            random_state=SEED, objective="reg:squarederror",
            n_jobs=-1, tree_method="hist"
        ))

    def make_pipeline(kind, estimator):
        if kind == "tree":
            return Pipeline([('preprocessor', preprocessor), ('model', estimator)])
        else:
            return Pipeline([
                ('preprocessor', preprocessor),
                ('poly', PolynomialFeatures(include_bias=False)),
                ('scaler', StandardScaler(with_mean=False)),
                ('model', estimator)
            ])

    param_grids = {
        "KNN": {"poly__degree":[1,2,3], "model__n_neighbors":[3,4,5,6,7,8,9,10]},
        "Linear Regression (Poly)": {"poly__degree":[1,2,3]},
        "Ridge": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000]},
        "Lasso": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000]},
        "ElasticNet": {"poly__degree":[1,2,3], "model__alpha":[0.001,0.01,0.1,1,10,100,1000], "model__l1_ratio":[0.1,0.5,0.9]},
        "SGDRegressor": {"poly__degree":[1,2,3], "model__learning_rate":["constant","invscaling","adaptive"]},
        "SVR": {"poly__degree":[1,2,3], "model__kernel":["poly","rbf","sigmoid"], "model__degree":[1,2,3]},
        "Decision Tree": {"model__max_depth":[10,15,20,25,30], "model__min_samples_split":[5,6,7,8,9,10], "model__min_samples_leaf":[2,3,4,5], "model__max_leaf_nodes":[None,10,20,30]},
        "Random Forest": {"model__n_estimators":[100,200,300], "model__min_samples_split":[5,6,7,8,9,10], "model__max_depth":[5,10,15,20,25,30]},
    }
    if "XGBRegressor" in model_zoo:
        param_grids["XGBRegressor"] = {
            "model__n_estimators":[200,400],
            "model__max_depth":[3,5,7],
            "model__learning_rate":[0.03,0.1,0.3],
            "model__subsample":[0.8,1.0],
            "model__colsample_bytree":[0.8,1.0],
        }

    model_name = st.selectbox("튜닝할 모델 선택", list(model_zoo.keys()), index=0)
    kind, estimator = model_zoo[model_name]
    pipe = make_pipeline(kind, estimator)
    grid = param_grids[model_name]

    if st.button("GridSearch 실행"):
        gs = GridSearchCV(pipe, grid, cv=int(cv), scoring=scoring, n_jobs=-1, refit=True, return_train_score=True)
        with st.spinner("GridSearchCV 실행 중..."):
            gs.fit(X_train, y_train)

        st.subheader("베스트 결과")
        st.json(gs.best_params_)
        if scoring == "neg_root_mean_squared_error":
            st.write(f"Best CV RMSE: {-gs.best_score_:.6f}")
        else:
            st.write(f"Best CV {scoring}: {gs.best_score_:.6f}")

        y_pred = gs.predict(X_test)
        st.write(f"Test RMSE: {rmse(y_test, y_pred):.6f}")
        st.write(f"Test R²  : {r2_score(y_test, y_pred):.6f}")

        # ▶ 모델링 탭에서 즉시 활용할 수 있도록 저장
        st.session_state["best_estimator"] = gs.best_estimator_
        st.session_state["best_params"] = gs.best_params_
        st.session_state["best_name"] = model_name
        st.session_state["best_cv_score"] = gs.best_score_
        st.session_state["best_scoring"] = scoring
        st.session_state["best_split_key"] = st.session_state.get("split_key")

        cvres = pd.DataFrame(gs.cv_results_)
        cols = ["rank_test_score","mean_test_score","std_test_score","mean_train_score","std_train_score","params"]
        st.dataframe(cvres[cols].sort_values("rank_test_score").reset_index(drop=True))

    if model_name == "XGBRegressor" and not XGB_AVAILABLE:
        st.warning("xgboost가 설치되어 있지 않습니다. requirements.txt에 `xgboost`를 추가하고 재배포해 주세요.")

# --- 4.8 GridSearch 튜닝 (RandomForest, Colab 그리드) ---
# --- 4.7 머신러닝 모델링 (Colab 설정 그대로/베스트 자동 적용) ---
with tabs[7]:
    st.header("머신러닝 모델링 (Colab 설정)")

    # split 보장
    if "split_colab" not in st.session_state or st.session_state.get("split_key") != float(test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X_colab_base, y_all, test_size=test_size, random_state=SEED, shuffle=True
        )
        st.session_state["split_colab"] = (X_train, X_test, y_train, y_test)
        st.session_state["split_key"] = float(test_size)
    X_train, X_test, y_train, y_test = st.session_state["split_colab"]

    # ▶ 베스트 모델 있으면 사용, 없으면 RF 베이스라인
    if "best_estimator" in st.session_state:
        model = st.session_state["best_estimator"]  # 이미 fit됨
        st.caption(f"현재 모델: GridSearch 베스트 모델 사용 ({st.session_state.get('best_name')})")
        if st.session_state.get("best_split_key") != st.session_state.get("split_key"):
            st.warning("주의: 베스트 모델은 이전 분할로 학습됨. 새 분할로 다시 튜닝해 주세요.", icon="⚠️")
    else:
        model = Pipeline([('preprocessor', preprocessor),
                          ('model', RandomForestRegressor(random_state=SEED))])
        model.fit(X_train, y_train)
        st.caption("현재 모델: 기본 RandomForest (미튜닝)")

    # 지표 출력
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    st.metric("Train R²", f"{r2_score(y_train, y_pred_tr):.3f}")
    st.metric("Test  R²", f"{r2_score(y_test,  y_pred_te):.3f}")
    st.metric("Train RMSE", f"{rmse(y_train, y_pred_tr):.3f}")
    st.metric("Test  RMSE", f"{rmse(y_test,  y_pred_te):.3f}")

    if "best_params" in st.session_state:
        with st.expander("베스트 하이퍼파라미터 보기"):
            st.json(st.session_state["best_params"])

# --- 4.9 예측 실행 — 입력 묶음/장르구분 생성 & 베스트 모델 사용 ---
from sklearn.base import clone

with tabs[8]:
    st.header("평점 예측")

    # 선택지 준비
    genre_opts   = sorted({g for sub in raw_df['장르'].dropna().apply(clean_cell_colab) for g in sub})
    week_opts    = sorted({d for sub in raw_df['방영요일'].dropna().apply(clean_cell_colab) for d in sub})
    plat_opts    = sorted({p for sub in raw_df['플랫폼'].dropna().apply(clean_cell_colab) for p in sub})
    gender_opts  = sorted(raw_df['성별'].dropna().unique())
    role_opts    = sorted(raw_df['역할'].dropna().unique())
    quarter_opts = sorted(raw_df['방영분기'].dropna().unique())
    married_opts = sorted(raw_df['결혼여부'].dropna().unique())

    # 입력을 두 묶음으로 배치
    st.subheader("1) 입력")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**① 컨텐츠 특성**")
        input_age     = st.number_input("나이", 10, 80, 30)
        input_gender  = st.selectbox("성별", gender_opts) if gender_opts else st.text_input("성별 입력", "")
        input_role    = st.selectbox("역할", role_opts) if role_opts else st.text_input("역할 입력", "")
        input_married = st.selectbox("결혼여부", married_opts) if married_opts else st.text_input("결혼여부 입력", "")
        input_genre   = st.multiselect("장르 (멀티 선택)", genre_opts, default=genre_opts[:1] if genre_opts else [])

        # 나이 → 연령대 자동 산출 + 장르구분 생성
        derived_age_group = age_to_age_group(int(input_age))
        if len(input_genre) == 0:
            genre_group_label = "장르없음"
        elif len(input_genre) == 1:
            genre_group_label = "단일장르"
        else:
            genre_group_label = "멀티장르"

        st.caption(f"자동 연령대: **{derived_age_group}**  |  장르구분: **{genre_group_label}**")

    with col_right:
        st.markdown("**② 편성 특성**")
        input_quarter = st.selectbox("방영분기", quarter_opts) if quarter_opts else st.text_input("방영분기 입력", "")
        input_week    = st.multiselect("방영요일 (멀티 선택)", week_opts, default=week_opts[:1] if week_opts else [])
        input_plat    = st.multiselect("플랫폼 (멀티 선택)", plat_opts, default=plat_opts[:1] if plat_opts else [])

    predict_btn = st.button("예측 실행")

    if predict_btn:
        # 1) 예측 모델: 베스트 있으면 clone해서 전체 데이터로 재학습
        if "best_estimator" in st.session_state:
            model_full = clone(st.session_state["best_estimator"])
            st.caption(f"예측 모델: GridSearch 베스트 재학습 사용 ({st.session_state.get('best_name')})")
        else:
            model_full = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=SEED))
            ])
            st.caption("예측 모델: 기본 RandomForest (미튜닝)")

        # 2) 전체 데이터로 재학습
        model_full.fit(X_colab_base, y_all)

        # 3) 사용자 입력 → DF (멀티라벨은 리스트 유지, 장르구분 추가)
        user_raw = pd.DataFrame([{
            '나이'    : input_age,
            '성별'    : input_gender,
            '역할'    : input_role,
            '결혼여부': input_married,
            '방영분기': input_quarter,
            '연령대'  : derived_age_group,   # 자동 매핑
            '장르'    : input_genre,         # list
            '방영요일' : input_week,          # list
            '플랫폼'  : input_plat,          # list
            '장르구분' : genre_group_label,   # 새 파생 변수(현재 모델에는 미사용)
        }])

        # 4) 멀티라벨 변환 + X 스키마 정렬
        user_mlb = colab_multilabel_transform(user_raw, cols=('장르','방영요일','플랫폼'))

        # 학습 X 스키마와 컬럼 정합 (여분은 제거, 부족분은 0으로 채움)
        user_base = pd.concat([X_colab_base.iloc[:0].copy(), user_mlb], ignore_index=True)
        # 드롭 대상 제거(훈련 시 제외했던 컬럼들)
        user_base = user_base.drop(columns=[c for c in drop_cols if c in user_base.columns], errors='ignore')
        # 훈련에 없는 컬럼은 삭제, 있는데 빠진 컬럼은 0으로 채움
        for c in X_colab_base.columns:
            if c not in user_base.columns:
                user_base[c] = 0
        user_base = user_base[X_colab_base.columns].tail(1)

        # 5) 예측
        pred = model_full.predict(user_base)[0]
        st.success(f"💡 예상 평점: {pred:.2f}")

        # 참고: 장르구분을 실제 특징으로 쓰고 싶다면,
        #  - 학습 데이터(df_mlb)에도 동일 규칙으로 '장르구분'을 만들어 X_colab_base에 포함시키고
        #  - preprocessor의 범주형 목록에 '장르구분'을 추가하세요.

