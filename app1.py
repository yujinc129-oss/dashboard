import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import matplotlib
import matplotlib.font_manager as fm
import platform, os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px

if st.session_state.get("font_cache_cleared") is not True:
    import shutil
    cache_dir = matplotlib.get_cachedir()
    shutil.rmtree(cache_dir, ignore_errors=True)
    st.session_state["font_cache_cleared"] = True

def set_korean_font():
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 1) 프로젝트에 포함한 폰트(권장) - 파일을 여기에 두세요: ./fonts/NanumGothic.ttf
    candidates = [
        os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf"),
        os.path.join(os.getcwd(), "fonts", "NanumGothic.ttf"),
    ]

    # 2) Windows 기본 폰트 경로도 시도
    if platform.system() == "Windows":
        candidates += [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\malgunbd.ttf",
        ]

    # 3) 시스템 폰트 검색(마지막 폴백)
    sys_fonts = fm.findSystemFonts(fontext="ttf")
    for f in sys_fonts:
        if any(k in f.lower() for k in ("nanum", "malgun", "applegothic", "notosanscjk", "gulim", "dotum", "batang")):
            candidates.append(f)

    # 등록 시도
    for path in candidates:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)  # 폰트 등록
                family = fm.FontProperties(fname=path).get_name()
                matplotlib.rcParams['font.family'] = family
                return family
            except Exception:
                pass
    return None

family = set_korean_font()
# 선택된 폰트를 확인하고 싶으면:
# st.write("Using font:", family)

# =========================
# 0. 유틸리티 함수 및 클래스
# =========================
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    def fit(self, X, y=None):
        lists = X.squeeze()
        self.mlb.fit(lists)
        return self
    def transform(self, X):
        lists = X.squeeze()
        return self.mlb.transform(lists)

def clean_cell(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except:
            return [x.strip()]
    return [str(x)]

def safe_eval(val):
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
    return []

def flatten_list_str(x):
    if isinstance(x, list):
        return ','.join(map(str, x))
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return ','.join(map(str, parsed))
        except:
            pass
    return str(x) if not pd.isna(x) else ''

def preprocess_ml_features(X: pd.DataFrame) -> pd.DataFrame:
    for col in ['장르','플랫폼','방영요일']:
        if col in X.columns:
            X[col] = X[col].apply(safe_eval).apply(flatten_list_str)
    return X.fillna('')

# =========================
# 1. 데이터 로드
# =========================
@st.cache_data
def load_data():
    raw = pd.read_json('drama_data.json')
    return pd.DataFrame({c: pd.Series(v) for c,v in raw.items()})

raw_df = load_data()              # EDA용 원본
df      = raw_df.copy()           # ML용 복사본

# =========================
# 2. 머신러닝용 전처리
# =========================
mlb_cols = ['장르','플랫폼','방영요일']
for col in mlb_cols:
    df[col] = df[col].apply(clean_cell)
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(df[col])
    df = pd.concat([
        df,
        pd.DataFrame(arr, columns=[f"{col}_{c.upper()}" for c in mlb.classes_], index=df.index)
    ], axis=1)
#df.drop(columns=mlb_cols, inplace=True)

# =========================
# 3. EDA용 리스트 풀기
# =========================
genre_list       = [g for sub in raw_df['장르'].dropna().apply(safe_eval) for g in sub]
broadcaster_list = [b for sub in raw_df['플랫폼'].dropna().apply(safe_eval) for b in sub]
week_list        = [w for sub in raw_df['방영요일'].dropna().apply(safe_eval) for w in sub]
unique_genres    = sorted(set(genre_list))

# =========================
# 4. Streamlit 레이아웃
# =========================
st.set_page_config(page_title="K-드라마 분석/예측", page_icon="🎬", layout="wide")
st.title("K-드라마 데이터 분석 및 예측 대시보드")

# 사이드바: ML 파라미터 + 예측 입력
with st.sidebar:
    st.header("🤖 모델 설정")
    model_type   = st.selectbox('모델 선택', ['Random Forest','Linear Regression'])
    test_size    = st.slider('테스트셋 비율', 0.1,0.5,0.2,0.05)
    feature_cols = st.multiselect('특성 선택',['나이','방영년도','성별','장르','배우명','플랫폼','결혼여부'], default=['나이','방영년도','장르'])
    st.markdown("---")
    st.header("🎯 평점 예측 입력")
    input_age     = st.number_input("배우 나이",10,80,30)
    input_year    = st.number_input("방영년도",2000,2025,2021)
    input_gender  = st.selectbox("성별", sorted(raw_df['성별'].dropna().unique()))
    input_genre   = st.multiselect("장르", unique_genres, default=unique_genres[:1])
    input_plat    = st.multiselect("플랫폼", sorted(set(broadcaster_list)), default=list({broadcaster_list[0]}))
    input_married = st.selectbox("결혼여부", sorted(raw_df['결혼여부'].dropna().unique()))
    predict_btn   = st.button("예측 실행")

# 탭 구성
tabs = st.tabs(["🗂개요","📊기초통계","📈분포/교차","💬워드클라우드","⚙️필터","🔍전체보기","🤖ML모델","🔧튜닝","🎯예측"])

# --- 4.1 데이터 개요 ---
with tabs[0]:
    st.header("데이터 개요")
    c1,c2,c3 = st.columns(3)
    c1.metric("샘플 수", df.shape[0])
    c2.metric("컬럼 수", df.shape[1])
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
    ax.set_title("히스토그램")
    st.pyplot(fig)

# --- 4.3 분포/교차분석 ---
with tabs[2]:
    st.header("분포 및 교차분석")
    # 1) 점수 분포 & Top10
    st.subheader("전체 평점 분포")
    fig1 = px.histogram(raw_df, x='점수', nbins=20); st.plotly_chart(fig1)
    st.subheader("Top 10 평점 작품")
    top10 = raw_df.nlargest(10,'점수')[['드라마명','점수']].sort_values('점수')
    fig2 = px.bar(top10, x='점수', y='드라마명', orientation='h', text='점수'); st.plotly_chart(fig2)
    # 2) 연도별 플랫폼 수(원본 explode)
    st.subheader("연도별 주요 플랫폼 작품 수")
    ct = (
        pd.DataFrame({'방영년도': raw_df['방영년도'], '플랫폼': raw_df['플랫폼']})
        .explode('플랫폼')
        .groupby(['방영년도', '플랫폼']).size().reset_index(name='count'))
    ct['플랫폼_up'] = ct['플랫폼'].str.upper()
    focus = ['KBS', 'MBC', 'TVN', 'NETFLIX', 'SBS']

    fig3 = px.line(
        ct[ct['플랫폼_up'].isin(focus)],
        x='방영년도',
        y='count',
        color='플랫폼',
        log_y=True,  # y축 로그 스케일 적용
        title="연도별 주요 플랫폼 작품 수 (로그 스케일)")
    st.plotly_chart(fig3)

    # 3) 멀티장르 vs 단일장르
    st.subheader("멀티장르 vs 단일장르 평균 평점 (배우 단위 박스플롯)")

    # 배우별 장르 다양성 계산 → 멀티/단일 라벨
    ag = (
        pd.DataFrame({'배우명': raw_df['배우명'], '장르': raw_df['장르']})
        .explode('장르')
        .groupby('배우명')['장르'].nunique()
    )
    multi_set = set(ag[ag > 1].index)

    label_map = {name: ('멀티장르' if name in multi_set else '단일장르') for name in ag.index}

    # 배우 단위 평균 점수
    actor_mean = (
        raw_df.groupby('배우명', as_index=False)['점수'].mean()
              .rename(columns={'점수': '배우평균점수'})
    )
    actor_mean['장르구분'] = actor_mean['배우명'].map(label_map)

    # 박스플롯
    fig_box = px.box(
        actor_mean, 
        x='장르구분', 
        y='배우평균점수',
        title="멀티장르 vs 단일장르 배우 단위 평균 점수 분포"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    # 4) 주연 배우 결혼 상태별 평균 점수 비교
    st.subheader("주연 배우 결혼 상태별 평균 점수 비교")

    # 주연 배우만 필터링
    main_roles = raw_df[raw_df['역할'] == '주연'].copy()

    # 결혼상태 컬럼 생성: '미혼' vs '미혼 외'
    main_roles['결혼상태'] = main_roles['결혼여부'].apply(lambda x: '미혼' if x == '미혼' else '미혼 외')

    # 결혼 상태별 평균 점수 계산
    avg_scores_by_marriage = main_roles.groupby('결혼상태')['점수'].mean()

    # 시각화
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(avg_scores_by_marriage.index, avg_scores_by_marriage.values,
                  color=['mediumseagreen', 'gray'])

    # 수치 표시
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005,
                f'{yval:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title('주연 배우 결혼 상태별 평균 점수 비교', fontsize=14)
    ax.set_ylabel('평균 점수')
    ax.set_xlabel('결혼 상태')
    ax.set_ylim(min(avg_scores_by_marriage.values) - 0.05, max(avg_scores_by_marriage.values) + 0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    st.pyplot(fig)
    st.subheader("장르별 작품 수 및 평균 점수")

    # 한글 폰트(Windows) + 마이너스 깨짐 방지
    import matplotlib, platform
    if platform.system() == "Windows":
        matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # 장르가 리스트인 경우 펼치기 + 결측 제거
    df_exploded = raw_df.explode('장르')
    df_exploded = df_exploded.dropna(subset=['장르', '점수'])

    # 장르별 평균 점수 및 작품 수
    genre_score = df_exploded.groupby('장르')['점수'].mean().round(3)
    genre_count = df_exploded['장르'].value_counts()

    genre_df = (pd.DataFrame({'평균 점수': genre_score, '작품 수': genre_count})
                .reset_index().rename(columns={'index': '장르'}))

    # 보기 좋게 작품 수 기준 정렬
    genre_df = genre_df.sort_values('작품 수', ascending=False).reset_index(drop=True)

    # 시각화
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 막대그래프: 작품 수
    bars = ax1.bar(range(len(genre_df)), genre_df['작품 수'], color='lightgray')
    ax1.set_ylabel('작품 수', fontsize=12)
    ax1.set_xticks(range(len(genre_df)))
    ax1.set_xticklabels(genre_df['장르'], rotation=45, ha='right')

    # 막대 위 수치
    for i, rect in enumerate(bars):
        h = rect.get_height()
        ax1.text(i, h + max(2, h*0.01), f'{int(h)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444')

    # 선그래프: 평균 점수(보조축)
    ax2 = ax1.twinx()
    sns.lineplot(x=range(len(genre_df)), y=genre_df['평균 점수'],
                 marker='o', linewidth=2, ax=ax2)
    ax2.set_ylabel('평균 점수', fontsize=12)
    ax2.tick_params(axis='y')
    ax2.set_ylim(genre_df['평균 점수'].min() - 0.1, genre_df['평균 점수'].max() + 0.1)

    # 점 위 수치
    for i, v in enumerate(genre_df['평균 점수']):
        ax2.text(i, v + 0.01, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('장르별 작품 수 및 평균 점수', fontsize=14)
    ax1.set_xlabel('장르', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("방영 요일별 작품 수 및 평균 점수 (월→일)")

    import matplotlib.pyplot as plt
    import pandas as pd

    # 1) 데이터 전처리
    df_exploded = raw_df.explode('방영요일').dropna(subset=['방영요일', '점수']).copy()
    df_exploded['방영요일'] = df_exploded['방영요일'].astype(str).str.strip().str.lower()

    ordered_days_en = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    day_label_ko = {
        'monday':'월', 'tuesday':'화', 'wednesday':'수', 'thursday':'목',
        'friday':'금', 'saturday':'토', 'sunday':'일'
    }

    # 2) 집계 (없 는 요일은 0/NaN 처리 후 정렬)
    mean_score_by_day = (
        df_exploded.groupby('방영요일')['점수'].mean()
        .reindex(ordered_days_en)
    )
    count_by_day = (
        df_exploded['방영요일'].value_counts()
        .reindex(ordered_days_en)
        .fillna(0).astype(int)
    )

    # 3) 시각화
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 왼쪽 Y축: 작품 수(막대)
    bars = ax1.bar(ordered_days_en, count_by_day.values, alpha=0.3, color='tab:gray')
    ax1.set_ylabel('작품 수', color='tab:gray', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:gray')

    # 막대 위 수치
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{int(h)}',
                 ha='center', va='bottom', fontsize=9, color='black')

    # 오른쪽 Y축: 평균 점수(선)
    ax2 = ax1.twinx()
    ax2.plot(ordered_days_en, mean_score_by_day.values, marker='o', color='tab:blue')
    ax2.set_ylabel('평균 점수', color='tab:blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    if mean_score_by_day.notna().any():
        ax2.set_ylim(mean_score_by_day.min() - 0.05, mean_score_by_day.max() + 0.05)

    # 점 위 수치
    for x, y in zip(ordered_days_en, mean_score_by_day.values):
        if pd.notna(y):
            ax2.text(x, y + 0.005, f'{y:.3f}', color='tab:blue', fontsize=9, ha='center')

    # x축 한글 레이블
    ax1.set_xticks(ordered_days_en)
    ax1.set_xticklabels([day_label_ko[d] for d in ordered_days_en])

    plt.title('방영 요일별 작품 수 및 평균 점수 (월요일 → 일요일 순)', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)



# --- 4.4 워드클라우드 ---
with tabs[3]:
    st.header("워드클라우드")
    if genre_list:
        wc = WordCloud(width=800,height=400,background_color='white').generate(' '.join(genre_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    if broadcaster_list:
        wc = WordCloud(width=800,height=400,background_color='white').generate(' '.join(broadcaster_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)
    if week_list:
        wc = WordCloud(width=800,height=400,background_color='white').generate(' '.join(week_list))
        fig,ax=plt.subplots(); ax.imshow(wc,interpolation='bilinear'); ax.axis('off'); st.pyplot(fig)

# --- 4.5 실시간 필터 ---
with tabs[4]:
    st.header("실시간 필터")
    smin,smax = float(raw_df['점수'].min()), float(raw_df['점수'].max())
    sfilter = st.slider("최소 평점", smin,smax,smin)
    yfilter = st.slider("방영년도 범위", int(raw_df['방영년도'].min()),int(raw_df['방영년도'].max()),(2000,2025))
    filt = raw_df[(raw_df['점수']>=sfilter)&raw_df['방영년도'].between(*yfilter)]
    st.dataframe(filt.head(20))

# --- 4.6 전체 미리보기 ---
with tabs[5]:
    st.header("원본 전체보기")
    st.dataframe(raw_df, use_container_width=True)

# --- 4.7 머신러닝 모델링 ---
with tabs[6]:
    st.header("머신러닝 모델링")
    if feature_cols:
        X = df[feature_cols].copy()
        y = raw_df['점수'].astype(float)
        X = preprocess_ml_features(X)
        X = pd.get_dummies(X, drop_first=True)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
        model = RandomForestRegressor(random_state=42) if model_type=="Random Forest" else LinearRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        st.metric("R²", f"{r2_score(y_test,y_pred):.3f}")
        st.metric("MSE", f"{mean_squared_error(y_test,y_pred):.3f}")
    else:
        st.warning("사이드바에서 특성을 선택하세요.")

# --- 4.8 GridSearch 튜닝 ---
with tabs[7]:
    st.header("GridSearchCV 튜닝")
    st.info("모델을 선택하고 GridSearchCV를 실행해 보세요.")

    # ⚠️ 전제: X_train, X_test, y_train, y_test 가 이미 준비되어 있다고 가정
    # 회귀 문제 기준 scoring 예시: neg_mean_squared_error (원하면 바꿔도 됨)
    scoring = st.selectbox("스코어링", ["neg_mean_squared_error", "r2"], index=0)
    cv = st.number_input("CV 폴드 수", min_value=3, max_value=10, value=5, step=1)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    import pandas as pd
    import numpy as np

    # 1) 모델 팩토리 (전부 'model' 이라는 스텝 이름으로 통일)
    model_zoo = {
        "KNN": KNeighborsRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(max_iter=10000),
        "SGDRegressor": SGDRegressor(max_iter=10000),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
    }

    # 2) 공통 파이프라인: 다수 모델에선 Poly+Scale가 유효
    #    - 트리/랜덤포레스트는 다항/스케일 불필요 → 그 모델은 별도 파이프라인 사용
    def make_pipeline(name):
        if name in ["Decision Tree", "Random Forest"]:
            return Pipeline([("model", model_zoo[name])])  # 단순
        else:
            return Pipeline([
                ("poly", PolynomialFeatures(include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", model_zoo[name]),
            ])

    # 3) 파라미터 그리드 (모두 model__* 로 정규화)
    #    * 요청 주신 범위를 그대로 반영하되, 오타/네이밍을 파이프라인에 맞게 수정
    param_grids = {
        "KNN": {
            "poly__degree": [1, 2, 3],
            "model__n_neighbors": [3,4,5,6,7,8,9,10],
        },
        "Linear Regression": {
            "poly__degree": [1, 2, 3],
        },
        "Ridge": {
            "poly__degree": [1, 2, 3],
            "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        },
        "Lasso": {
            "poly__degree": [1, 2, 3],
            "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        },
        "ElasticNet": {
            "poly__degree": [1, 2, 3],
            "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
        "SGDRegressor": {
            "poly__degree": [1, 2, 3],
            "model__learning_rate": ["constant", "invscaling", "adaptive"],
            # 필요시 eta0 등도 추가 가능: "model__eta0": [0.001, 0.01, 0.1]
        },
        "SVR": {
            "poly__degree": [1, 2, 3],  # poly 커널일 때만 의미 있지만, 함께 튜닝해도 무방
            "model__kernel": ["poly", "rbf", "sigmoid"],
            "model__degree": [1, 2, 3],
        },
        "Decision Tree": {
            # poly/scale 없음
            "model__max_depth": [10, 15, 20, 25, 30],
            "model__min_samples_split": [5, 6, 7, 8, 9, 10],
            "model__min_samples_leaf": [2, 3, 4, 5],
            "model__max_leaf_nodes": [None, 10, 20, 30],
        },
        "Random Forest": {
            # poly/scale 없음
            "model__n_estimators": [100, 200, 300],
            "model__min_samples_split": [5, 6, 7, 8, 9, 10],
            "model__max_depth": [5, 10, 15, 20, 25, 30],
        },
    }

    model_name = st.selectbox(
        "튜닝할 모델 선택",
        list(model_zoo.keys()),
        index=0
    )

    run = st.button("GridSearch 실행")

    if run:
        pipe = make_pipeline(model_name)
        grid = param_grids[model_name]

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        with st.spinner("GridSearchCV 실행 중..."):
            gs.fit(X_train, y_train)

        st.success("완료!")

        # 결과 요약
        st.subheader("베스트 결과")
        st.write("Best Params:", gs.best_params_)
        st.write("Best CV Score:", gs.best_score_)

        # 테스트셋 평가
        y_pred = gs.predict(X_test)
        from sklearn.metrics import mean_squared_error, r2_score
        test_mse = mean_squared_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        st.write(f"Test MSE: {test_mse:.6f}")
        st.write(f"Test R2 : {test_r2:.6f}")

        # CV 상세 결과 표
        cvres = pd.DataFrame(gs.cv_results_)
        wanted_cols = [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "params",
        ]
        st.dataframe(cvres[wanted_cols].sort_values("rank_test_score").reset_index(drop=True))


# --- 4.9 예측 실행 ---
with tabs[8]:
    st.header("평점 예측")
    st.subheader("1) 모델 설정")
    model_type  = st.selectbox('모델 선택', ['Random Forest', 'Linear Regression'])
    test_size   = st.slider('테스트셋 비율', 0.1, 0.5, 0.2, 0.05)
    feature_cols = st.multiselect(
        '특성 선택',
        ['나이','방영년도','성별','장르','배우명','플랫폼','결혼여부'],
        default=['나이','방영년도','장르']
    )

    st.markdown("---")
    st.subheader("2) 예측 입력")
    input_age     = st.number_input("배우 나이", 10, 80, 30)
    input_year    = st.number_input("방영년도", 2000, 2025, 2021)
    input_gender  = st.selectbox("성별", sorted(df['성별'].dropna().unique()))

    genre_opts    = sorted(unique_genres)
    default_genre = [genre_opts[0]] if genre_opts else []
    input_genre   = st.multiselect("장르", genre_opts, default=default_genre)

    platform_opts = sorted(set(broadcaster_list))
    default_plat  = [platform_opts[0]] if platform_opts else []
    input_plat    = st.multiselect("플랫폼", platform_opts, default=default_plat)

    input_married = st.selectbox("결혼여부", sorted(df['결혼여부'].dropna().unique()))

    predict_btn   = st.button("예측 실행")

    if predict_btn:
        # --- 1. 훈련 데이터 전처리 ---
        X_all = raw_df[feature_cols].copy()
        y_all = df['점수'].astype(float)
        X_all = preprocess_ml_features(X_all)
        X_all = pd.get_dummies(X_all, columns=[c for c in X_all.columns if X_all[c].dtype=='object'])

        # --- 2. 입력 데이터 전처리 ---
        user_df = pd.DataFrame([{
            '나이': input_age,
            '방영년도': input_year,
            '성별': input_gender,
            '장르': input_genre,
            '배우명': df['배우명'].dropna().iloc[0],  # 필요시 selectbox로 변경
            '플랫폼': input_plat,
            '결혼여부': input_married
        }])
        u = preprocess_ml_features(user_df)
        u = pd.get_dummies(u, columns=[c for c in u.columns if u[c].dtype=='object'])
        for c in X_all.columns:
            if c not in u.columns:
                u[c] = 0
        u = u[X_all.columns]

        # --- 3. 모델 학습 & 예측 ---
        model = RandomForestRegressor(n_estimators=100, random_state=42) \
                if model_type=="Random Forest" else LinearRegression()
        model.fit(X_all, y_all)
        pred = model.predict(u)[0]

        st.success(f"💡 예상 평점: {pred:.2f}")


# =========================
# 6. 예측 실행
# =========================
        if predict_btn:
            user_input = pd.DataFrame([{
                '나이': input_age,
                '방영년도': input_year,
                '성별': input_gender,
                '장르': input_genre,
                '배우명': st.selectbox("배우명", sorted(df['배우명'].dropna().unique())),  # 모델링 탭과 동일하게
                '플랫폼': input_plat,
                '결혼여부': input_married
            }])
        
            # 전처리 & 인코딩
            X_all = df[feature_cols].copy()
            y_all = df['점수'].astype(float)
            X_all = preprocess_ml_features(X_all)
            X_all = pd.get_dummies(X_all, columns=[c for c in X_all.columns if X_all[c].dtype == 'object'])
        
            user_proc = preprocess_ml_features(user_input)
            user_proc = pd.get_dummies(user_proc, columns=[c for c in user_proc.columns if user_proc[c].dtype == 'object'])
            for col in X_all.columns:
                if col not in user_proc.columns:
                    user_proc[col] = 0
            user_proc = user_proc[X_all.columns]
        
            # 모델 학습 & 예측
            model = RandomForestRegressor(n_estimators=100, random_state=42) if model_type=="Random Forest" else LinearRegression()
            model.fit(X_all, y_all)
            prediction = model.predict(user_proc)[0]
        
            st.success(f"💡 예상 평점: {prediction:.2f}")
