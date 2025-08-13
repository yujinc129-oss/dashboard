# preprocess_v2.py
# -*- coding: utf-8 -*-
import ast
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ------------------------------------------------------------
# 0) 데이터 로드 (행 순서 고정)
# ------------------------------------------------------------
def load_data(json_path: str = "drama_d.json") -> pd.DataFrame:
    """
    drama_d.json({col: {row_idx: value}} 형태)를 DataFrame으로 로드.
    행 순서를 0..N-1로 고정해 CV/GS 재현성 보장.
    """
    raw = pd.read_json(json_path)
    if isinstance(raw, pd.Series):
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    else:
        raw = pd.DataFrame({c: pd.Series(v) for c, v in raw.to_dict().items()})
    return raw.reset_index(drop=True)


# ------------------------------------------------------------
# 1) 멀티라벨 셀 정리
# ------------------------------------------------------------
def clean_cell_colab(x):
    """
    문자열/리스트/결측 혼재한 셀을 리스트[str]로 정리.
    "['a','b']" 같은 문자열도 안전하게 파싱.
    """
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


# ------------------------------------------------------------
# 2) MultiLabelBinarizer: fit_transform / transform
#    state는 Streamlit st.session_state 같은 dict로 전달 가능
# ------------------------------------------------------------
def _get_state(state: Optional[dict]) -> dict:
    class _Dummy(dict): ...
    return state if state is not None else _Dummy()

def colab_multilabel_fit_transform(
    df: pd.DataFrame,
    cols: Tuple[str, ...] = ("genres", "day", "network"),
    state: Optional[dict] = None
) -> pd.DataFrame:
    """
    학습 단계: 멀티라벨 컬럼을 MLBin으로 원-핫(클래스 대문자화).
    세션(state)에 클래스 저장 → 추론 단계에서 동일 스키마 보장.
    """
    state = _get_state(state)
    out = df.copy()

    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)

        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(out[col])
        classes = [c.strip().upper() for c in mlb.classes_]  # 대문자 통일
        new_cols = [f"{col}_{c}" for c in classes]

        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors="ignore")
        out = pd.concat([out, pd.DataFrame(arr, columns=new_cols, index=out.index)], axis=1)

        # 세션 저장(대문자 클래스 버전)
        state[f"mlb_{col}"] = None  # 실객체는 pickle 이슈될 수 있으니 클래스만 저장
        state[f"mlb_classes_{col}"] = classes

    return out


def colab_multilabel_transform(
    df: pd.DataFrame,
    cols: Tuple[str, ...] = ("genres", "day", "network"),
    state: Optional[dict] = None,
    ref_df_mlb: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    추론 단계: 저장된 클래스(세션) 사용해 동일 원-핫 스키마 생성.
    세션이 비어있으면 ref_df_mlb(학습 시 생성된 df_mlb)에서 스키마 유추.
    """
    state = _get_state(state)
    out = df.copy()

    for col in cols:
        out[col] = out[col].apply(clean_cell_colab)

        classes = state.get(f"mlb_classes_{col}")
        if not classes and ref_df_mlb is not None:
            prefix = f"{col}_"
            classes = [c[len(prefix):] for c in ref_df_mlb.columns if c.startswith(prefix)]

        if not classes:
            # 마지막 폴백: 입력에서 fit (재현성↓)
            mlb = MultiLabelBinarizer()
            mlb.fit(out[col])
            classes = [c.strip().upper() for c in mlb.classes_]

        # target 스키마에 맞춰 transform
        # points: MultiLabelBinarizer 없이 직접 매핑
        binarized = np.zeros((len(out), len(classes)), dtype=int)
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        for i, vals in enumerate(out[col]):
            for v in vals:
                vv = str(v).strip().upper()
                j = cls_to_idx.get(vv)
                if j is not None:
                    binarized[i, j] = 1

        new_cols = [f"{col}_{c}" for c in classes]
        out = out.drop(columns=[c for c in new_cols if c in out.columns], errors="ignore")
        out = pd.concat([out, pd.DataFrame(binarized, columns=new_cols, index=out.index)], axis=1)

    return out


# ------------------------------------------------------------
# 3) 숫자형/문자형 처리, OHE 카테고리 고정, 학습 객체 생성
# ------------------------------------------------------------
CAT_CANDIDATES = ["role", "gender", "air_q", "married", "age_group"]
MLB_COLS = ("genres", "day", "network")

def _to_numeric_inplace(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _collect_ohe_categories(df: pd.DataFrame, cat_cols: List[str]) -> List[np.ndarray]:
    cats = []
    for c in cat_cols:
        vals = pd.Series(df[c].dropna().astype(str).unique())
        # 정렬 고정(사전식) → OHE 컬럼 순서 재현
        cats.append(np.array(sorted(vals)))
    return cats

def build_training_objects(
    raw_df: pd.DataFrame,
    state: Optional[dict] = None,
    include_year: bool = False  # True면 'start airing'을 수치 특성으로 포함
):
    """
    raw_df로부터 학습용 객체 생성:
    - df_mlb: 멀티라벨 원-핫 병합된 원본
    - X_colab_base: 학습 입력(드롭/포함 정책 반영)
    - y_all: 타깃(score)
    - preprocessor: 고정 카테고리 OHE가 들어간 ColumnTransformer
    - drop_cols: 원본에서 드롭한 컬럼(추론 정렬용)
    - ohe_categories: OHE에 쓴 카테고리(재현성 보장)
    """
    state = _get_state(state)

    # 1) 멀티라벨 원-핫
    df_mlb = colab_multilabel_fit_transform(raw_df, cols=MLB_COLS, state=state)

    # 2) 숫자형 캐스팅
    _to_numeric_inplace(df_mlb, ["score", "start airing"])

    # 3) 입력/타깃 분리
    base_drop = ["배우명", "드라마명", "genres", "day", "network", "score"]
    if not include_year:
        base_drop.append("start airing")  # 기존 앱과 동일(연도 제외)
    drop_cols = [c for c in base_drop if c in df_mlb.columns]

    X_colab_base = df_mlb.drop(columns=drop_cols, errors="ignore").copy()
    y_all = df_mlb["score"].copy()

    # 4) OHE 카테고리 고정(훈련 데이터에서 추출) + Dense 보장 + 카테고리 순서 고정
    cat_cols = [c for c in CAT_CANDIDATES if c in X_colab_base.columns]
    raw_cats = _collect_ohe_categories(X_colab_base, cat_cols)  # 보통 dict 또는 list-like
    
    # dict -> list-of-lists 변환 + 사전식 정렬로 순서 고정
    if isinstance(raw_cats, dict):
        ohe_categories = [sorted([str(v) for v in raw_cats[c]]) for c in cat_cols]
    else:
        # 이미 list-of-lists라면 정렬만 한 번 더 보장
        ohe_categories = [sorted([str(v) for v in arr]) for arr in raw_cats]
    
    # sklearn 1.2+ / 1.1- 호환: Dense OHE 강제
    ohe_common = dict(categories=ohe_categories, handle_unknown="ignore", drop="first")
    try:
        ohe = OneHotEncoder(sparse_output=False, **ohe_common)  # >=1.2
    except TypeError:
        ohe = OneHotEncoder(sparse=False, **ohe_common)         # <=1.1
    
    preprocessor = ColumnTransformer(
        transformers=[("cat", ohe, cat_cols)],
        remainder="passthrough"
    )

    # 5) 학습 컬럼 순서 고정(예: 정렬) → 추론 시 컬럼 정렬 안정화
    X_colab_base = X_colab_base.reindex(sorted(X_colab_base.columns), axis=1)

    return X_colab_base, y_all, preprocessor, drop_cols, df_mlb, cat_cols, ohe_categories


# ------------------------------------------------------------
# 4) 추론: 사용자 입력을 학습 스키마에 정렬
# ------------------------------------------------------------
def align_user_to_training(
    user_raw: pd.DataFrame,
    X_colab_base: pd.DataFrame,
    drop_cols: List[str],
    state: Optional[dict] = None,
    ref_df_mlb: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    user_raw(원 스키마) → 멀티라벨 transform → 드롭 → 누락 컬럼 채움 → 학습 컬럼 순서로 정렬.
    """
    state = _get_state(state)

    user_mlb = colab_multilabel_transform(
        user_raw, cols=MLB_COLS, state=state, ref_df_mlb=ref_df_mlb
    )

    user_base = pd.concat([X_colab_base.iloc[:0].copy(), user_mlb], ignore_index=True)
    user_base = user_base.drop(columns=[c for c in drop_cols if c in user_base.columns], errors="ignore")

    # 누락 컬럼 0으로 채우기(원-핫/멀티라벨)
    for c in X_colab_base.columns:
        if c not in user_base.columns:
            user_base[c] = 0

    # 학습 컬럼 순서와 동일하게
    user_base = user_base.reindex(X_colab_base.columns, axis=1)

    # 단일 샘플만 반환
    return user_base.tail(1)
