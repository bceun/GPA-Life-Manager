from html import escape
from io import BytesIO
from itertools import product

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="Campus Twin",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# 사용자 정의 스타일
# =========================================================
st.markdown(
    """
    <style>
    .recommendation-card {
        border: 1px solid #d9dee8;
        border-radius: 14px;
        padding: 20px;
        min-height: 420px;
        background-color: #ffffff;
        margin-bottom: 10px;
        overflow-wrap: break-word;
    }

    .recommendation-card-selected {
        border: 3px solid #ff4b4b;
        border-radius: 14px;
        padding: 18px;
        min-height: 420px;
        background-color: #fff5f5;
        box-shadow: 0 4px 14px rgba(255, 75, 75, 0.16);
        margin-bottom: 10px;
        overflow-wrap: break-word;
    }

    .card-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .card-score-label {
        font-size: 0.95rem;
        margin-bottom: 4px;
    }

    .card-score {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .card-delta-positive {
        color: #11823b;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .card-reason {
        background-color: #eaf3ff;
        border-radius: 9px;
        padding: 12px;
        margin-top: 12px;
        margin-bottom: 10px;
    }

    .selected-badge {
        display: inline-block;
        padding: 4px 9px;
        margin-bottom: 10px;
        border-radius: 20px;
        background-color: #ff4b4b;
        color: white;
        font-size: 0.85rem;
        font-weight: 700;
    }

    .difficulty-box-low {
        background-color: #eaf8ef;
        border-left: 5px solid #2ca25f;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
    }

    .difficulty-box-medium {
        background-color: #fff8df;
        border-left: 5px solid #e6a700;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
    }

    .difficulty-box-high {
        background-color: #fff0f0;
        border-left: 5px solid #d94343;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
    }

    .step-card {
        border: 1px solid #d9dee8;
        border-radius: 12px;
        padding: 16px;
        background-color: #fafbfc;
        min-height: 230px;
    }

    .progress-summary {
        text-align: center;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 모델 및 점수 설정
# =========================================================
MODEL_TARGET_MIN = 0.0
MODEL_TARGET_MAX = 2.009058
DISPLAY_SCORE_MAX = 4.5
EPSILON = 1e-8

DAYS = [
    "월요일",
    "화요일",
    "수요일",
    "목요일",
    "금요일",
    "토요일",
    "일요일"
]


# =========================================================
# 모델 불러오기
# =========================================================
@st.cache_resource
def load_models():
    model = joblib.load("lr_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


try:
    lr_model, scaler = load_models()

except Exception as error:
    st.error(
        "모델 파일을 불러오지 못했습니다. "
        "lr_model.pkl과 scaler.pkl 파일을 확인해 주세요."
    )
    st.exception(error)
    st.stop()


# =========================================================
# 변수 설정
# =========================================================
FEATURE_COLS = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "screen_time",
    "physical_activity",
    "stress"
]

FEATURE_LABELS = {
    "study_hours": "공부 시간",
    "attendance": "출석률",
    "sleep_hours": "수면 시간",
    "screen_time": "스크린타임",
    "physical_activity": "신체활동",
    "stress": "스트레스"
}

FEATURE_UNITS = {
    "study_hours": "시간",
    "attendance": "%",
    "sleep_hours": "시간",
    "screen_time": "시간",
    "physical_activity": "시간",
    "stress": "단계"
}

LABEL_TO_FEATURE = {
    label: feature
    for feature, label in FEATURE_LABELS.items()
}


SENSITIVITY_SETTINGS = {
    "study_hours": {
        "change": 1.0,
        "min": 0.0,
        "max": 12.0,
        "description": "공부 시간 1시간 증가"
    },
    "attendance": {
        "change": 5.0,
        "min": 0.0,
        "max": 100.0,
        "description": "출석률 5% 증가"
    },
    "sleep_hours": {
        "change": 1.0,
        "min": 0.0,
        "max": 12.0,
        "description": "수면 시간 1시간 증가"
    },
    "screen_time": {
        "change": -1.0,
        "min": 0.0,
        "max": 16.0,
        "description": "스크린타임 1시간 감소"
    },
    "physical_activity": {
        "change": 1.0,
        "min": 0.0,
        "max": 8.0,
        "description": "신체활동 1시간 증가"
    },
    "stress": {
        "change": -1.0,
        "min": 1.0,
        "max": 10.0,
        "description": "스트레스 1단계 감소"
    }
}


# =========================================================
# 예측 관련 함수
# =========================================================
def make_input_df(user_values):
    input_df = pd.DataFrame([user_values])

    missing_cols = [
        column
        for column in FEATURE_COLS
        if column not in input_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"필요한 입력 컬럼이 없습니다: {missing_cols}"
        )

    return input_df[FEATURE_COLS]


def convert_to_display_score(model_score):
    clipped_score = np.clip(
        model_score,
        MODEL_TARGET_MIN,
        MODEL_TARGET_MAX
    )

    converted_score = (
        (clipped_score - MODEL_TARGET_MIN)
        / (MODEL_TARGET_MAX - MODEL_TARGET_MIN)
        * DISPLAY_SCORE_MAX
    )

    return float(
        np.clip(
            converted_score,
            0.0,
            DISPLAY_SCORE_MAX
        )
    )


def predict_score(input_df):
    input_scaled = scaler.transform(input_df)
    prediction = lr_model.predict(input_scaled)

    raw_score = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    raw_score = float(
        np.clip(
            raw_score,
            MODEL_TARGET_MIN,
            MODEL_TARGET_MAX
        )
    )

    return {
        "raw_score": raw_score,
        "display_score": convert_to_display_score(
            raw_score
        )
    }


def predict_from_values(user_values):
    return predict_score(
        make_input_df(user_values)
    )


def calculate_total_hours(user_values):
    return (
        user_values["study_hours"]
        + user_values["sleep_hours"]
        + user_values["screen_time"]
        + user_values["physical_activity"]
    )


def create_range(start, stop, step):
    if stop < start:
        return np.array([float(start)])

    values = np.arange(
        start,
        stop + step / 2,
        step
    )

    return np.round(values, 2)


# =========================================================
# 민감도 분석 및 비교
# =========================================================
def analyze_sensitivity(user_values):
    base_prediction = predict_from_values(
        user_values
    )

    results = []

    for feature, setting in SENSITIVITY_SETTINGS.items():
        changed_values = user_values.copy()

        changed_value = float(
            np.clip(
                changed_values[feature]
                + setting["change"],
                setting["min"],
                setting["max"]
            )
        )

        changed_values[feature] = changed_value

        changed_prediction = predict_from_values(
            changed_values
        )

        results.append({
            "생활요인 변화": setting["description"],
            "현재 값": user_values[feature],
            "변경 값": changed_value,
            "변경 후 환산 점수": (
                changed_prediction["display_score"]
            ),
            "환산 점수 변화량": (
                changed_prediction["display_score"]
                - base_prediction["display_score"]
            ),
            "원본 점수 변화량": (
                changed_prediction["raw_score"]
                - base_prediction["raw_score"]
            )
        })

    result_df = pd.DataFrame(results)

    return result_df.sort_values(
        "환산 점수 변화량",
        ascending=False
    ).reset_index(drop=True)


def compare_plans(current_values, future_values):
    current_prediction = predict_from_values(
        current_values
    )

    future_prediction = predict_from_values(
        future_values
    )

    comparison_rows = []

    for feature in FEATURE_COLS:
        comparison_rows.append({
            "생활요인": FEATURE_LABELS[feature],
            "현재": current_values[feature],
            "미래 계획": future_values[feature],
            "변화": (
                future_values[feature]
                - current_values[feature]
            ),
            "단위": FEATURE_UNITS[feature]
        })

    return {
        "current_raw_score": current_prediction["raw_score"],
        "future_raw_score": future_prediction["raw_score"],
        "current_display_score": (
            current_prediction["display_score"]
        ),
        "future_display_score": (
            future_prediction["display_score"]
        ),
        "display_score_change": (
            future_prediction["display_score"]
            - current_prediction["display_score"]
        ),
        "comparison_df": pd.DataFrame(
            comparison_rows
        )
    }


# =========================================================
# 추천 알고리즘
# =========================================================
def calculate_change_cost(
    current_values,
    candidate_values
):
    normalized_changes = {
        "study_hours": (
            abs(
                candidate_values["study_hours"]
                - current_values["study_hours"]
            ) / 12.0
        ),
        "attendance": (
            abs(
                candidate_values["attendance"]
                - current_values["attendance"]
            ) / 100.0
        ),
        "sleep_hours": (
            abs(
                candidate_values["sleep_hours"]
                - current_values["sleep_hours"]
            ) / 12.0
        ),
        "screen_time": (
            abs(
                candidate_values["screen_time"]
                - current_values["screen_time"]
            ) / 16.0
        ),
        "physical_activity": (
            abs(
                candidate_values["physical_activity"]
                - current_values["physical_activity"]
            ) / 8.0
        ),
        "stress": (
            abs(
                candidate_values["stress"]
                - current_values["stress"]
            ) / 9.0
        )
    }

    return float(
        sum(normalized_changes.values())
    )


def describe_change_cost(change_cost):
    if change_cost < 0.08:
        return "낮음"

    if change_cost < 0.18:
        return "보통"

    return "높음"


def calculate_difficulty_score(change_cost):
    difficulty = int(
        np.ceil(change_cost * 12)
    )

    return int(
        np.clip(
            difficulty,
            1,
            5
        )
    )


def describe_difficulty(difficulty_score):
    descriptions = {
        1: "매우 쉬움",
        2: "쉬움",
        3: "보통",
        4: "어려움",
        5: "매우 어려움"
    }

    return descriptions.get(
        difficulty_score,
        "보통"
    )


def get_difficulty_message(difficulty_score):
    if difficulty_score <= 2:
        return {
            "class_name": "difficulty-box-low",
            "title": "무리 없이 시작할 수 있는 계획입니다.",
            "message": (
                "현재 생활과의 차이가 크지 않습니다. "
                "이번 주부터 목표 수준으로 실천해도 부담이 비교적 적습니다."
            )
        }

    if difficulty_score == 3:
        return {
            "class_name": "difficulty-box-medium",
            "title": "일정 조정이 필요한 계획입니다.",
            "message": (
                "한 번에 모든 목표를 적용하기보다 "
                "공부·수면·출석 계획을 단계적으로 조정하는 것이 좋습니다."
            )
        }

    return {
        "class_name": "difficulty-box-high",
        "title": "단계적으로 실행할 것을 권장합니다.",
        "message": (
            "생활 변화 폭이 큰 계획입니다. "
            "3주 단계적 계획을 따라 목표 수준에 천천히 도달하세요."
        )
    }


def is_same_plan(first_plan, second_plan):
    return all(
        abs(
            float(first_plan[feature])
            - float(second_plan[feature])
        ) < EPSILON
        for feature in FEATURE_COLS
    )


def has_any_change(
    current_values,
    candidate_values
):
    return not is_same_plan(
        current_values,
        candidate_values
    )


def generate_candidate_values(
    current_values,
    changeable_features,
    max_study,
    min_sleep,
    max_screen_reduction,
    max_attendance_increase
):
    if "study_hours" in changeable_features:
        study_stop = min(
            max_study,
            current_values["study_hours"] + 2.0,
            12.0
        )

        study_values = create_range(
            current_values["study_hours"],
            study_stop,
            0.5
        )

    else:
        study_values = np.array([
            current_values["study_hours"]
        ])

    if "sleep_hours" in changeable_features:
        sleep_start = max(
            current_values["sleep_hours"],
            min_sleep
        )

        sleep_stop = min(
            12.0,
            current_values["sleep_hours"] + 1.5
        )

        if sleep_start > sleep_stop:
            sleep_values = np.array([
                current_values["sleep_hours"]
            ])

        else:
            sleep_values = create_range(
                sleep_start,
                sleep_stop,
                0.5
            )

    else:
        sleep_values = np.array([
            current_values["sleep_hours"]
        ])

    if "screen_time" in changeable_features:
        screen_start = max(
            0.0,
            current_values["screen_time"]
            - max_screen_reduction
        )

        screen_values = create_range(
            screen_start,
            current_values["screen_time"],
            0.5
        )

    else:
        screen_values = np.array([
            current_values["screen_time"]
        ])

    if "attendance" in changeable_features:
        attendance_stop = min(
            100.0,
            current_values["attendance"]
            + max_attendance_increase
        )

        attendance_values = create_range(
            current_values["attendance"],
            attendance_stop,
            1.0
        )

    else:
        attendance_values = np.array([
            current_values["attendance"]
        ])

    return {
        "study_hours": study_values,
        "attendance": attendance_values,
        "sleep_hours": sleep_values,
        "screen_time": screen_values
    }


def generate_recommendation_candidates(
    current_values,
    target_score,
    changeable_features,
    max_study,
    min_sleep,
    max_screen_reduction,
    max_attendance_increase
):
    candidate_values = generate_candidate_values(
        current_values=current_values,
        changeable_features=changeable_features,
        max_study=max_study,
        min_sleep=min_sleep,
        max_screen_reduction=max_screen_reduction,
        max_attendance_increase=max_attendance_increase
    )

    current_prediction = predict_from_values(
        current_values
    )

    candidates = []

    combinations = product(
        candidate_values["study_hours"],
        candidate_values["attendance"],
        candidate_values["sleep_hours"],
        candidate_values["screen_time"]
    )

    for (
        study_hours,
        attendance,
        sleep_hours,
        screen_time
    ) in combinations:

        candidate = current_values.copy()

        candidate["study_hours"] = float(
            study_hours
        )
        candidate["attendance"] = float(
            attendance
        )
        candidate["sleep_hours"] = float(
            sleep_hours
        )
        candidate["screen_time"] = float(
            screen_time
        )

        total_hours = calculate_total_hours(
            candidate
        )

        if total_hours > 24:
            continue

        prediction = predict_from_values(
            candidate
        )

        change_cost = calculate_change_cost(
            current_values,
            candidate
        )

        difficulty_score = (
            calculate_difficulty_score(
                change_cost
            )
        )

        candidates.append({
            "study_hours": candidate["study_hours"],
            "attendance": candidate["attendance"],
            "sleep_hours": candidate["sleep_hours"],
            "screen_time": candidate["screen_time"],
            "physical_activity": (
                candidate["physical_activity"]
            ),
            "stress": candidate["stress"],
            "예상 환산 점수": (
                prediction["display_score"]
            ),
            "원본 예측 점수": (
                prediction["raw_score"]
            ),
            "현재 대비 점수 변화": (
                prediction["display_score"]
                - current_prediction["display_score"]
            ),
            "변화 부담 수치": change_cost,
            "변화 부담": describe_change_cost(
                change_cost
            ),
            "실행 난이도 점수": difficulty_score,
            "실행 난이도": describe_difficulty(
                difficulty_score
            ),
            "목표 달성": (
                prediction["display_score"]
                >= target_score
            ),
            "현재와 다른 계획": has_any_change(
                current_values,
                candidate
            ),
            "현재보다 개선": (
                prediction["display_score"]
                > current_prediction["display_score"]
                + EPSILON
            ),
            "주요 생활시간 합계": total_hours
        })

    return pd.DataFrame(candidates)


def plan_key(row):
    return tuple(
        round(float(row[feature]), 4)
        for feature in FEATURE_COLS
    )


def choose_unique_row(
    sorted_dataframe,
    used_plan_keys
):
    for _, row in sorted_dataframe.iterrows():
        key = plan_key(row)

        if key not in used_plan_keys:
            used_plan_keys.add(key)
            return row

    return None


def select_recommendations(
    candidates,
    target_score
):
    if candidates.empty:
        return {
            "target_reached": False,
            "recommendations": pd.DataFrame(),
            "candidate_pool_type": "후보 없음"
        }

    changed_candidates = candidates[
        candidates["현재와 다른 계획"]
    ].copy()

    if changed_candidates.empty:
        return {
            "target_reached": False,
            "recommendations": pd.DataFrame(),
            "candidate_pool_type": "변경 후보 없음"
        }

    reached_candidates = changed_candidates[
        changed_candidates["예상 환산 점수"]
        >= target_score
    ].copy()

    target_reached = not reached_candidates.empty

    if target_reached:
        selection_pool = reached_candidates.copy()
        candidate_pool_type = "목표 달성 후보"

    else:
        improved_candidates = changed_candidates[
            changed_candidates["현재보다 개선"]
        ].copy()

        if not improved_candidates.empty:
            selection_pool = improved_candidates.copy()
            candidate_pool_type = "목표 미달 개선 후보"

        else:
            selection_pool = changed_candidates.copy()
            candidate_pool_type = "목표 미달 변경 후보"

    selection_pool["균형 점수"] = (
        (
            selection_pool["예상 환산 점수"]
            / DISPLAY_SCORE_MAX
        )
        * 0.65
        + (
            1.0
            - np.clip(
                selection_pool["변화 부담 수치"],
                0.0,
                1.0
            )
        )
        * 0.35
    )

    if target_reached:
        minimum_label = "최소 변화형"
        balanced_label = "균형형"
        performance_label = "성과 우선형"

    else:
        minimum_label = "최소 변화 개선형"
        balanced_label = "균형 개선형"
        performance_label = "최고 성과형"

    used_plan_keys = set()

    minimum_sorted = selection_pool.sort_values(
        by=[
            "변화 부담 수치",
            "예상 환산 점수"
        ],
        ascending=[
            True,
            False
        ]
    )

    minimum_row = choose_unique_row(
        minimum_sorted,
        used_plan_keys
    )

    balanced_sorted = selection_pool.sort_values(
        by=[
            "균형 점수",
            "예상 환산 점수",
            "변화 부담 수치"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    balanced_row = choose_unique_row(
        balanced_sorted,
        used_plan_keys
    )

    performance_sorted = selection_pool.sort_values(
        by=[
            "예상 환산 점수",
            "변화 부담 수치"
        ],
        ascending=[
            False,
            True
        ]
    )

    performance_row = choose_unique_row(
        performance_sorted,
        used_plan_keys
    )

    selected_rows = [
        (
            minimum_label,
            minimum_row,
            "현재 생활과의 차이가 가장 작은 계획입니다."
        ),
        (
            balanced_label,
            balanced_row,
            "예상 성과와 생활 변화 부담을 함께 고려한 계획입니다."
        ),
        (
            performance_label,
            performance_row,
            "설정한 조건에서 가장 높은 점수를 예측한 계획입니다."
        )
    ]

    recommendation_rows = []

    for recommendation_type, row, reason in selected_rows:
        if row is None:
            continue

        recommendation_rows.append({
            "추천 유형": recommendation_type,
            "공부 시간": row["study_hours"],
            "출석률": row["attendance"],
            "수면 시간": row["sleep_hours"],
            "스크린타임": row["screen_time"],
            "신체활동": row["physical_activity"],
            "스트레스": row["stress"],
            "예상 환산 점수": (
                row["예상 환산 점수"]
            ),
            "현재 대비 점수 변화": (
                row["현재 대비 점수 변화"]
            ),
            "원본 예측 점수": (
                row["원본 예측 점수"]
            ),
            "변화 부담": row["변화 부담"],
            "변화 부담 수치": (
                row["변화 부담 수치"]
            ),
            "실행 난이도": row["실행 난이도"],
            "실행 난이도 점수": (
                row["실행 난이도 점수"]
            ),
            "주요 생활시간 합계": (
                row["주요 생활시간 합계"]
            ),
            "추천 이유": reason
        })

    return {
        "target_reached": target_reached,
        "recommendations": pd.DataFrame(
            recommendation_rows
        ),
        "candidate_pool_type": candidate_pool_type
    }


# =========================================================
# 추천 설명 및 계획 함수
# =========================================================
def build_change_summary(
    current_values,
    recommendation
):
    changes = []

    mapping = {
        "study_hours": (
            "공부 시간",
            "공부 시간",
            "시간"
        ),
        "attendance": (
            "출석률",
            "출석률",
            "%p"
        ),
        "sleep_hours": (
            "수면 시간",
            "수면 시간",
            "시간"
        ),
        "screen_time": (
            "스크린타임",
            "스크린타임",
            "시간"
        )
    }

    for current_key, (
        recommendation_column,
        label,
        unit
    ) in mapping.items():

        current_value = float(
            current_values[current_key]
        )

        recommended_value = float(
            recommendation[recommendation_column]
        )

        difference = (
            recommended_value
            - current_value
        )

        if abs(difference) < EPSILON:
            continue

        direction = (
            "증가"
            if difference > 0
            else "감소"
        )

        if current_key == "attendance":
            changes.append(
                f"{label} "
                f"{abs(difference):.0f}{unit} "
                f"{direction}"
            )

        else:
            changes.append(
                f"{label} "
                f"{abs(difference):.1f}{unit} "
                f"{direction}"
            )

    if not changes:
        return "현재 생활 유지"

    return ", ".join(changes)


def generate_simple_action_plan(
    current_values,
    recommendation
):
    actions = []
    weekend_actions = []
    goals = []
    checks = []

    study_target = float(
        recommendation["공부 시간"]
    )

    attendance_target = float(
        recommendation["출석률"]
    )

    sleep_target = float(
        recommendation["수면 시간"]
    )

    screen_target = float(
        recommendation["스크린타임"]
    )

    study_change = (
        study_target
        - current_values["study_hours"]
    )

    attendance_change = (
        attendance_target
        - current_values["attendance"]
    )

    sleep_change = (
        sleep_target
        - current_values["sleep_hours"]
    )

    screen_change = (
        screen_target
        - current_values["screen_time"]
    )

    if study_change > EPSILON:
        actions.append(
            f"하루 공부 {study_target:.1f}시간 확보"
        )
        goals.append(
            f"공부 {study_target:.1f}시간"
        )
        checks.append(
            "공부 목표 달성 여부"
        )

    else:
        actions.append(
            f"공부 {study_target:.1f}시간 유지"
        )
        goals.append(
            "현재 공부 루틴 유지"
        )

    if attendance_change > EPSILON:
        actions.append(
            f"출석률 {attendance_target:.0f}% 목표"
        )
        goals.append(
            f"출석률 {attendance_target:.0f}%"
        )
        checks.append(
            "지각·결석 여부"
        )

    if sleep_change > EPSILON:
        actions.append(
            f"수면 {sleep_target:.1f}시간 확보"
        )
        goals.append(
            f"수면 {sleep_target:.1f}시간"
        )
        checks.append(
            "취침·기상 시간"
        )

    else:
        actions.append(
            f"수면 {sleep_target:.1f}시간 유지"
        )

    if screen_change < -EPSILON:
        actions.append(
            f"스크린타임 {screen_target:.1f}시간 이하"
        )
        goals.append(
            f"스크린타임 {screen_target:.1f}시간 이하"
        )
        checks.append(
            "스크린타임 확인"
        )

    else:
        actions.append(
            f"스크린타임 {screen_target:.1f}시간 이내"
        )

    weekend_actions = [
        "한 주 학습 내용 복습",
        f"수면 {sleep_target:.1f}시간 유지",
        f"스크린타임 {screen_target:.1f}시간 이하"
    ]

    if not checks:
        checks.append(
            "계획 실천 여부"
        )

    weekly_rows = []

    for day in DAYS:
        if day in [
            "토요일",
            "일요일"
        ]:
            study_text = (
                f"복습 포함 {study_target:.1f}시간 이내"
            )
            attendance_text = "해당 없음"

        else:
            study_text = f"{study_target:.1f}시간"
            attendance_text = (
                f"{attendance_target:.0f}% 수준 유지"
            )

        weekly_rows.append({
            "요일": day,
            "공부 목표": study_text,
            "출석 목표": attendance_text,
            "수면 목표": f"{sleep_target:.1f}시간",
            "스크린타임 목표": (
                f"{screen_target:.1f}시간 이하"
            )
        })

    return {
        "weekday_actions": actions,
        "weekend_actions": weekend_actions,
        "core_goals": goals,
        "check_items": checks,
        "weekly_df": pd.DataFrame(
            weekly_rows
        )
    }


# =========================================================
# 3주 단계적 실행계획
# =========================================================
def interpolate_value(
    current_value,
    target_value,
    ratio
):
    return (
        current_value
        + (target_value - current_value)
        * ratio
    )


def generate_three_week_plan(
    current_values,
    recommendation
):
    target_values = {
        "study_hours": float(
            recommendation["공부 시간"]
        ),
        "attendance": float(
            recommendation["출석률"]
        ),
        "sleep_hours": float(
            recommendation["수면 시간"]
        ),
        "screen_time": float(
            recommendation["스크린타임"]
        )
    }

    ratios = [
        ("1주차", 1 / 3, "생활 변화에 적응"),
        ("2주차", 2 / 3, "목표 수준에 근접"),
        ("3주차", 1.0, "최종 추천 목표 실천")
    ]

    rows = []

    for week_name, ratio, key_goal in ratios:
        study_value = interpolate_value(
            current_values["study_hours"],
            target_values["study_hours"],
            ratio
        )

        attendance_value = interpolate_value(
            current_values["attendance"],
            target_values["attendance"],
            ratio
        )

        sleep_value = interpolate_value(
            current_values["sleep_hours"],
            target_values["sleep_hours"],
            ratio
        )

        screen_value = interpolate_value(
            current_values["screen_time"],
            target_values["screen_time"],
            ratio
        )

        rows.append({
            "주차": week_name,
            "공부 시간": round(study_value, 1),
            "출석률": round(attendance_value, 1),
            "수면 시간": round(sleep_value, 1),
            "스크린타임": round(screen_value, 1),
            "핵심 목표": key_goal
        })

    return pd.DataFrame(rows)


# =========================================================
# 추천 비교
# =========================================================
def make_recommendation_comparison(
    current_values,
    recommendations
):
    rows = []

    for _, recommendation in recommendations.iterrows():
        rows.append({
            "추천 유형": recommendation["추천 유형"],
            "공부 시간 변화": (
                recommendation["공부 시간"]
                - current_values["study_hours"]
            ),
            "출석률 변화": (
                recommendation["출석률"]
                - current_values["attendance"]
            ),
            "수면 시간 변화": (
                recommendation["수면 시간"]
                - current_values["sleep_hours"]
            ),
            "스크린타임 변화": (
                recommendation["스크린타임"]
                - current_values["screen_time"]
            ),
            "예상 환산 점수": (
                recommendation["예상 환산 점수"]
            ),
            "현재 대비 점수 변화": (
                recommendation["현재 대비 점수 변화"]
            ),
            "변화 부담": recommendation["변화 부담"],
            "실행 난이도": recommendation["실행 난이도"]
        })

    return pd.DataFrame(rows)


# =========================================================
# 카드 HTML
# =========================================================
def build_recommendation_card_html(
    row,
    current_values,
    selected_type
):
    is_selected = (
        row["추천 유형"]
        == selected_type
    )

    card_class = (
        "recommendation-card-selected"
        if is_selected
        else "recommendation-card"
    )

    selected_badge = (
        '<div class="selected-badge">선택된 추천안</div>'
        if is_selected
        else ""
    )

    recommendation_type = escape(
        str(row["추천 유형"])
    )

    recommendation_reason = escape(
        str(row["추천 이유"])
    )

    change_summary = escape(
        build_change_summary(
            current_values,
            row
        )
    )

    return (
        f'<div class="{card_class}">'
        f'{selected_badge}'
        f'<div class="card-title">{recommendation_type}</div>'
        f'<div class="card-score-label">예상 학업성과 점수</div>'
        f'<div class="card-score">'
        f'{row["예상 환산 점수"]:.2f} / {DISPLAY_SCORE_MAX:.2f}'
        f'</div>'
        f'<div class="card-delta-positive">'
        f'↑ {row["현재 대비 점수 변화"]:+.2f}'
        f'</div>'
        f'<p><b>공부 시간:</b> {row["공부 시간"]:.1f}시간</p>'
        f'<p><b>출석률:</b> {row["출석률"]:.0f}%</p>'
        f'<p><b>수면 시간:</b> {row["수면 시간"]:.1f}시간</p>'
        f'<p><b>스크린타임:</b> {row["스크린타임"]:.1f}시간</p>'
        f'<p><b>실행 난이도:</b> '
        f'{escape(str(row["실행 난이도"]))} '
        f'({int(row["실행 난이도 점수"])}/5)</p>'
        f'<div class="card-reason">{recommendation_reason}</div>'
        f'<small>{change_summary}</small>'
        f'</div>'
    )


# =========================================================
# 엑셀 함수
# =========================================================
def apply_excel_style(workbook):
    header_fill = PatternFill(
        fill_type="solid",
        fgColor="D9EAF7"
    )

    header_font = Font(
        bold=True
    )

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )

        for column_cells in worksheet.columns:
            max_length = 0

            column_letter = get_column_letter(
                column_cells[0].column
            )

            for cell in column_cells:
                cell.alignment = Alignment(
                    vertical="center",
                    wrap_text=True
                )

                if cell.value is not None:
                    max_length = max(
                        max_length,
                        len(str(cell.value))
                    )

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                max(max_length + 3, 12),
                40
            )

        for row in worksheet.iter_rows(
            min_row=2
        ):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.00"

        worksheet.sheet_view.showGridLines = False


def recommendations_to_xlsx(
    recommendations,
    comparison_df,
    current_values,
    target_score
):
    output = BytesIO()

    current_df = pd.DataFrame({
        "항목": [
            "현재 공부 시간",
            "현재 출석률",
            "현재 수면 시간",
            "현재 스크린타임",
            "현재 신체활동",
            "현재 스트레스",
            "목표 학업성과 점수"
        ],
        "값": [
            current_values["study_hours"],
            current_values["attendance"],
            current_values["sleep_hours"],
            current_values["screen_time"],
            current_values["physical_activity"],
            current_values["stress"],
            target_score
        ]
    })

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        recommendations.to_excel(
            writer,
            sheet_name="추천 계획",
            index=False
        )

        comparison_df.to_excel(
            writer,
            sheet_name="현재 대비 변화",
            index=False
        )

        current_df.to_excel(
            writer,
            sheet_name="입력 조건",
            index=False
        )

        apply_excel_style(writer.book)

    output.seek(0)

    return output.getvalue()


def selected_plan_to_xlsx(
    selected_plan,
    weekly_plan,
    three_week_plan,
    current_values
):
    output = BytesIO()

    selected_df = pd.DataFrame([
        selected_plan.to_dict()
    ])

    summary_df = pd.DataFrame({
        "구분": [
            "추천 유형",
            "생활 변화",
            "추천 이유",
            "실행 난이도"
        ],
        "내용": [
            selected_plan["추천 유형"],
            build_change_summary(
                current_values,
                selected_plan
            ),
            selected_plan["추천 이유"],
            (
                f"{selected_plan['실행 난이도']} "
                f"({selected_plan['실행 난이도 점수']}/5)"
            )
        ]
    })

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        selected_df.to_excel(
            writer,
            sheet_name="선택한 추천안",
            index=False
        )

        summary_df.to_excel(
            writer,
            sheet_name="계획 요약",
            index=False
        )

        three_week_plan.to_excel(
            writer,
            sheet_name="3주 단계계획",
            index=False
        )

        weekly_plan["weekly_df"].to_excel(
            writer,
            sheet_name="주간 실천표",
            index=False
        )

        pd.DataFrame({
            "평일 실천사항": (
                weekly_plan["weekday_actions"]
            )
        }).to_excel(
            writer,
            sheet_name="평일 계획",
            index=False
        )

        pd.DataFrame({
            "주말 실천사항": (
                weekly_plan["weekend_actions"]
            )
        }).to_excel(
            writer,
            sheet_name="주말 계획",
            index=False
        )

        apply_excel_style(writer.book)

    output.seek(0)

    return output.getvalue()


# =========================================================
# 체크박스 상태 관리
# =========================================================
def clear_daily_checkboxes():
    keys_to_delete = [
        key
        for key in st.session_state.keys()
        if str(key).startswith("daily_check_")
    ]

    for key in keys_to_delete:
        del st.session_state[key]


# =========================================================
# 세션 상태 초기화
# =========================================================
DEFAULT_SESSION_VALUES = {
    "current_values": None,
    "prediction_result": None,
    "sensitivity_df": None,
    "comparison_result": None,
    "recommendation_result": None,
    "recommendation_conditions": None,
    "selected_recommendation_type": None,
    "previous_recommendation_type": None
}

for key, default_value in DEFAULT_SESSION_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================================================
# 상단 소개
# =========================================================
st.title("🎓 Campus Twin")

st.subheader(
    "AI 기반 대학생활 시뮬레이션 및 생활계획 추천"
)

st.write("""
생활습관을 바탕으로 학업성과를 분석하고,
미래 계획을 시뮬레이션할 수 있습니다.

목표와 현실적인 조건을 입력하면 추천 계획을 탐색하고,
선택한 추천안을 3주 단계계획과 일주일 실천표로 변환합니다.
""")

st.warning("""
학업성과 환산 점수는 실제 학교 성적이나 공식 GPA가 아닙니다.
모델의 원본 예측값을 4.5점 범위로 변환한 참고용 지표입니다.
""")


# =========================================================
# 탭 구성
# =========================================================
(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5
) = st.tabs([
    "현재 상태",
    "개인별 분석",
    "미래 시뮬레이션",
    "추천 계획",
    "프로젝트 설명"
])


# =========================================================
# 탭 1: 현재 상태
# =========================================================
with tab1:
    st.header("현재 생활습관 입력")

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        study_hours = st.number_input(
            "하루 공부 시간",
            min_value=0.0,
            max_value=12.0,
            value=4.0,
            step=0.5,
            key="current_study"
        )

        attendance = st.number_input(
            "출석률",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
            key="current_attendance"
        )

        sleep_hours = st.number_input(
            "하루 수면 시간",
            min_value=0.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
            key="current_sleep"
        )

    with input_col2:
        screen_time = st.number_input(
            "하루 스크린타임",
            min_value=0.0,
            max_value=16.0,
            value=5.0,
            step=0.5,
            key="current_screen"
        )

        physical_activity = st.number_input(
            "하루 신체활동 시간",
            min_value=0.0,
            max_value=8.0,
            value=1.0,
            step=0.5,
            key="current_activity"
        )

        stress = st.slider(
            "스트레스 수준",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key="current_stress"
        )

    current_values = {
        "study_hours": study_hours,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "screen_time": screen_time,
        "physical_activity": physical_activity,
        "stress": stress
    }

    total_hours = calculate_total_hours(
        current_values
    )

    st.caption(
        f"입력된 주요 생활시간 합계: "
        f"{total_hours:.1f}시간"
    )

    if total_hours > 24:
        st.warning(
            "주요 생활시간 합이 하루 24시간을 초과합니다."
        )

    if st.button(
        "현재 상태 분석하기",
        type="primary"
    ):
        try:
            st.session_state.current_values = (
                current_values.copy()
            )

            st.session_state.prediction_result = (
                predict_from_values(
                    current_values
                )
            )

            st.session_state.sensitivity_df = (
                analyze_sensitivity(
                    current_values
                )
            )

            st.session_state.comparison_result = None
            st.session_state.recommendation_result = None
            st.session_state.recommendation_conditions = None
            st.session_state.selected_recommendation_type = None
            st.session_state.previous_recommendation_type = None

            clear_daily_checkboxes()

            if "recommendation_selector" in st.session_state:
                del st.session_state[
                    "recommendation_selector"
                ]

        except Exception as error:
            st.error(
                "예측 중 오류가 발생했습니다."
            )
            st.exception(error)

    if st.session_state.prediction_result is not None:
        prediction_result = (
            st.session_state.prediction_result
        )

        st.divider()
        st.subheader("예측 결과")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "학업성과 환산 점수",
                (
                    f"{prediction_result['display_score']:.2f}"
                    f" / {DISPLAY_SCORE_MAX:.2f}"
                )
            )

        with result_col2:
            st.metric(
                "모델 원본 예측 점수",
                f"{prediction_result['raw_score']:.3f}"
            )

        st.caption(
            "예측 결과는 실제 성적을 확정하거나 보장하지 않습니다."
        )


# =========================================================
# 탭 2: 개인별 분석
# =========================================================
with tab2:
    st.header("개인별 민감도 분석")

    if (
        st.session_state.current_values is None
        or st.session_state.sensitivity_df is None
    ):
        st.info(
            "'현재 상태' 탭에서 먼저 분석해 주세요."
        )

    else:
        sensitivity_df = (
            st.session_state.sensitivity_df.copy()
        )

        display_df = sensitivity_df.copy()

        display_df["현재 값"] = (
            display_df["현재 값"]
            .map(lambda value: f"{value:.1f}")
        )

        display_df["변경 값"] = (
            display_df["변경 값"]
            .map(lambda value: f"{value:.1f}")
        )

        display_df["변경 후 환산 점수"] = (
            display_df["변경 후 환산 점수"]
            .map(lambda value: f"{value:.2f}")
        )

        display_df["환산 점수 변화량"] = (
            display_df["환산 점수 변화량"]
            .map(lambda value: f"{value:+.3f}")
        )

        display_df["원본 점수 변화량"] = (
            display_df["원본 점수 변화량"]
            .map(lambda value: f"{value:+.4f}")
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        chart_df = sensitivity_df[
            [
                "생활요인 변화",
                "환산 점수 변화량"
            ]
        ].set_index(
            "생활요인 변화"
        )

        st.bar_chart(
            chart_df,
            use_container_width=True
        )

        best_result = sensitivity_df.iloc[0]

        st.success(
            f"가장 큰 예측 상승 요인: "
            f"{best_result['생활요인 변화']} "
            f"({best_result['환산 점수 변화량']:+.3f})"
        )

        st.caption(
            "모델 예측 반응이며 실제 인과관계를 의미하지 않습니다."
        )


# =========================================================
# 탭 3: 미래 시뮬레이션
# =========================================================
with tab3:
    st.header("미래 생활계획 시뮬레이션")

    if st.session_state.current_values is None:
        st.info(
            "'현재 상태' 탭에서 먼저 분석해 주세요."
        )

    else:
        current = st.session_state.current_values

        future_col1, future_col2 = st.columns(2)

        with future_col1:
            future_study = st.number_input(
                "계획 공부 시간",
                0.0,
                12.0,
                float(current["study_hours"]),
                0.5,
                key="future_study"
            )

            future_attendance = st.number_input(
                "계획 출석률",
                0.0,
                100.0,
                float(current["attendance"]),
                1.0,
                key="future_attendance"
            )

            future_sleep = st.number_input(
                "계획 수면 시간",
                0.0,
                12.0,
                float(current["sleep_hours"]),
                0.5,
                key="future_sleep"
            )

        with future_col2:
            future_screen = st.number_input(
                "계획 스크린타임",
                0.0,
                16.0,
                float(current["screen_time"]),
                0.5,
                key="future_screen"
            )

            future_activity = st.number_input(
                "계획 신체활동 시간",
                0.0,
                8.0,
                float(current["physical_activity"]),
                0.5,
                key="future_activity"
            )

            future_stress = st.slider(
                "계획 스트레스 수준",
                1.0,
                10.0,
                float(current["stress"]),
                0.5,
                key="future_stress"
            )

        future_values = {
            "study_hours": future_study,
            "attendance": future_attendance,
            "sleep_hours": future_sleep,
            "screen_time": future_screen,
            "physical_activity": future_activity,
            "stress": future_stress
        }

        if st.button(
            "현재 계획과 비교하기",
            type="primary"
        ):
            try:
                st.session_state.comparison_result = (
                    compare_plans(
                        current,
                        future_values
                    )
                )

            except Exception as error:
                st.error(
                    "미래 계획 비교 중 오류가 발생했습니다."
                )
                st.exception(error)

        if st.session_state.comparison_result is not None:
            comparison_result = (
                st.session_state.comparison_result
            )

            st.divider()

            metric_col1, metric_col2, metric_col3 = (
                st.columns(3)
            )

            metric_col1.metric(
                "현재 점수",
                f"{comparison_result['current_display_score']:.2f}"
            )

            metric_col2.metric(
                "미래 점수",
                f"{comparison_result['future_display_score']:.2f}"
            )

            metric_col3.metric(
                "예상 변화",
                f"{comparison_result['display_score_change']:+.2f}"
            )

            st.dataframe(
                comparison_result["comparison_df"],
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# 탭 4: 추천 계획
# =========================================================
with tab4:
    st.header("목표 기반 추천 계획")

    if st.session_state.current_values is None:
        st.info(
            "'현재 상태' 탭에서 먼저 분석해 주세요."
        )

    else:
        current = st.session_state.current_values

        current_prediction = predict_from_values(
            current
        )

        st.metric(
            "현재 학업성과 환산 점수",
            (
                f"{current_prediction['display_score']:.2f}"
                f" / {DISPLAY_SCORE_MAX:.2f}"
            )
        )

        st.subheader("목표 및 현실성 조건")

        condition_col1, condition_col2 = st.columns(2)

        with condition_col1:
            default_target = min(
                DISPLAY_SCORE_MAX,
                round(
                    current_prediction["display_score"]
                    + 0.3,
                    1
                )
            )

            target_score = st.number_input(
                "목표 학업성과 환산 점수",
                min_value=0.0,
                max_value=DISPLAY_SCORE_MAX,
                value=float(default_target),
                step=0.1
            )

            max_study = st.number_input(
                "하루 최대 공부 시간",
                min_value=float(
                    current["study_hours"]
                ),
                max_value=12.0,
                value=float(
                    min(
                        12.0,
                        current["study_hours"] + 2.0
                    )
                ),
                step=0.5
            )

            min_sleep = st.number_input(
                "반드시 확보할 최소 수면 시간",
                min_value=0.0,
                max_value=12.0,
                value=float(
                    current["sleep_hours"]
                ),
                step=0.5
            )

        with condition_col2:
            max_screen_reduction = st.number_input(
                "줄일 수 있는 최대 스크린타임",
                min_value=0.0,
                max_value=float(
                    current["screen_time"]
                ),
                value=float(
                    min(
                        2.0,
                        current["screen_time"]
                    )
                ),
                step=0.5
            )

            max_attendance_increase = st.number_input(
                "가능한 출석률 최대 증가폭",
                min_value=0.0,
                max_value=float(
                    100.0 - current["attendance"]
                ),
                value=float(
                    min(
                        10.0,
                        100.0 - current["attendance"]
                    )
                ),
                step=1.0
            )

            changeable_labels = st.multiselect(
                "변경 가능한 생활요인",
                options=[
                    "공부 시간",
                    "출석률",
                    "수면 시간",
                    "스크린타임"
                ],
                default=[
                    "공부 시간",
                    "출석률",
                    "수면 시간",
                    "스크린타임"
                ]
            )

        changeable_features = [
            LABEL_TO_FEATURE[label]
            for label in changeable_labels
        ]

        if st.button(
            "추천 계획 찾기",
            type="primary"
        ):
            if not changeable_features:
                st.warning(
                    "변경 가능한 생활요인을 한 개 이상 선택해 주세요."
                )

            elif min_sleep > (
                current["sleep_hours"] + 1.5
            ):
                st.warning(
                    "최소 수면시간을 현재보다 1.5시간 이내로 설정해 주세요."
                )

            else:
                try:
                    with st.spinner(
                        "가능한 생활계획을 탐색하고 있습니다."
                    ):
                        candidates = (
                            generate_recommendation_candidates(
                                current_values=current,
                                target_score=target_score,
                                changeable_features=(
                                    changeable_features
                                ),
                                max_study=max_study,
                                min_sleep=min_sleep,
                                max_screen_reduction=(
                                    max_screen_reduction
                                ),
                                max_attendance_increase=(
                                    max_attendance_increase
                                )
                            )
                        )

                        result = select_recommendations(
                            candidates,
                            target_score
                        )

                        st.session_state.recommendation_result = result

                        st.session_state.recommendation_conditions = {
                            "target_score": target_score,
                            "candidate_count": len(
                                candidates
                            )
                        }

                        st.session_state.selected_recommendation_type = None
                        st.session_state.previous_recommendation_type = None

                        clear_daily_checkboxes()

                        if "recommendation_selector" in st.session_state:
                            del st.session_state[
                                "recommendation_selector"
                            ]

                except Exception as error:
                    st.error(
                        "추천 계획 탐색 중 오류가 발생했습니다."
                    )
                    st.exception(error)

        if st.session_state.recommendation_result is not None:
            result = (
                st.session_state.recommendation_result
            )

            conditions = (
                st.session_state.recommendation_conditions
            )

            recommendations = result[
                "recommendations"
            ]

            st.divider()
            st.header("추천 결과")

            st.caption(
                f"총 {conditions['candidate_count']:,}개 "
                "후보 계획을 평가했습니다."
            )

            if recommendations.empty:
                st.error(
                    "추천안을 생성하지 못했습니다. "
                    "조건 범위를 넓혀 주세요."
                )

            else:
                if result["target_reached"]:
                    st.success(
                        "목표 점수를 달성하는 후보를 찾았습니다."
                    )

                else:
                    st.warning(
                        "목표에는 도달하지 못했지만 "
                        "현재보다 개선되는 후보를 제시합니다."
                    )

                recommendation_types = (
                    recommendations[
                        "추천 유형"
                    ].tolist()
                )

                default_index = 0

                if (
                    st.session_state.selected_recommendation_type
                    in recommendation_types
                ):
                    default_index = recommendation_types.index(
                        st.session_state.selected_recommendation_type
                    )

                selected_type = st.radio(
                    "일주일 계획으로 사용할 추천안을 선택하세요.",
                    options=recommendation_types,
                    index=default_index,
                    horizontal=True,
                    key="recommendation_selector"
                )

                if (
                    st.session_state.previous_recommendation_type
                    != selected_type
                ):
                    clear_daily_checkboxes()

                    st.session_state.previous_recommendation_type = (
                        selected_type
                    )

                st.session_state.selected_recommendation_type = (
                    selected_type
                )

                st.subheader("추천안 카드")

                card_columns = st.columns(
                    len(recommendations)
                )

                for column, (_, row) in zip(
                    card_columns,
                    recommendations.iterrows()
                ):
                    with column:
                        card_html = (
                            build_recommendation_card_html(
                                row=row,
                                current_values=current,
                                selected_type=selected_type
                            )
                        )

                        st.markdown(
                            card_html,
                            unsafe_allow_html=True
                        )

                selected_plan = recommendations[
                    recommendations["추천 유형"]
                    == selected_type
                ].iloc[0]

                st.success(
                    f"선택한 추천안: **{selected_type}**"
                )

                st.header("선택한 계획 요약")

                summary_col1, summary_col2, summary_col3 = (
                    st.columns(3)
                )

                summary_col1.metric(
                    "예상 점수",
                    (
                        f"{selected_plan['예상 환산 점수']:.2f}"
                        f" / {DISPLAY_SCORE_MAX:.2f}"
                    )
                )

                summary_col2.metric(
                    "현재 대비 변화",
                    (
                        f"{selected_plan['현재 대비 점수 변화']:+.2f}"
                    )
                )

                summary_col3.metric(
                    "실행 난이도",
                    (
                        f"{selected_plan['실행 난이도 점수']}"
                        " / 5"
                    )
                )

                st.write(
                    "**생활 변화:** "
                    + build_change_summary(
                        current,
                        selected_plan
                    )
                )

                st.write(
                    "**추천 이유:** "
                    + selected_plan["추천 이유"]
                )

                difficulty_info = get_difficulty_message(
                    int(
                        selected_plan[
                            "실행 난이도 점수"
                        ]
                    )
                )

                st.markdown(
                    (
                        f'<div class="{difficulty_info["class_name"]}">'
                        f'<b>{escape(difficulty_info["title"])}</b><br>'
                        f'{escape(difficulty_info["message"])}'
                        f'</div>'
                    ),
                    unsafe_allow_html=True
                )

                # -----------------------------------------
                # 3주 단계적 실행계획
                # -----------------------------------------
                st.divider()
                st.header("3주 단계적 실행계획")

                three_week_plan = generate_three_week_plan(
                    current,
                    selected_plan
                )

                week_columns = st.columns(3)

                for week_column, (_, week) in zip(
                    week_columns,
                    three_week_plan.iterrows()
                ):
                    with week_column:
                        week_html = (
                            '<div class="step-card">'
                            f'<h3>{escape(str(week["주차"]))}</h3>'
                            f'<p><b>공부:</b> '
                            f'{week["공부 시간"]:.1f}시간</p>'
                            f'<p><b>출석률:</b> '
                            f'{week["출석률"]:.1f}%</p>'
                            f'<p><b>수면:</b> '
                            f'{week["수면 시간"]:.1f}시간</p>'
                            f'<p><b>스크린타임:</b> '
                            f'{week["스크린타임"]:.1f}시간</p>'
                            f'<p><b>핵심:</b> '
                            f'{escape(str(week["핵심 목표"]))}</p>'
                            '</div>'
                        )

                        st.markdown(
                            week_html,
                            unsafe_allow_html=True
                        )

                # -----------------------------------------
                # 일주일 실천계획
                # -----------------------------------------
                weekly_plan = generate_simple_action_plan(
                    current,
                    selected_plan
                )

                st.divider()
                st.header("일주일 실천계획")

                weekday_col, weekend_col = st.columns(2)

                with weekday_col:
                    st.subheader("평일 핵심 행동")

                    for action in weekly_plan[
                        "weekday_actions"
                    ]:
                        st.write(f"- {action}")

                with weekend_col:
                    st.subheader("주말 핵심 행동")

                    for action in weekly_plan[
                        "weekend_actions"
                    ]:
                        st.write(f"- {action}")

                goal_col, check_col = st.columns(2)

                with goal_col:
                    st.subheader("이번 주 목표")

                    for goal in weekly_plan[
                        "core_goals"
                    ]:
                        st.write(f"✅ {goal}")

                with check_col:
                    st.subheader("매일 확인")

                    for item in weekly_plan[
                        "check_items"
                    ]:
                        st.write(f"□ {item}")

                st.subheader("요일별 실천표")

                st.dataframe(
                    weekly_plan["weekly_df"],
                    use_container_width=True,
                    hide_index=True
                )

                # -----------------------------------------
                # 실천 체크 및 진행률
                # -----------------------------------------
                st.divider()
                st.header("이번 주 실천 현황")

                st.write(
                    "실천을 완료한 요일을 체크하세요."
                )

                day_columns = st.columns(7)

                completed_days = 0

                for day_column, day in zip(
                    day_columns,
                    DAYS
                ):
                    checkbox_key = (
                        f"daily_check_"
                        f"{selected_type}_"
                        f"{day}"
                    )

                    with day_column:
                        checked = st.checkbox(
                            day.replace(
                                "요일",
                                ""
                            ),
                            key=checkbox_key
                        )

                    if checked:
                        completed_days += 1

                progress_ratio = (
                    completed_days
                    / len(DAYS)
                )

                progress_percent = int(
                    round(
                        progress_ratio * 100
                    )
                )

                st.progress(
                    progress_ratio
                )

                st.markdown(
                    (
                        '<div class="progress-summary">'
                        f'{completed_days}일 / 7일 완료 '
                        f'· 진행률 {progress_percent}%'
                        '</div>'
                    ),
                    unsafe_allow_html=True
                )

                if completed_days == 0:
                    st.info(
                        "첫 실천일을 완료하면 체크해 보세요."
                    )

                elif completed_days < 4:
                    st.info(
                        "좋은 시작입니다. 작은 실천을 계속 이어가세요."
                    )

                elif completed_days < 7:
                    st.success(
                        "절반 이상 완료했습니다. 목표까지 조금 남았습니다."
                    )

                else:
                    st.success(
                        "이번 주 계획을 모두 완료했습니다! 🎉"
                    )

                if st.button(
                    "이번 주 체크 초기화"
                ):
                    clear_daily_checkboxes()
                    st.rerun()

                # -----------------------------------------
                # 전체 추천 결과
                # -----------------------------------------
                st.divider()

                with st.expander(
                    "전체 추천 계획 비교 보기"
                ):
                    display_recommendations = (
                        recommendations[
                            [
                                "추천 유형",
                                "공부 시간",
                                "출석률",
                                "수면 시간",
                                "스크린타임",
                                "예상 환산 점수",
                                "현재 대비 점수 변화",
                                "변화 부담",
                                "실행 난이도",
                                "추천 이유"
                            ]
                        ].copy()
                    )

                    st.dataframe(
                        display_recommendations,
                        use_container_width=True,
                        hide_index=True
                    )

                    comparison = (
                        make_recommendation_comparison(
                            current,
                            recommendations
                        )
                    )

                    st.dataframe(
                        comparison,
                        use_container_width=True,
                        hide_index=True
                    )

                # -----------------------------------------
                # 엑셀 다운로드
                # -----------------------------------------
                comparison = (
                    make_recommendation_comparison(
                        current,
                        recommendations
                    )
                )

                all_xlsx_data = recommendations_to_xlsx(
                    recommendations=recommendations,
                    comparison_df=comparison,
                    current_values=current,
                    target_score=conditions[
                        "target_score"
                    ]
                )

                selected_xlsx_data = (
                    selected_plan_to_xlsx(
                        selected_plan=selected_plan,
                        weekly_plan=weekly_plan,
                        three_week_plan=three_week_plan,
                        current_values=current
                    )
                )

                download_col1, download_col2 = (
                    st.columns(2)
                )

                with download_col1:
                    st.download_button(
                        "전체 추천안 엑셀 다운로드",
                        data=all_xlsx_data,
                        file_name=(
                            "campus_twin_all_recommendations.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        )
                    )

                with download_col2:
                    st.download_button(
                        "선택한 계획 엑셀 다운로드",
                        data=selected_xlsx_data,
                        file_name=(
                            "campus_twin_selected_plan.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        )
                    )

                st.caption(
                    "체크 진행률은 현재 브라우저 세션에서만 유지됩니다."
                )


# =========================================================
# 탭 5: 프로젝트 설명
# =========================================================
with tab5:
    st.header("프로젝트 설명")

    st.subheader("프로젝트 목표")

    st.write("""
    Campus Twin은 대학생의 생활습관을 입력받아
    학업성과를 예측하고,
    미래 생활계획과 목표 기반 추천안을 제공하는 웹서비스입니다.

    사용자는 추천안을 선택하고
    3주 단계계획과 일주일 실천표를 확인할 수 있습니다.
    """)

    st.subheader("사용 변수")

    feature_info = pd.DataFrame({
        "변수명": FEATURE_COLS,
        "설명": [
            "하루 공부 시간",
            "출석률",
            "하루 수면 시간",
            "하루 스크린타임",
            "하루 신체활동 시간",
            "스트레스 수준"
        ]
    })

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("사용 모델")

    st.write("""
    - Linear Regression
    - StandardScaler 기반 입력 표준화
    """)

    st.subheader("현재 기능")

    st.write("""
    - 생활습관 기반 학업성과 예측
    - 개인별 민감도 분석
    - 미래 계획 시뮬레이션
    - 목표 기반 추천안 자동 탐색
    - 최소 변화형·균형형·성과 우선형 추천
    - 추천안 선택 및 카드 강조
    - 실행 난이도 안내
    - 3주 단계적 실행계획
    - 일주일 핵심 실천계획
    - 요일별 체크박스
    - 7일 기준 진행률 표시
    - 추천 계획 엑셀 다운로드
    """)

    st.subheader("한계 및 주의사항")

    st.warning("""
    학업성과 환산 점수는 실제 대학의 GPA가 아닙니다.

    본 서비스는 학습 데이터의 통계적 패턴을 이용한
    모의 분석 및 계획 지원 시스템이며,
    실제 성적이나 학업 결과를 보장하지 않습니다.

    체크박스 진행률은 현재 접속 세션에서만 유지됩니다.
    """)
