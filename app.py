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
# 모델 점수 및 화면 표시 설정
# =========================================================
MODEL_TARGET_MIN = 0.0
MODEL_TARGET_MAX = 2.009058
DISPLAY_SCORE_MAX = 4.5

EPSILON = 1e-8


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
# 공통 함수
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

    display_score = convert_to_display_score(raw_score)

    return {
        "raw_score": raw_score,
        "display_score": display_score
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


def is_same_plan(
    first_plan,
    second_plan
):
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

        candidate_changed = has_any_change(
            current_values,
            candidate
        )

        score_improved = (
            prediction["display_score"]
            > current_prediction["display_score"]
            + EPSILON
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
            "목표 달성": (
                prediction["display_score"]
                >= target_score
            ),
            "현재와 다른 계획": candidate_changed,
            "현재보다 개선": score_improved,
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
            selection_pool = improved_candidates
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
            (
                "설정한 조건 안에서 현재 생활과의 차이가 "
                "가장 작은 계획입니다."
            )
        ),
        (
            balanced_label,
            balanced_row,
            (
                "예상 성과와 생활 변화 부담을 함께 "
                "고려한 절충 계획입니다."
            )
        ),
        (
            performance_label,
            performance_row,
            (
                "설정한 조건 안에서 가장 높은 "
                "학업성과 점수를 예측한 계획입니다."
            )
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
            "주요 생활시간 합계": (
                row["주요 생활시간 합계"]
            ),
            "추천 이유": reason
        })

    recommendations = pd.DataFrame(
        recommendation_rows
    )

    return {
        "target_reached": target_reached,
        "recommendations": recommendations,
        "candidate_pool_type": candidate_pool_type
    }


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
            "변화 부담": (
                recommendation["변화 부담"]
            )
        })

    return pd.DataFrame(rows)


def build_change_summary(
    current_values,
    recommendation
):
    changes = []

    change_items = [
        ("공부 시간", "study_hours", "공부 시간", "시간"),
        ("출석률", "attendance", "출석률", "%"),
        ("수면 시간", "sleep_hours", "수면 시간", "시간"),
        ("스크린타임", "screen_time", "스크린타임", "시간")
    ]

    recommendation_columns = {
        "study_hours": "공부 시간",
        "attendance": "출석률",
        "sleep_hours": "수면 시간",
        "screen_time": "스크린타임"
    }

    for (
        label,
        current_key,
        _,
        unit
    ) in change_items:

        recommendation_column = (
            recommendation_columns[current_key]
        )

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

        if difference > 0:
            direction = "증가"
        else:
            direction = "감소"

        if current_key == "attendance":
            changes.append(
                f"{label} {abs(difference):.0f}%p {direction}"
            )

        else:
            changes.append(
                f"{label} {abs(difference):.1f}{unit} {direction}"
            )

    if not changes:
        return "현재 생활과 동일한 계획입니다."

    return ", ".join(changes)


def recommendations_to_xlsx(
    recommendations,
    comparison_df,
    current_values,
    target_score
):
    output = BytesIO()

    export_recommendations = (
        recommendations.copy()
    )

    export_comparison = (
        comparison_df.copy()
    )

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

        export_recommendations.to_excel(
            writer,
            sheet_name="추천 계획",
            index=False
        )

        export_comparison.to_excel(
            writer,
            sheet_name="현재 대비 변화",
            index=False
        )

        current_df.to_excel(
            writer,
            sheet_name="입력 조건",
            index=False
        )

        workbook = writer.book

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
                    vertical="center"
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
                    35
                )

            for row in worksheet.iter_rows(
                min_row=2
            ):
                for cell in row:
                    if isinstance(
                        cell.value,
                        float
                    ):
                        cell.number_format = "0.00"

    output.seek(0)

    return output.getvalue()


# =========================================================
# 세션 상태 초기화
# =========================================================
DEFAULT_SESSION_VALUES = {
    "current_values": None,
    "prediction_result": None,
    "sensitivity_df": None,
    "comparison_result": None,
    "recommendation_result": None,
    "recommendation_conditions": None
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
현재 생활습관을 바탕으로 학업성과를 분석하고,
생활계획을 변경했을 때 예측 결과가 어떻게 달라지는지
가상으로 비교할 수 있습니다.

목표 학업성과 점수와 현실적인 생활조건을 설정하면
여러 후보 계획을 탐색하여 맞춤형 추천안을 제공합니다.
""")

st.warning("""
표시되는 학업성과 환산 점수는 실제 학교 성적이나
공식 GPA가 아닙니다.

현재 모델의 원본 예측 점수를 4.5점 범위로 변환한
자기관리 참고용 지표입니다.
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
            "공부·수면·스크린타임·신체활동의 합이 "
            "하루 24시간을 초과합니다. "
            "입력값을 다시 확인해 주세요."
        )

    if st.button(
        "현재 상태 분석하기",
        type="primary"
    ):
        try:
            prediction_result = predict_from_values(
                current_values
            )

            sensitivity_df = analyze_sensitivity(
                current_values
            )

            st.session_state.current_values = (
                current_values.copy()
            )

            st.session_state.prediction_result = (
                prediction_result
            )

            st.session_state.sensitivity_df = (
                sensitivity_df
            )

            st.session_state.comparison_result = None
            st.session_state.recommendation_result = None
            st.session_state.recommendation_conditions = None

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

        st.info("""
        선형회귀 모델에는 학습 당시와 동일하게
        StandardScaler로 변환한 입력값을 전달합니다.

        모델 원본 점수는 사용자 이해를 돕기 위해
        4.5점 범위의 학업성과 환산 점수로 변환됩니다.
        """)

        st.caption(
            "예측 결과는 실제 성적이나 향후 학업 결과를 "
            "확정하거나 보장하지 않습니다."
        )


# =========================================================
# 탭 2: 개인별 분석
# =========================================================
with tab2:
    st.header("개인별 민감도 분석")

    st.write("""
    현재 입력값에서 생활요인을 하나씩 변경했을 때
    학업성과 환산 점수가 얼마나 달라지는지 계산합니다.
    """)

    if (
        st.session_state.current_values is None
        or st.session_state.sensitivity_df is None
    ):
        st.info(
            "'현재 상태' 탭에서 생활습관을 입력한 뒤 "
            "'현재 상태 분석하기' 버튼을 눌러 주세요."
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

        st.subheader(
            "생활요인별 학업성과 변화량"
        )

        chart_df = sensitivity_df[
            [
                "생활요인 변화",
                "환산 점수 변화량"
            ]
        ].copy()

        chart_df = chart_df.set_index(
            "생활요인 변화"
        )

        st.bar_chart(
            chart_df["환산 점수 변화량"],
            use_container_width=True
        )

        best_result = sensitivity_df.iloc[0]

        if best_result["환산 점수 변화량"] > 0:
            st.success(
                f"현재 입력에서는 "
                f"'{best_result['생활요인 변화']}' 시나리오가 "
                f"가장 큰 예측 상승을 보였습니다. "
                f"학업성과 예상 변화량은 "
                f"{best_result['환산 점수 변화량']:+.3f}입니다."
            )

        else:
            st.warning(
                "현재 설정된 변화 시나리오에서는 "
                "학업성과 점수가 상승하는 항목이 "
                "확인되지 않았습니다."
            )

        st.caption(
            "민감도 분석은 모델 예측값의 반응을 "
            "보여주는 것이며 실제 인과관계를 의미하지 않습니다."
        )


# =========================================================
# 탭 3: 미래 시뮬레이션
# =========================================================
with tab3:
    st.header("미래 생활계획 시뮬레이션")

    st.write("""
    현재 생활습관과 앞으로 실천할 계획을 비교하여
    학업성과 환산 점수가 어떻게 달라지는지 확인합니다.
    """)

    if st.session_state.current_values is None:
        st.info(
            "'현재 상태' 탭에서 먼저 "
            "생활습관을 분석해 주세요."
        )

    else:
        current = st.session_state.current_values

        st.subheader("미래 계획 입력")

        future_col1, future_col2 = st.columns(2)

        with future_col1:
            future_study = st.number_input(
                "계획 공부 시간",
                min_value=0.0,
                max_value=12.0,
                value=float(current["study_hours"]),
                step=0.5,
                key="future_study"
            )

            future_attendance = st.number_input(
                "계획 출석률",
                min_value=0.0,
                max_value=100.0,
                value=float(current["attendance"]),
                step=1.0,
                key="future_attendance"
            )

            future_sleep = st.number_input(
                "계획 수면 시간",
                min_value=0.0,
                max_value=12.0,
                value=float(current["sleep_hours"]),
                step=0.5,
                key="future_sleep"
            )

        with future_col2:
            future_screen = st.number_input(
                "계획 스크린타임",
                min_value=0.0,
                max_value=16.0,
                value=float(current["screen_time"]),
                step=0.5,
                key="future_screen"
            )

            future_activity = st.number_input(
                "계획 신체활동 시간",
                min_value=0.0,
                max_value=8.0,
                value=float(
                    current["physical_activity"]
                ),
                step=0.5,
                key="future_activity"
            )

            future_stress = st.slider(
                "계획 스트레스 수준",
                min_value=1.0,
                max_value=10.0,
                value=float(current["stress"]),
                step=0.5,
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

        future_total_hours = calculate_total_hours(
            future_values
        )

        st.caption(
            f"미래 계획의 주요 생활시간 합계: "
            f"{future_total_hours:.1f}시간"
        )

        if future_total_hours > 24:
            st.warning(
                "미래 계획의 주요 생활시간 합이 "
                "하루 24시간을 초과합니다. "
                "입력값을 다시 확인해 주세요."
            )

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
            st.subheader("시뮬레이션 결과")

            metric_col1, metric_col2, metric_col3 = (
                st.columns(3)
            )

            with metric_col1:
                st.metric(
                    "현재 학업성과 점수",
                    (
                        f"{comparison_result['current_display_score']:.2f}"
                        f" / {DISPLAY_SCORE_MAX:.2f}"
                    )
                )

            with metric_col2:
                st.metric(
                    "미래 학업성과 점수",
                    (
                        f"{comparison_result['future_display_score']:.2f}"
                        f" / {DISPLAY_SCORE_MAX:.2f}"
                    )
                )

            with metric_col3:
                st.metric(
                    "예상 변화",
                    (
                        f"{comparison_result['display_score_change']:+.2f}"
                    )
                )

            comparison_df = (
                comparison_result["comparison_df"].copy()
            )

            display_comparison_df = (
                comparison_df.copy()
            )

            display_comparison_df["현재"] = (
                display_comparison_df["현재"]
                .map(lambda value: f"{value:.1f}")
            )

            display_comparison_df["미래 계획"] = (
                display_comparison_df["미래 계획"]
                .map(lambda value: f"{value:.1f}")
            )

            display_comparison_df["변화"] = (
                display_comparison_df["변화"]
                .map(lambda value: f"{value:+.1f}")
            )

            st.dataframe(
                display_comparison_df,
                use_container_width=True,
                hide_index=True
            )

            score_change = comparison_result[
                "display_score_change"
            ]

            if score_change > 0:
                st.success(
                    "현재 계획보다 미래 계획의 "
                    "학업성과 환산 점수가 높게 나타났습니다."
                )

            elif score_change < 0:
                st.warning(
                    "미래 계획의 학업성과 환산 점수가 "
                    "현재 계획보다 낮게 나타났습니다."
                )

            else:
                st.info(
                    "현재 계획과 미래 계획의 "
                    "학업성과 환산 점수가 동일합니다."
                )

            with st.expander(
                "원본 모델 점수 확인"
            ):
                st.write(
                    "현재 원본 점수:",
                    f"{comparison_result['current_raw_score']:.4f}"
                )

                st.write(
                    "미래 원본 점수:",
                    f"{comparison_result['future_raw_score']:.4f}"
                )


# =========================================================
# 탭 4: 추천 계획
# =========================================================
with tab4:
    st.header("목표 기반 추천 계획")

    st.write("""
    목표 학업성과 점수와 생활조건을 설정하면
    가능한 생활계획 조합을 자동으로 탐색합니다.
    """)

    if st.session_state.current_values is None:
        st.info(
            "'현재 상태' 탭에서 먼저 "
            "현재 생활습관을 분석해 주세요."
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
                    100.0
                    - current["attendance"]
                ),
                value=float(
                    min(
                        10.0,
                        100.0
                        - current["attendance"]
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

        st.caption(
            "추천 탐색은 공부 시간 최대 2시간 증가, "
            "수면 시간 최대 1.5시간 증가 범위 안에서 진행됩니다."
        )

        if st.button(
            "추천 계획 찾기",
            type="primary"
        ):
            if not changeable_features:
                st.warning(
                    "변경 가능한 생활요인을 "
                    "한 개 이상 선택해 주세요."
                )

            elif min_sleep > (
                current["sleep_hours"] + 1.5
            ):
                st.warning(
                    "최소 수면 시간이 현재 수면보다 "
                    "1.5시간을 초과하여 높습니다. "
                    "현실적인 범위로 조정해 주세요."
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

                        recommendation_result = (
                            select_recommendations(
                                candidates,
                                target_score
                            )
                        )

                        st.session_state.recommendation_result = (
                            recommendation_result
                        )

                        st.session_state.recommendation_conditions = {
                            "target_score": target_score,
                            "candidate_count": len(
                                candidates
                            ),
                            "changeable_labels": (
                                changeable_labels
                            )
                        }

                except Exception as error:
                    st.error(
                        "추천 계획 탐색 중 오류가 발생했습니다."
                    )
                    st.exception(error)

        if (
            st.session_state.recommendation_result
            is not None
        ):
            recommendation_result = (
                st.session_state.recommendation_result
            )

            recommendation_conditions = (
                st.session_state.recommendation_conditions
            )

            recommendations = (
                recommendation_result["recommendations"]
            )

            st.divider()
            st.subheader("추천 결과")

            st.caption(
                f"총 "
                f"{recommendation_conditions['candidate_count']:,}개 "
                "후보 계획을 평가했습니다."
            )

            if recommendations.empty:
                st.error(
                    "현재 설정으로는 서로 다른 추천 계획을 "
                    "생성하지 못했습니다. "
                    "변경 가능 항목이나 조건 범위를 넓혀 주세요."
                )

            else:
                if recommendation_result["target_reached"]:
                    st.success(
                        "목표 점수를 달성하는 후보 계획을 "
                        "찾았습니다."
                    )

                else:
                    st.warning(
                        "설정한 조건에서는 목표 점수에 "
                        "도달하지 못했습니다. "
                        "현재 상태보다 개선되는 후보 중 "
                        "가장 적합한 계획을 제시합니다."
                    )

                st.caption(
                    "추천 기준: "
                    f"{recommendation_result['candidate_pool_type']}"
                )

                card_columns = st.columns(
                    len(recommendations)
                )

                for column, (_, row) in zip(
                    card_columns,
                    recommendations.iterrows()
                ):
                    with column:
                        st.subheader(
                            row["추천 유형"]
                        )

                        st.metric(
                            "예상 학업성과 점수",
                            (
                                f"{row['예상 환산 점수']:.2f}"
                                f" / {DISPLAY_SCORE_MAX:.2f}"
                            ),
                            (
                                f"{row['현재 대비 점수 변화']:+.2f}"
                            )
                        )

                        st.write(
                            f"**공부 시간:** "
                            f"{row['공부 시간']:.1f}시간"
                        )

                        st.write(
                            f"**출석률:** "
                            f"{row['출석률']:.0f}%"
                        )

                        st.write(
                            f"**수면 시간:** "
                            f"{row['수면 시간']:.1f}시간"
                        )

                        st.write(
                            f"**스크린타임:** "
                            f"{row['스크린타임']:.1f}시간"
                        )

                        st.write(
                            f"**변화 부담:** "
                            f"{row['변화 부담']}"
                        )

                        st.info(
                            row["추천 이유"]
                        )

                        st.caption(
                            build_change_summary(
                                current,
                                row
                            )
                        )

                st.subheader("추천 계획 비교")

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
                            "주요 생활시간 합계",
                            "추천 이유"
                        ]
                    ].copy()
                )

                numeric_columns = [
                    "공부 시간",
                    "출석률",
                    "수면 시간",
                    "스크린타임",
                    "예상 환산 점수",
                    "현재 대비 점수 변화",
                    "주요 생활시간 합계"
                ]

                for column in numeric_columns:
                    display_recommendations[
                        column
                    ] = (
                        display_recommendations[
                            column
                        ]
                        .map(
                            lambda value: (
                                f"{value:.2f}"
                            )
                        )
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

                st.subheader(
                    "현재 생활 대비 추천 변화"
                )

                display_comparison = (
                    comparison.copy()
                )

                for column in [
                    "공부 시간 변화",
                    "출석률 변화",
                    "수면 시간 변화",
                    "스크린타임 변화",
                    "현재 대비 점수 변화"
                ]:
                    display_comparison[column] = (
                        display_comparison[column]
                        .map(
                            lambda value: (
                                f"{value:+.2f}"
                            )
                        )
                    )

                display_comparison[
                    "예상 환산 점수"
                ] = (
                    display_comparison[
                        "예상 환산 점수"
                    ]
                    .map(
                        lambda value: f"{value:.2f}"
                    )
                )

                st.dataframe(
                    display_comparison,
                    use_container_width=True,
                    hide_index=True
                )

                xlsx_data = recommendations_to_xlsx(
                    recommendations=(
                        recommendations
                    ),
                    comparison_df=comparison,
                    current_values=current,
                    target_score=(
                        recommendation_conditions[
                            "target_score"
                        ]
                    )
                )

                st.download_button(
                    label="추천 계획 엑셀 다운로드",
                    data=xlsx_data,
                    file_name=(
                        "campus_twin_recommendations.xlsx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.spreadsheetml.sheet"
                    )
                )

                st.caption(
                    "추천 결과는 모델이 학습한 통계적 패턴과 "
                    "사용자가 입력한 조건에 기반한 참고용 계획입니다."
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
    미래 생활계획을 시뮬레이션하며,
    목표와 현실적인 생활조건에 맞는 추천 계획을
    탐색하는 웹서비스입니다.
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

    st.subheader("추천 계획 생성 방식")

    st.write("""
    사용자가 설정한 목표 점수와 생활조건 안에서
    가능한 생활계획 조합을 생성합니다.

    각 후보 계획을 예측모델에 입력하고
    학업성과 환산 점수와 현재 생활 대비 변화 부담을 계산합니다.

    목표 달성 여부에 따라 최소 변화형, 균형형,
    성과 우선형 또는 개선형 추천안을 제공합니다.
    """)

    recommendation_info = pd.DataFrame({
        "추천 유형": [
            "최소 변화형",
            "균형형",
            "성과 우선형",
            "최소 변화 개선형",
            "균형 개선형",
            "최고 성과형"
        ],
        "적용 상황": [
            "목표 달성 가능",
            "목표 달성 가능",
            "목표 달성 가능",
            "목표 달성 불가능",
            "목표 달성 불가능",
            "목표 달성 불가능"
        ],
        "선정 기준": [
            "목표를 달성하면서 현재 생활과 가장 비슷한 계획",
            "성과와 생활 변화 부담을 함께 고려한 계획",
            "목표 달성 후보 중 예상 점수가 가장 높은 계획",
            "현재보다 개선되면서 변화가 가장 적은 계획",
            "개선 효과와 변화 부담을 함께 고려한 계획",
            "현실적 조건 안에서 가장 높은 점수의 계획"
        ]
    })

    st.dataframe(
        recommendation_info,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("점수 표시 방식")

    st.write(f"""
    모델의 원본 예측 점수는 사용자 이해를 돕기 위해
    {DISPLAY_SCORE_MAX:.1f}점 범위의
    학업성과 환산 점수로 변환합니다.
    """)

    st.code(
        """
학업성과 환산 점수
= 원본 예측 점수
÷ 원본 데이터 최대 점수
× 4.5
        """.strip(),
        language="text"
    )

    st.subheader("현재 기능")

    st.write("""
    - 생활습관 기반 학업성과 예측
    - 개인별 민감도 분석
    - 생활요인별 예측 변화량 시각화
    - 현재 생활과 미래 생활계획 비교
    - 목표와 제약조건 기반 추천 계획 탐색
    - 목표 달성 여부에 따른 추천 유형 변경
    - 중복되지 않는 추천안 제공
    - 추천 이유 및 변화 내용 설명
    - 추천 결과 엑셀 다운로드
    - 비현실적인 시간 입력 경고
    """)

    st.subheader("한계 및 주의사항")

    st.warning("""
    학업성과 환산 점수는 실제 대학의 GPA나
    공식 학점 산정 결과가 아닙니다.

    본 서비스는 학습 데이터의 통계적 패턴을 이용한
    모의 분석 및 계획 지원 시스템입니다.

    생활습관과 학업성과 사이의 인과관계를 증명하지 않으며,
    실제 성적이나 학업 결과를 보장하지 않습니다.
    """)
