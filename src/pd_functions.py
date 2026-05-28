import numpy as np
import pandas as pd
import streamlit as st

from src.utils import get_global_store


def get_ready_test(results_path: str, uploaded_file) -> pd.DataFrame:
    """This function prepares the test file to be evaluated and
    check if it has the correct format.
    """
    results = pd.read_csv(results_path)
    results.columns = ["id", "real"]

    test = pd.read_csv(uploaded_file)
    if test.columns.to_list() != ["Id", "poisonous"]:
        st.error('Column names must match "Id" and "poisonous" - case sensitive!')
        return 0
    if test.shape != (1625, 2):
        st.error("Your file should contain 1625 rows and 2 columns")
        return 0
    if (
        (test.poisonous.unique().tolist() != [0, 1])
        & (test.poisonous.unique().tolist() != [1, 0])
        & (test.poisonous.unique().tolist() != [1])
        & (test.poisonous.unique().tolist() != [0])
    ):
        st.error("Predictions should only have values of 0 and 1")
        return 0
    if (test.Id == results.id).sum() != 1625:
        st.error(
            "Your Id column might be wrong or mixed up. "
            "You should have same Id's as the test file. "
            "Order of Id's should also be the same.",
        )
        return 0
    test.columns = ["id", "preds"]

    return test.astype("int32")


def get_metrics(results_path: str, test: pd.DataFrame) -> pd.DataFrame:
    """Calculates metrics and prepares a single-row DataFrame for GSheets submission."""
    results = pd.read_csv(results_path)
    results.columns = ["id", "real"]

    row_evaluation = (
        results.astype("int32")
        .merge(test, how="left", on="id")
        .assign(
            tp=lambda df_: np.where(
                (df_["real"] == 1) & (df_["preds"] == 1),
                True,
                False,
            ),
            correct=lambda df_: df_["real"] == df_["preds"],
            fn=lambda df_: np.where(
                (df_["real"] == 1) & (df_["preds"] == 0),
                True,
                False,
            ),
            opportunity_cost=lambda df_: np.where(
                (df_["real"] == 0) & (df_["preds"] == 1),
                True,
                False,
            ),
        )
        .agg(
            {
                "tp": "sum",
                "correct": "sum",
                "fn": "sum",
                "opportunity_cost": "sum",
            },
        )
    )

    return pd.DataFrame(
        [
            {
                "Participant": st.session_state.user_name,
                "Scoring metric": round(
                    row_evaluation["tp"]
                    / (row_evaluation["tp"] + row_evaluation["fn"])
                    * 0.95
                    + row_evaluation["correct"] / results.shape[0] * 0.05,
                    4,
                ),
                "Recall": round(
                    row_evaluation["tp"]
                    / (row_evaluation["tp"] + row_evaluation["fn"]),
                    4,
                ),
                "Accuracy": round(row_evaluation["correct"] / results.shape[0], 4),
                "Hospitalized": int(row_evaluation["fn"]),
                "Edible but uneaten": int(row_evaluation["opportunity_cost"]),
                "submission_time": pd.Timestamp.now().isoformat(),
                "batch": st.session_state.batch,
            },
        ],
    )


def update_submissions(participant_results: pd.DataFrame) -> None:
    """Writes results to GSheets and updates memory store to avoid re-reading."""
    store = get_global_store()
    batch = st.session_state.batch

    current_submissions = store.setdefault("submissions", {}).get(batch, pd.DataFrame())
    updated_df = pd.concat(
        [current_submissions, participant_results],
        ignore_index=True,
    )
    store["submissions"][batch] = updated_df

    try:
        store["gsheet_conn"].update(worksheet=batch, data=updated_df)

        if batch != "anonymous" and store.get("alltime_submissions") is not None:
            store["alltime_submissions"] = pd.concat(
                [store["alltime_submissions"], participant_results],
                ignore_index=True,
            )
    except Exception as e:
        st.error(f"Sync failed: {e}")
