import pandas as pd
import numpy as np


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    columns,
    prefix=None,
    keywords=None
):

    if keywords is None:
        keywords = []

    keywords = [
        str(k).lower()
        for k in keywords
    ]

    # Exact keyword combination
    for column in columns:

        column_lower = str(column).lower()

        if prefix is not None:

            if not column_lower.startswith(
                prefix.lower()
            ):
                continue

        if all(
            keyword in column_lower
            for keyword in keywords
        ):

            return column

    # Fallback
    for column in columns:

        column_lower = str(column).lower()

        if prefix is not None:

            if not column_lower.startswith(
                prefix.lower()
            ):
                continue

        if any(
            keyword in column_lower
            for keyword in keywords
        ):

            return column

    return None


# ============================================================
# SAFE VALUE
# ============================================================

def safe_value(row, column):

    if column is None:
        return None

    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    if isinstance(
        value,
        (np.integer, np.floating)
    ):

        return float(value)

    return value


# ============================================================
# OBJECT INFORMATION
# ============================================================

def get_object_information(row):

    columns = row.index

    target = {

        "semi_major_axis": safe_value(
            row,
            find_column(
                columns,
                "t_",
                ["sma"]
            )
        ),

        "eccentricity": safe_value(
            row,
            find_column(
                columns,
                "t_",
                ["ecc"]
            )
        ),

        "inclination": safe_value(
            row,
            find_column(
                columns,
                "t_",
                ["inc"]
            )
        ),

        "apogee": safe_value(
            row,
            find_column(
                columns,
                "t_",
                ["apogee"]
            )
        ),

        "perigee": safe_value(
            row,
            find_column(
                columns,
                "t_",
                ["perigee"]
            )
        ),

        "rcs": safe_value(
            row,
            find_column(
                columns,
                "t_",
                ["rcs"]
            )
        )
    }


    chaser = {

        "semi_major_axis": safe_value(
            row,
            find_column(
                columns,
                "c_",
                ["sma"]
            )
        ),

        "eccentricity": safe_value(
            row,
            find_column(
                columns,
                "c_",
                ["ecc"]
            )
        ),

        "inclination": safe_value(
            row,
            find_column(
                columns,
                "c_",
                ["inc"]
            )
        ),

        "apogee": safe_value(
            row,
            find_column(
                columns,
                "c_",
                ["apogee"]
            )
        ),

        "perigee": safe_value(
            row,
            find_column(
                columns,
                "c_",
                ["perigee"]
            )
        ),

        "rcs": safe_value(
            row,
            find_column(
                columns,
                "c_",
                ["rcs"]
            )
        )
    }


    return {
        "target": target,
        "chaser": chaser
    }


# ============================================================
# CLOSE APPROACH INFORMATION
# ============================================================

def get_close_approach(row):

    columns = row.index

    miss_distance_column = find_column(
        columns,
        None,
        ["miss_distance"]
    )

    relative_speed_column = find_column(
        columns,
        None,
        ["relative_speed"]
    )

    position_r = find_column(
        columns,
        None,
        ["relative_position_r"]
    )

    position_t = find_column(
        columns,
        None,
        ["relative_position_t"]
    )

    position_n = find_column(
        columns,
        None,
        ["relative_position_n"]
    )

    velocity_r = find_column(
        columns,
        None,
        ["relative_velocity_r"]
    )

    velocity_t = find_column(
        columns,
        None,
        ["relative_velocity_t"]
    )

    velocity_n = find_column(
        columns,
        None,
        ["relative_velocity_n"]
    )


    return {

        "time_to_tca": safe_value(
            row,
            "time_to_tca"
        ),

        "miss_distance": safe_value(
            row,
            miss_distance_column
        ),

        "relative_speed": safe_value(
            row,
            relative_speed_column
        ),

        "relative_position": {

            "r": safe_value(
                row,
                position_r
            ),

            "t": safe_value(
                row,
                position_t
            ),

            "n": safe_value(
                row,
                position_n
            )
        },

        "relative_velocity": {

            "r": safe_value(
                row,
                velocity_r
            ),

            "t": safe_value(
                row,
                velocity_t
            ),

            "n": safe_value(
                row,
                velocity_n
            )
        }
    }


# ============================================================
# EVENT DATA
# ============================================================

def get_event_data(df, event_id):

    event_df = df[
        df["event_id"].astype(str)
        == str(event_id)
    ].copy()

    if event_df.empty:

        return None

    event_df["time_to_tca"] = pd.to_numeric(
        event_df["time_to_tca"],
        errors="coerce"
    )

    event_df = (
        event_df
        .sort_values(
            "time_to_tca",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # Latest row chronologically
    latest_row = event_df.iloc[-1]

    return {

        "event_id": str(event_id),

        "number_of_cdms": len(event_df),

        "objects": get_object_information(
            latest_row
        ),

        "close_approach": get_close_approach(
            latest_row
        ),

        "timeline": (
            event_df[
                [
                    column
                    for column in [
                        "time_to_tca",
                        find_column(
                            event_df.columns,
                            None,
                            ["miss_distance"]
                        ),
                        find_column(
                            event_df.columns,
                            None,
                            ["relative_speed"]
                        ),
                        find_column(
                            event_df.columns,
                            None,
                            ["risk"]
                        )
                    ]
                    if column is not None
                    and column in event_df.columns
                ]
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .where(
                pd.notnull(
                    event_df
                ),
                None
            )
            .to_dict(
                orient="records"
            )
        )
    }
    