import os
import pandas as pd

from datetime import datetime, timedelta


class ForecastEngine:

    def __init__(self):

        # -----------------------------------
        # Forecast file path
        # -----------------------------------
        self.file_path = (

            "data/ml_data/"
            "predictions_fail.csv"
        )

        # -----------------------------------
        # Load predictions
        # -----------------------------------
        self.predictions_df = (
            self.load_predictions()
        )

    # -----------------------------------
    # Load forecast CSV
    # -----------------------------------
    def load_predictions(self):

        try:

            # --------------------------------
            # File existence check
            # --------------------------------
            if not os.path.exists(
                self.file_path
            ):

                print(

                    "[ForecastEngine] "

                    "Forecast file not found"
                )

                return pd.DataFrame()

            # --------------------------------
            # Read CSV
            # --------------------------------
            df = pd.read_csv(
                self.file_path
            )

            # --------------------------------
            # Normalize column names
            # --------------------------------
            df.columns = (

                df.columns

                .str.strip()

                .str.lower()
            )

            # --------------------------------
            # Minimal required columns
            # --------------------------------
            required_columns = {

                "ticker",
                "date"
            }

            missing = (

                required_columns
                - set(df.columns)
            )

            if missing:

                print(

                    "[ForecastEngine] "

                    f"Missing columns: "
                    f"{missing}"
                )

                return pd.DataFrame()

            # --------------------------------
            # Normalize dates
            # --------------------------------
            df["date"] = pd.to_datetime(
                df["date"]
            )

            # --------------------------------
            # Normalize tickers
            # --------------------------------
            df["ticker"] = (

                df["ticker"]

                .astype(str)

                .str.upper()

                .str.strip()
            )

            print(

                "[ForecastEngine] "

                f"Loaded {len(df)} "
                f"forecast rows"
            )

            return df

        except Exception as e:

            print(

                "[ForecastEngine] "

                f"Load error: {e}"
            )

            return pd.DataFrame()

    # -----------------------------------
    # Get forecast prediction
    # -----------------------------------
    def predict(

        self,

        symbol,

        target_date=None
    ):

        try:

            # --------------------------------
            # Empty dataframe
            # --------------------------------
            if self.predictions_df.empty:

                return None

            # --------------------------------
            # Normalize symbol
            # --------------------------------
            symbol = symbol.upper().strip()

            # --------------------------------
            # Default:
            # tomorrow forecast
            # --------------------------------
            if target_date is None:

                target_date = (

                    datetime.now()

                    + timedelta(days=1)
                )

            else:

                target_date = pd.to_datetime(
                    target_date
                )

            # --------------------------------
            # Filter ticker rows
            # --------------------------------
            matches = self.predictions_df[

                self.predictions_df[
                    "ticker"
                ] == symbol
            ]

            # --------------------------------
            # No ticker found
            # --------------------------------
            if matches.empty:

                print(

                    "[ForecastEngine] "

                    f"No forecast for "
                    f"{symbol}"
                )

                return None

            # --------------------------------
            # Future predictions only
            # --------------------------------
            future_matches = matches[

                matches["date"] >=
                target_date
            ]

            # --------------------------------
            # Fallback to nearest date
            # --------------------------------
            if future_matches.empty:

                future_matches = matches

            # --------------------------------
            # Find nearest prediction
            # --------------------------------
            future_matches = (
                future_matches.copy()
            )

            future_matches[
                "date_diff"
            ] = abs(

                future_matches["date"]

                - target_date
            )

            future_matches = (
                future_matches.sort_values(
                    "date_diff"
                )
            )

            row = future_matches.iloc[0]

            # --------------------------------
            # Convert row to dictionary
            # --------------------------------
            prediction = row.to_dict()

            # --------------------------------
            # Remove unwanted fields
            # --------------------------------
            prediction.pop(
                "current_price",
                None
            )

            prediction.pop(
                "price_target",
                None
            )

            prediction.pop(
                "date_diff",
                None
            )

            # --------------------------------
            # Normalize prediction values
            # --------------------------------
            prediction["ticker"] = symbol

            prediction["prediction_date"] = str(
                row["date"].date()
            )

            # --------------------------------
            # Normalize confidence
            # --------------------------------
            if "confidence" in prediction:

                prediction["confidence"] = float(

                    prediction["confidence"]
                )

            else:

                prediction["confidence"] = 70.0

            # --------------------------------
            # Normalize numeric values
            # --------------------------------
            numeric_fields = [

                "forecast_price",

                "expected_return",

                "upper_band",

                "lower_band"
            ]

            for field in numeric_fields:

                if field in prediction:

                    try:

                        prediction[field] = float(
                            prediction[field]
                        )

                    except:

                        pass

            return prediction

        except Exception as e:

            print(

                "[ForecastEngine] "

                f"Prediction error: {e}"
            )

            return None

    # -----------------------------------
    # Build readable forecast context
    # -----------------------------------
    def build_forecast_context(
        self,
        prediction
    ):

        if not prediction:

            return ""

        lines = [

            "Forecast Analysis",
            "",
        ]

        # --------------------------------
        # Pretty formatting
        # --------------------------------
        for key, value in prediction.items():

            # ----------------------------
            # Skip raw internal fields
            # ----------------------------
            if key == "date":

                continue

            # ----------------------------
            # Confidence formatting
            # ----------------------------
            if key == "confidence":

                value = (
                    round(value * 100, 1)
                )

                value = f"{value}%"

            # ----------------------------
            # Expected return formatting
            # ----------------------------
            elif key == "expected_return":

                value = (
                    round(value * 100, 2)
                )

                value = f"{value:+.2f}%"

            # ----------------------------
            # Price formatting
            # ----------------------------
            elif isinstance(value, float):

                if (
                    "price" in key
                    or "band" in key
                ):

                    value = f"${value:.2f}"

            # ----------------------------
            # Pretty key formatting
            # ----------------------------
            pretty_key = (

                key

                .replace("_", " ")

                .title()
            )

            lines.append(
                f"{pretty_key}: {value}"
            )

        return "\n".join(lines).strip()