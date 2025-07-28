import numpy as np
from scipy.interpolate import interp1d
from utils.datetime_utils import DateUtils


class YieldCurve:
    """
    Represents a zero-coupon yield curve for discounting and forward rate calculation.
    Simplified: uses linear interpolation on zero rates.
    """

    def __init__(self, valuation_date, tenors_years, zero_rates):
        if len(tenors_years) != len(zero_rates):
            raise ValueError("Tenors and zero rates must have the same length.")
        if not all(tenors_years[i] < tenors_years[i + 1] for i in range(len(tenors_years) - 1)):
            raise ValueError("Tenors must be strictly increasing.")

        self.valuation_date = valuation_date
        self.tenors_years = np.array(tenors_years)
        self.zero_rates = np.array(zero_rates)  # Annual compounding assumed for zero rates

        # Create an interpolation function for zero rates
        self._zero_rate_interp = interp1d(self.tenors_years, self.zero_rates, kind='linear',
                                          fill_value="extrapolate")

    def get_zero_rate(self, date):
        """Gets the zero rate for a given date."""
        time_to_maturity_years = DateUtils.year_fraction(self.valuation_date, date, "ACT/365")
        if time_to_maturity_years <= 0:
            return self.zero_rates[0] if len(self.zero_rates) > 0 else 0.0  # Or handle error
        return self._zero_rate_interp(time_to_maturity_years).item()  # .item() to get scalar from array

    def get_discount_factor(self, date):
        """Calculates the discount factor for a given date."""
        time_to_maturity_years = DateUtils.year_fraction(self.valuation_date, date, "ACT/365")
        if time_to_maturity_years <= 0:
            return 1.0  # Discount factor for today is 1
        zero_rate = self.get_zero_rate(date)
        return np.exp(-zero_rate * time_to_maturity_years)

    def get_forward_rate(self, start_date, end_date, day_count_convention="ACT/365"):
        """
        Calculates the forward rate between two future dates.
        Assumes continuous compounding for zero rates.
        """
        if start_date < self.valuation_date or end_date < self.valuation_date:
            raise ValueError("Forward dates must be in the future relative to valuation date.")
        if end_date <= start_date:
            raise ValueError("End date must be after start date for forward rate calculation.")

        df_start = self.get_discount_factor(start_date)
        df_end = self.get_discount_factor(end_date)

        # Avoid division by zero if dates are too close or identical
        if df_end == 0:
            return float('inf')  # Or handle as error

        year_frac = DateUtils.year_fraction(start_date, end_date, day_count_convention)
        if year_frac == 0:
            return 0.0  # Or handle as error

        # Continuous compounding formula for forward rate
        return (np.log(df_start) - np.log(df_end)) / year_frac

