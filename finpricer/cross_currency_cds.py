import datetime
import math
import numpy as np
from utils.datetime_utils import DateUtils
from utils.curve_utils import YieldCurve


def interpolate_curve(time_points, values, target_time):
    """
    Linearly interpolates a value from a curve (e.g., hazard rate, discount factor).
    Assumes time_points are sorted.
    """
    return np.interp(target_time, time_points, values)


def get_hazard_rate(t, hazard_curve):
    """
    Retrieves interpolated hazard rate for time t.
    hazard_curve: List of (time_to_maturity, hazard_rate) tuples.
    """
    times = [point[0] for point in hazard_curve]
    rates = [point[1] for point in hazard_curve]
    if t < times[0]:
        return rates[0]  # Assume constant hazard rate before first point
    if t > times[-1]:
        return rates[-1]  # Assume constant hazard rate after last point
    return interpolate_curve(times, rates, t)


def get_discount_factor(t, discount_curve):
    """
    Retrieves interpolated discount factor for time t.
    discount_curve: List of (time_to_maturity, discount_factor) tuples.
    """
    times = [point[0] for point in discount_curve]
    dfs = [point[1] for point in discount_curve]
    if t < times[0]:
        return dfs[0]  # Assume constant DF before first point
    if t > times[-1]:
        return dfs[-1]  # Assume constant DF after last point
    return interpolate_curve(times, dfs, t)


def survival_probability(t, hazard_curve, valuation_date_time_zero=0.0):
    """
    Calculates survival probability to time t using integrated hazard rates.
    Assumes hazard rates are constant between given points.
    A more accurate approach would integrate piecewise constant hazard rates.
    For simplicity here, we assume a continuous, interpolated hazard rate.
    P(T > t) = exp(-cumulative_hazard_rate(t))
    """
    if t < valuation_date_time_zero:
        return 1.0  # Cannot default before valuation

    # A more robust way would be to integrate the interpolated hazard rate
    # over the interval [valuation_date_time_zero, t].
    # For a simple approximation, we'll use the hazard rate at time t
    # multiplied by t if the curve points are sparse or constant.
    # If the curve is defined by cumulative hazard rates: P(T>t) = exp(-H(t)).
    # If the curve is defined by instantaneous hazard rates: H(t) = integral(h(u)du from 0 to t).

    # Let's assume for this example, hazard_curve provides *cumulative* hazard rates,
    # or that the points are dense enough that average hazard_rate(t) * t is a good approx.
    # A more correct approach for instantaneous hazard rates (h(t)) would be:
    # Cumulative_hazard = integral from 0 to t of h(tau) d(tau)
    # Using a simple average hazard rate for the interval [0,t]:
    avg_hazard_rate = get_hazard_rate(t, hazard_curve)  # If this returns instantaneous, then multiply by t
    return math.exp(-avg_hazard_rate * t)


def price_cross_currency_cds(
        valuation_date: datetime.date,
        maturity_date: datetime.date,
        notional: float,
        recovery_rate: float,  # 0 to 1
        premium_rate: float,  # Annual rate, e.g., 0.01 for 100 bps
        payment_frequency: str,  # 'quarterly', 'semi-annually', 'annually'
        hazard_curve: list[tuple[float, float]],  # (time_to_maturity_years, hazard_rate_per_year)
        base_ccy_discount_curve: list[tuple[float, float]]  # (time_to_maturity_years, discount_factor)
):
    """
    Prices a Cross-Currency Credit Default Swap and calculates its par spread.

    Args:
        valuation_date (datetime.date): The date of valuation.
        maturity_date (datetime.date): The maturity date of the CDS.
        notional (float): The notional amount of the CDS (in the base currency).
        recovery_rate (float): The recovery rate (e.g., 0.4 for 40%).
        premium_rate (float): The annual premium rate (for calculating the value, set to 0 for par spread calculation).
        payment_frequency (str): 'quarterly', 'semi-annually', or 'annually'.
        hazard_curve (list[tuple[float, float]]): A list of (time_to_maturity_years, hazard_rate_per_year)
            tuples defining the credit curve for the reference entity. This curve captures the credit
            risk of the *foreign* entity, but is used to derive survival probabilities for the
            base currency swap.
        base_ccy_discount_curve (list[tuple[float, float]]): A list of (time_to_maturity_years, discount_factor)
            tuples defining the discount curve for the base currency of the swap.

    Returns:
        dict: A dictionary containing 'pv_premium_leg', 'pv_protection_leg', 'cds_value', 'par_spread'.
    """

    if valuation_date >= maturity_date:
        raise ValueError("Maturity date must be after valuation date.")
    if not (0 <= recovery_rate <= 1):
        raise ValueError("Recovery rate must be between 0 and 1.")

    payment_interval_months = {
        'annually': 12,
        'semi-annually': 6,
        'quarterly': 3
    }.get(payment_frequency)

    if payment_interval_months is None:
        raise ValueError("Invalid payment_frequency. Choose 'annually', 'semi-annually', or 'quarterly'.")

    # --- Generate payment dates ---
    payment_dates = []
    current_date = maturity_date
    while current_date > valuation_date:
        payment_dates.insert(0, current_date)
        # Simple date calculation, adjust for month-end if necessary in real world
        try:
            current_date = current_date.replace(month=current_date.month - payment_interval_months)
        except ValueError:  # Handle crossing year boundary
            current_date = current_date.replace(year=current_date.year - 1,
                                                month=current_date.month - payment_interval_months + 12)
        if current_date < valuation_date:
            payment_dates.insert(0, valuation_date)  # The first "payment" interval starts from valuation_date
            payment_dates = sorted(list(set(payment_dates)))  # Remove duplicates and sort

    # Ensure valuation date is the first effective "payment" date
    if payment_dates[0] != valuation_date:
        payment_dates.insert(0, valuation_date)

    # Filter out dates before valuation_date (due to simple generation logic)
    payment_dates = [d for d in payment_dates if d >= valuation_date]

    pv_premium_leg = 0.0
    pv_protection_leg = 0.0
    premium_leg_pv01_denominator = 0.0  # Denominator for par spread calculation

    # First period starts at valuation_date (t=0)
    prev_time = 0.0
    prev_survival_prob = 1.0  # Survival probability at t=0

    # Iterate through payment intervals
    for i in range(1, len(payment_dates)):
        current_date = payment_dates[i]
        start_of_period_date = payment_dates[i - 1]

        time_to_current = DateUtils.year_fraction(valuation_date, current_date)
        time_to_start_of_period = DateUtils.year_fraction(valuation_date, start_of_period_date)

        # Ensure times are non-negative
        time_to_current = max(0.0, time_to_current)
        time_to_start_of_period = max(0.0, time_to_start_of_period)

        # --- Premium Leg Calculation ---
        # Accrual period for premium payment
        day_count_fraction = DateUtils.year_fraction(start_of_period_date, current_date)

        # Survival probability to the payment date
        survival_prob_current = survival_probability(time_to_current, hazard_curve)

        # Discount factor to the payment date
        df_current = get_discount_factor(time_to_current, base_ccy_discount_curve)

        # Expected premium payment
        # Note: Standard CDS pricing assumes premium is paid at the *end* of the period,
        # conditional on survival.
        expected_premium_payment = notional * premium_rate * day_count_fraction * survival_prob_current * df_current
        pv_premium_leg += expected_premium_payment

        # Denominator for par spread calculation (PV01 of the premium leg, effectively)
        premium_leg_pv01_denominator += notional * day_count_fraction * survival_prob_current * df_current

        # --- Protection Leg Calculation ---
        # Probability of default in this interval
        prob_default_in_interval = prev_survival_prob - survival_prob_current

        # Mid-point of the interval for discounting the default payment
        # Assumes default payment occurs at the mid-point of the period
        time_to_mid_point = (time_to_start_of_period + time_to_current) / 2.0
        df_mid = get_discount_factor(time_to_mid_point, base_ccy_discount_curve)

        # Expected protection payment
        expected_protection_payment = notional * (1 - recovery_rate) * prob_default_in_interval * df_mid
        pv_protection_leg += expected_protection_payment

        # Update for next iteration
        prev_survival_prob = survival_prob_current
        prev_time = time_to_current

    cds_value = pv_premium_leg - pv_protection_leg

    # Calculate Par Spread: The spread that makes the CDS value zero
    par_spread = pv_protection_leg / premium_leg_pv01_denominator if premium_leg_pv01_denominator != 0 else 0.0

    return {
        'pv_premium_leg': pv_premium_leg,
        'pv_protection_leg': pv_protection_leg,
        'cds_value': cds_value,
        'par_spread': par_spread
    }


# --- Example Usage ---
if __name__ == "__main__":
    # --- Input Parameters ---
    valuation_date = datetime.date(2023, 1, 15)
    maturity_date = datetime.date(2028, 1, 15)  # 5-year CDS
    notional = 10_000_000  # USD Notional
    recovery_rate = 0.40  # 40% recovery

    # Premium rate for initial pricing (e.g., if you want to value an existing CDS)
    # For par spread calculation, this value doesn't affect the final 'par_spread' result
    # as the 'par_spread' is derived to make value zero.
    premium_rate_input = 0.0100  # 100 bps

    payment_frequency = 'quarterly'  # Common for CDS

    # --- Hazard Rate Curve (for a "foreign" entity, e.g., German company) ---
    # Represented as (time_in_years, instantaneous_hazard_rate_per_year)
    # These rates are derived from the creditworthiness of the foreign entity.
    # In a real scenario, this might involve cross-currency basis swaps to convert
    # foreign-currency credit spreads into base-currency implied hazard rates.
    hazard_curve = [
        (0.5, 0.005),
        (1.0, 0.006),
        (2.0, 0.007),
        (3.0, 0.008),
        (5.0, 0.010),
        (7.0, 0.011),
        (10.0, 0.012)
    ]

    # --- Base Currency (e.g., USD) Discount Curve ---
    # Represented as (time_in_years, discount_factor)
    # These are for the currency the CDS is denominated in (USD).
    base_ccy_discount_curve = [
        (0.25, 0.998),  # 3 months
        (0.50, 0.995),  # 6 months
        (1.00, 0.990),  # 1 year
        (2.00, 0.980),  # 2 years
        (3.00, 0.970),  # 3 years
        (5.00, 0.950),  # 5 years
        (7.00, 0.930),  # 7 years
        (10.00, 0.900)  # 10 years
    ]

    print("--- Pricing Cross-Currency Credit Default Swap ---")
    print(f"Valuation Date: {valuation_date}")
    print(f"Maturity Date: {maturity_date}")
    print(f"Notional: ${notional:,.2f}")
    print(f"Recovery Rate: {recovery_rate:.2%}")
    print(f"Premium Rate (for valuing existing CDS): {premium_rate_input:.2%}")
    print(f"Payment Frequency: {payment_frequency.capitalize()}")

    results = price_cross_currency_cds(
        valuation_date,
        maturity_date,
        notional,
        recovery_rate,
        premium_rate_input,  # Use the input premium rate for 'cds_value'
        payment_frequency,
        hazard_curve,
        base_ccy_discount_curve
    )

    print("\n--- Results ---")
    print(f"PV of Premium Leg: ${results['pv_premium_leg']:,.2f}")
    print(f"PV of Protection Leg: ${results['pv_protection_leg']:,.2f}")
    print(f"CDS Value (at {premium_rate_input:.2%} spread): ${results['cds_value']:,.2f}")
    print(f"Calculated Par Spread: {results['par_spread']:.4%} ({results['par_spread'] * 10000:.0f} bps)")

    # Verify par spread by setting premium_rate to calculated par_spread
    # and re-running to see if CDS value is close to zero
    print("\n--- Verification (Pricing at Par Spread) ---")
    results_par_spread = price_cross_currency_cds(
        valuation_date,
        maturity_date,
        notional,
        recovery_rate,
        results['par_spread'],  # Use the calculated par spread
        payment_frequency,
        hazard_curve,
        base_ccy_discount_curve
    )
    print(
        f"CDS Value (at Par Spread {results['par_spread']:.4%}): ${results_par_spread['cds_value']:,.2f} (should be close to zero)")