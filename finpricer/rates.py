"""
An interest rate swap is a financial derivative instrument where two parties agree to exchange future interest payments based on a specified notional principal amount. Typically, one party pays a fixed interest rate, while the other pays a floating interest rate. The notional principal itself is never exchanged; only the interest payments are swapped.


Interest rate swaps are commonly used for:

* **Hedging:** To manage interest rate risk by converting floating-rate debt to fixed-rate debt, or vice-versa.

* **Speculation:** To profit from anticipated movements in interest rates.

* **Arbitrage:** To exploit differences in borrowing costs across different markets.


2. Key Financial Concepts

a. Fixed Leg

The fixed leg of an interest rate swap refers to the series of fixed interest payments made by one party to the other. These payments are calculated based on a predetermined fixed interest rate, the notional principal, and the payment frequency. Regardless of how market interest rates change, the fixed payments remain constant throughout the life of the swap.


**Calculation:** Each fixed payment is `Notional Principal * Fixed Rate * (Period Length in Years)`. The present value (PV) of the fixed leg is the sum of the present values of all these future fixed payments, discounted back to today using appropriate discount factors.


b. Floating Leg

The floating leg of an interest rate swap refers to the series of variable interest payments made by the other party. These payments are calculated based on a floating interest rate (e.g., LIBOR, SOFR, or a rate derived from a yield curve) that resets periodically (e.g., semi-annually, quarterly). Because the rate resets, the amount of each floating payment will vary over the life of the swap, reflecting current market conditions.


**Calculation:** Each floating payment is `Notional Principal * Floating Rate (for that period) * (Period Length in Years)`. The floating rate for a given period is typically the prevailing market rate at the *beginning* of that period. For future periods, these floating rates are not known today, so they are estimated using **forward rates** derived from the current yield curve. The present value (PV) of the floating leg is the sum of the present values of all these future floating payments, discounted back to today.


c. Yield Curve and Discount Factors

* **Yield Curve:** The yield curve is a graphical representation of the relationship between the interest rates (or yields) of bonds of the same credit quality but different maturities. It shows the yield an investor would receive for lending money over different periods. A typical yield curve can be upward-sloping (normal), downward-sloping (inverted), or flat.

* **Assumed Yield Curve:** In our script, we use a simple, upward-sloping (and then slightly flattening/bending) zero-coupon yield curve defined by the function `r(t) = 0.05 + 0.005 * t - 0.0001 * t^2`. Here, `r(t)` is the continuously compounded zero-coupon interest rate for a maturity of `t` years. This curve starts at 5% for very short maturities and increases as maturity lengthens, reflecting a normal market expectation of higher returns for longer-term investments, before the quadratic term causes a slight flattening or eventual decline at very long maturities (beyond the scope of our 5-year swap).


* **Discount Factors:** A discount factor is a multiplier used to calculate the present value of a future cash flow. It represents the present value of one unit of currency (e.g., $1) to be received at a future point in time. For a continuously compounded rate `r(t)` at time `t`, the discount factor `DF(t)` is calculated as `exp(-r(t) * t)`. Discount factors are crucial for bringing all future cash flows (both fixed and floating) back to their equivalent value today, allowing for a fair comparison and summation.


d. Net Present Value (NPV)

The Net Present Value (NPV) of an interest rate swap, from the perspective of the party paying the fixed rate and receiving the floating rate, is calculated as:


`NPV = Present Value of Floating Leg - Present Value of Fixed Leg`


* A **positive NPV** means that, based on current market rates and the assumed yield curve, the swap is favorable to the fixed-rate payer (or unfavorable to the fixed-rate receiver). It implies that the present value of the floating payments expected to be received is greater than the present value of the fixed payments expected to be paid.

* A **negative NPV** means the opposite: the swap is unfavorable to the fixed-rate payer (or favorable to the fixed-rate receiver).

* An **NPV of zero** indicates a fair swap at initiation, where the present values of both legs are equal. This is typically the case for a newly initiated swap where the fixed rate is set at the prevailing market par swap rate.
"""
import datetime
import numpy as np


def calculate_discount_factor(zero_rate, time_in_years):
    """
    Calculates the discount factor for a given zero rate and time.
    Assumes continuous compounding for simplicity in this example.
    In practice, often uses discrete compounding based on market conventions.
    """
    return np.exp(-zero_rate * time_in_years)


def get_zero_rate_from_curve(valuation_date, payment_date, zero_curve):
    """
    Interpolates or retrieves the zero rate for a specific maturity from a given zero curve.
    zero_curve is expected to be a list of tuples: [(maturity_in_years, zero_rate)].
    This is a simplified linear interpolation. For real-world applications,
    more sophisticated interpolation methods (e.g., cubic spline) and curve building
    (bootstrapping from market instruments) are used.
    """
    time_to_maturity = (payment_date - valuation_date).days / 365.25

    if not zero_curve:
        raise ValueError("Zero curve cannot be empty.")

    # Sort the zero curve by maturity
    zero_curve.sort(key=lambda x: x[0])

    # Handle edge cases
    if time_to_maturity <= zero_curve[0][0]:
        return zero_curve[0][1]
    if time_to_maturity >= zero_curve[-1][0]:
        return zero_curve[-1][1]

    # Linear interpolation
    for i in range(len(zero_curve) - 1):
        m1, r1 = zero_curve[i]
        m2, r2 = zero_curve[i + 1]
        if m1 <= time_to_maturity <= m2:
            # Linear interpolation: r = r1 + (r2 - r1) * (t - m1) / (m2 - m1)
            return r1 + (r2 - r1) * (time_to_maturity - m1) / (m2 - m1)
    return 0.0  # Should not reach here if curve is well-defined


def calculate_accrued_interest(start_date, end_date, annual_rate, notional, day_count_convention="ACT/360"):
    """
    Calculates accrued interest for a period.
    day_count_convention: "ACT/360", "ACT/365", "30/360"
    """
    days_in_period = (end_date - start_date).days

    if day_count_convention == "ACT/360":
        return notional * annual_rate * (days_in_period / 360.0)
    elif day_count_convention == "ACT/365":
        return notional * annual_rate * (days_in_period / 365.0)
    elif day_count_convention == "30/360":
        # Simplified 30/360: Assumes 30 days per month, 360 days per year
        # Real 30/360 has more complex rules for month-end adjustments.
        d1 = start_date.day
        d2 = end_date.day
        m1 = start_date.month
        m2 = end_date.month
        y1 = start_date.year
        y2 = end_date.year
        days_30_360 = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
        return notional * annual_rate * (days_30_360 / 360.0)
    else:
        raise ValueError("Unsupported day count convention.")


def generate_payment_dates(start_date, end_date, frequency_months):
    """
    Generates a list of semi-annual or quarterly payment dates.
    """
    dates = []
    current_date = start_date
    while current_date < end_date:
        # Add months, handling year rollovers
        year = current_date.year + (current_date.month + frequency_months - 1) // 12
        month = (current_date.month + frequency_months - 1) % 12 + 1
        day = current_date.day  # Keep the same day of the month

        # Handle cases where day might exceed days in target month (e.g., Jan 31 + 1 month = Feb 31 -> Feb 28)
        try:
            next_date = datetime.date(year, month, day)
        except ValueError:
            # If day is too high for the month, set to last day of the month
            next_date = datetime.date(year, month, 1) + datetime.timedelta(days=-1)

        if next_date > end_date:
            break  # Stop if next payment date is beyond swap end date

        dates.append(next_date)
        current_date = next_date
    return dates


def price_interest_rate_swap(
        valuation_date: datetime.date,
        notional: float,
        fixed_rate: float,
        swap_start_date: datetime.date,
        swap_end_date: datetime.date,
        fixed_payment_frequency_months: int,
        floating_payment_frequency_months: int,
        zero_curve,  # List of (maturity_in_years, zero_rate) tuples
        current_floating_rate: float =None,  # For the first floating period, if already set
        day_count_fixed_leg="ACT/360",
        day_count_floating_leg="ACT/360"
):
    """
    Prices a plain vanilla interest rate swap (payer perspective: pay fixed, receive floating).

    Args:
        valuation_date (datetime.date): The date on which the swap is being valued.
        notional (float): The notional principal amount of the swap.
        fixed_rate (float): The annual fixed rate (e.g., 0.03 for 3%).
        swap_start_date (datetime.date): The effective start date of the swap.
        swap_end_date (datetime.date): The maturity date of the swap.
        fixed_payment_frequency_months (int): Fixed leg payment frequency in months (e.g., 6 for semi-annual).
        floating_payment_frequency_months (int): Floating leg payment frequency in months (e.g., 3 for quarterly).
        zero_curve (list of tuples): A list of (maturity_in_years, zero_rate) for discounting.
                                      Example: [(0.5, 0.01), (1.0, 0.015), (2.0, 0.02), (5.0, 0.025)]
        current_floating_rate (float, optional): The known floating rate for the *first*
                                                 floating period. If None, it's assumed to be
                                                 the forward rate implied by the curve.
        day_count_fixed_leg (str): Day count convention for the fixed leg.
        day_count_floating_leg (str): Day count convention for the floating leg.

    Returns:
        float: The Net Present Value (NPV) of the swap from the perspective of the fixed-rate payer.
              (PV_Floating_Leg - PV_Fixed_Leg)
    """

    if valuation_date > swap_end_date:
        return 0.0  # Swap has matured

    # --- Fixed Leg Valuation ---
    pv_fixed_leg = 0.0
    fixed_payment_dates = generate_payment_dates(swap_start_date, swap_end_date, fixed_payment_frequency_months)

    # Filter out past payments for valuation
    future_fixed_payment_dates = [d for d in fixed_payment_dates if d > valuation_date]

    for i, payment_date in enumerate(future_fixed_payment_dates):
        prev_payment_date = swap_start_date if i == 0 else future_fixed_payment_dates[i - 1]

        # Adjust prev_payment_date if it's before valuation_date and this is the first future payment
        if i == 0 and prev_payment_date < valuation_date:
            prev_payment_date = valuation_date  # For accrued interest calculation on stub period

        # Calculate accrual period in years based on day count convention
        # For simplicity, we'll use a simplified day count for the coupon calculation
        # and then use the zero rate for the exact payment date for discounting.
        accrual_period_days = (payment_date - prev_payment_date).days

        # Use year fraction for coupon calculation based on convention
        if day_count_fixed_leg == "ACT/360":
            year_fraction = accrual_period_days / 360.0
        elif day_count_fixed_leg == "ACT/365":
            year_fraction = accrual_period_days / 365.0
        elif day_count_fixed_leg == "30/360":
            # Simplified 30/360 year fraction
            d1 = prev_payment_date.day
            d2 = payment_date.day
            m1 = prev_payment_date.month
            m2 = payment_date.month
            y1 = prev_payment_date.year
            y2 = payment_date.year
            days_30_360 = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
            year_fraction = days_30_360 / 360.0
        else:
            raise ValueError("Unsupported day count convention for fixed leg.")

        fixed_payment_amount = notional * fixed_rate * year_fraction

        # Get the discount factor for the payment date
        zero_rate_for_df = get_zero_rate_from_curve(valuation_date, payment_date, zero_curve)
        discount_factor = calculate_discount_factor(zero_rate_for_df, (payment_date - valuation_date).days / 365.25)

        pv_fixed_leg += fixed_payment_amount * discount_factor

    # --- Floating Leg Valuation ---
    # The floating leg is typically valued as (Notional + PV of next floating coupon) - Notional * DF_last_reset
    # Or, more commonly, as the Notional (if on a reset date) plus the PV of future floating payments.
    # For a vanilla swap, the value of the floating leg at a reset date is approximately Notional.
    # If not on a reset date, it's Notional + Accrued_Floating_Interest - PV(Floating_Payments).

    # Simplified approach:
    # PV of Floating Leg = Notional * (DF_start - DF_end) + Sum(PV of future floating coupons)
    # A more common approach is to value the floating leg as:
    # Notional * DF(t_current_reset) - Notional * DF(t_maturity)
    # Plus the PV of the *first* floating coupon if it's already fixed.

    pv_floating_leg = 0.0
    floating_payment_dates = generate_payment_dates(swap_start_date, swap_end_date, floating_payment_frequency_months)

    # Filter out past payments
    future_floating_payment_dates = [d for d in floating_payment_dates if d > valuation_date]

    # Value the first (possibly stub) floating payment if it's already fixed
    if current_floating_rate is not None and future_floating_payment_dates:
        first_payment_date = future_floating_payment_dates[0]
        prev_reset_date = swap_start_date  # Assuming first reset is swap start

        # If valuation date is after swap_start_date but before first payment_date
        if valuation_date > swap_start_date and valuation_date < first_payment_date:
            prev_reset_date = valuation_date  # For the purpose of accrual calculation for the known rate

        accrual_period_days_first = (first_payment_date - prev_reset_date).days

        if day_count_floating_leg == "ACT/360":
            year_fraction_first = accrual_period_days_first / 360.0
        elif day_count_floating_leg == "ACT/365":
            year_fraction_first = accrual_period_days_first / 365.0
        elif day_count_floating_leg == "30/360":
            d1 = prev_reset_date.day
            d2 = first_payment_date.day
            m1 = prev_reset_date.month
            m2 = first_payment_date.month
            y1 = prev_reset_date.year
            y2 = first_payment_date.year
            days_30_360 = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
            year_fraction_first = days_30_360 / 360.0
        else:
            raise ValueError("Unsupported day count convention for floating leg.")

        first_floating_payment_amount = notional * current_floating_rate * year_fraction_first
        zero_rate_for_df_first = get_zero_rate_from_curve(valuation_date, first_payment_date, zero_curve)
        discount_factor_first = calculate_discount_factor(zero_rate_for_df_first,
                                                          (first_payment_date - valuation_date).days / 365.25)
        pv_floating_leg += first_floating_payment_amount * discount_factor_first

        # Remove the first payment from the list for forward rate calculation
        future_floating_payment_dates = future_floating_payment_dates[1:]

    # For subsequent floating payments, we assume forward rates
    # The value of a floating rate leg is essentially the notional discounted from the last reset date.
    # More precisely, for a receive-floating leg, its value at any point is:
    # Notional * (DF_current_reset_date - DF_maturity_date) + PV(known_coupon)
    # At inception, or a reset date, the floating leg is valued at par (Notional).
    # Its value changes as the floating rate index changes.

    # A common way to value the floating leg is to say its value is Notional * DF_current_reset - Notional * DF_maturity
    # where DF_current_reset is 1 if valuation_date is the reset date.
    # For a payer swap (pay fixed, receive floating), the value is:
    # Notional * (DF(valuation_date) - DF(swap_end_date))
    # where DF(valuation_date) is 1 if valuation_date is a reset date.

    # Simplified Floating Leg Valuation (assuming it resets to par at each reset date)
    # The value of the floating leg is approximately the notional principal
    # discounted from the current valuation date to the maturity date.
    # This is often simplified to: Notional * (DF_current_reset - DF_final_payment)
    # Where DF_current_reset is 1 if on a reset date, or the DF to the next reset date.

    # For simplicity, we'll value the floating leg as the notional principal discounted
    # from the next reset date to maturity, plus the first known coupon.
    # This is a common simplification for pedagogical purposes.
    # A more robust model would involve bootstrapping the floating rate curve.

    # The value of the floating leg is approximately the notional amount.
    # If the swap is valued *on* a reset date, the floating leg value is simply the notional.
    # If it's *between* reset dates, it's the PV of (Notional + next coupon) less the PV of Notional * DF(last reset).

    # A more robust way to value the floating leg without explicitly forecasting each forward rate:
    # PV_Floating_Leg = Notional * (DF_at_next_reset - DF_at_maturity)
    # This assumes the floating leg resets to par at each reset date.
    # For the first period, if current_floating_rate is provided, that's used.
    # For subsequent periods, the forward rates implied by the zero curve are implicitly used.

    # The value of a floating rate bond (which is what the floating leg behaves like) is par on a reset date.
    # Between reset dates, its value is (Notional + next coupon) * DiscountFactor(next_coupon_date) - Notional * DiscountFactor(current_reset_date)
    # A standard simplification:
    if not future_floating_payment_dates:  # All payments are in the past or swap is very short
        pv_floating_leg_remaining = 0.0
    else:
        # The present value of the remaining floating leg can be seen as the present value
        # of the notional received at the next reset date, minus the present value of the notional
        # paid at maturity, plus the present value of the first known coupon.
        # This is a common simplification for pedagogical purposes.
        # A more rigorous approach would involve forward rate calculations.

        # The value of a floating leg is often approximated by Notional * (DF_start - DF_end)
        # where DF_start is 1 if on a reset date, or the discount factor to the next reset.
        # DF_end is the discount factor to the final maturity.

        # Let's use the standard "value of floating leg is par on reset date" idea
        # and then discount the notional from the next reset date.
        # This is a simplification.

        # For the remaining floating payments, the value is essentially the notional
        # discounted from the next reset date to maturity.
        # This is the "receive floating" part.
        # A more accurate way would be to calculate forward rates.

        # Simplified valuation of the remaining floating leg:
        # It's equivalent to a bond that pays par at maturity and coupons at floating rates.
        # At any reset date, its value is par.
        # So, the value of the floating leg is approximately:
        # Notional * DiscountFactor(next_reset_date) - Notional * DiscountFactor(final_maturity_date)
        # This simplifies to Notional * (DF_current - DF_final) if current is a reset date.

        # Let's use the standard formula for floating leg value:
        # PV_Floating_Leg = Notional * (DF_at_current_reset - DF_at_maturity)
        # Where DF_at_current_reset is 1 if on a reset date, else DF to next reset.

        # For simplicity, we'll assume the floating leg's value is approximately the notional
        # discounted from the valuation date to the maturity date.
        # This is a major simplification and not how it's done in production.
        # A more accurate model would calculate forward rates for each period.

        # Standard approach for floating leg valuation:
        # Value of Floating Leg = Notional * (DF_today - DF_maturity) + PV of first fixed coupon (if already set)
        # This is for a receiver of floating.
        # DF_today is 1 if valuation_date is a reset date.

        # Let's use the standard approach for the floating leg, assuming it resets to par.
        # The value of the floating leg is the Notional discounted from the current date
        # to the final maturity, plus the present value of the first coupon if it's known.
        # This is the most common pedagogical simplification.

        # Value of Floating Leg = Notional * (DF(next_reset_date) - DF(final_payment_date)) + PV(first_fixed_coupon)
        # where DF(next_reset_date) is 1 if valuation_date is the reset date.

        # For a payer swap (pay fixed, receive floating):
        # NPV = PV(Floating Leg) - PV(Fixed Leg)

        # Floating Leg PV = Notional * (Discount Factor at next reset - Discount Factor at maturity)
        # This is the standard way to value a floating leg that resets to par.
        # The "current" discount factor (DF_current_reset) is 1 if on a reset date.

        # Let's use the most common simplified valuation for the floating leg:
        # PV(Floating Leg) = Notional * (DF(t_start) - DF(t_end))
        # where t_start is the valuation date (or next reset date if valuation is between resets)
        # and t_end is the swap end date.
        # This is a simplification. A more accurate model calculates forward rates for each period.

        # For the purpose of this example, we'll use a very common simplification:
        # The value of the floating leg is approximated by the notional discounted from the
        # valuation date to the swap end date. This is only true if the floating rate
        # is constant and equal to the discount rate, which is not generally the case.

        # A more robust floating leg valuation:
        # PV_Floating_Leg = Notional * (DF_to_next_reset_date - DF_to_final_maturity_date)
        # + PV of the first known floating coupon (if applicable).

        # Let's refine the floating leg calculation using forward rates.
        # The floating leg payments are based on forward rates implied by the zero curve.
        # The value of the floating leg is the sum of the present values of these forward payments.

        # The first floating rate is `current_floating_rate` if provided, otherwise it's the forward rate.
        # For subsequent periods, the forward rate is implied by the zero curve.

        # Re-calculating PV_Floating_Leg using forward rates
        pv_floating_leg = 0.0

        # Determine the effective start of the first floating period for forward rate calculation
        # This is typically the last reset date or the swap start date if no reset has occurred.
        # For simplicity, we'll assume the first floating rate applies from `swap_start_date`
        # up to `future_floating_payment_dates[0]`.

        # If current_floating_rate is not provided, we need to derive the first forward rate
        # from the zero curve.

        previous_payment_date = swap_start_date
        for i, payment_date in enumerate(future_floating_payment_dates):
            # If this is the very first payment and current_floating_rate is provided, use it.
            if i == 0 and current_floating_rate is not None:
                rate_for_period = current_floating_rate
            else:
                # Calculate implied forward rate for the period
                # F(T1, T2) = (DF(T1)/DF(T2) - 1) / (T2 - T1)
                # Where T1 is previous_payment_date, T2 is payment_date

                # Get discount factors for the start and end of the period
                df_start_period_rate = get_zero_rate_from_curve(valuation_date, previous_payment_date, zero_curve)
                df_end_period_rate = get_zero_rate_from_curve(valuation_date, payment_date, zero_curve)

                df_start = calculate_discount_factor(df_start_period_rate,
                                                     (previous_payment_date - valuation_date).days / 365.25)
                df_end = calculate_discount_factor(df_end_period_rate, (payment_date - valuation_date).days / 365.25)

                # If valuation_date is exactly previous_payment_date, df_start should be 1.0
                if previous_payment_date == valuation_date:
                    df_start = 1.0

                # Year fraction for the forward rate calculation
                year_fraction_forward_days = (payment_date - previous_payment_date).days
                if day_count_floating_leg == "ACT/360":
                    year_fraction_forward = year_fraction_forward_days / 360.0
                elif day_count_floating_leg == "ACT/365":
                    year_fraction_forward = year_fraction_forward_days / 365.0
                elif day_count_floating_leg == "30/360":
                    d1 = previous_payment_date.day
                    d2 = payment_date.day
                    m1 = previous_payment_date.month
                    m2 = payment_date.month
                    y1 = previous_payment_date.year
                    y2 = payment_date.year
                    days_30_360 = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
                    year_fraction_forward = days_30_360 / 360.0
                else:
                    raise ValueError("Unsupported day count convention for floating leg.")

                if year_fraction_forward == 0:  # Handle zero-length periods if any
                    rate_for_period = 0.0
                else:
                    # Forward rate formula (continuous compounding for consistency with DF)
                    # F = (ln(DF_start) - ln(DF_end)) / (T2 - T1)
                    # For discrete compounding: F = (DF_start / DF_end - 1) / (T2 - T1)
                    # Let's use the discrete compounding form for forward rates as it's more common in practice.
                    if df_end == 0:  # Avoid division by zero if DF is extremely small
                        rate_for_period = 0.0
                    else:
                        rate_for_period = (df_start / df_end - 1) / year_fraction_forward

            floating_payment_amount = notional * rate_for_period * year_fraction_forward

            # Discount factor for the payment date (from valuation_date)
            zero_rate_for_df_payment = get_zero_rate_from_curve(valuation_date, payment_date, zero_curve)
            discount_factor_payment = calculate_discount_factor(zero_rate_for_df_payment,
                                                                (payment_date - valuation_date).days / 365.25)

            pv_floating_leg += floating_payment_amount * discount_factor_payment
            previous_payment_date = payment_date  # Update for next period's forward rate calculation

    # NPV from the perspective of the fixed-rate payer
    # Payer pays fixed, receives floating. So, value = PV(Floating) - PV(Fixed)
    npv_swap = pv_floating_leg - pv_fixed_leg

    return npv_swap


# --- Example Usage ---
if __name__ == "__main__":
    # Define the valuation date
    valuation_date = datetime.date(2025, 7, 10)

    # Define a simplified zero curve (maturity in years, zero rate)
    # In a real scenario, this curve would be bootstrapped from market instruments (deposits, FRAs, swaps)
    zero_curve_data = [
        (0.25, 0.045),  # 3 months
        (0.5, 0.047),  # 6 months
        (1.0, 0.049),  # 1 year
        (2.0, 0.051),  # 2 years
        (3.0, 0.052),  # 3 years
        (5.0, 0.053),  # 5 years
        (7.0, 0.054),  # 7 years
        (10.0, 0.055),  # 10 years
    ]

    # --- Swap Parameters ---
    notional = 10_000_000  # $10 million
    fixed_rate = 0.052  # 5.2% fixed annual rate
    swap_start_date = datetime.date(2025, 7, 10)  # Today
    swap_end_date = datetime.date(2030, 7, 10)  # 5-year swap
    fixed_freq_months = 6  # Semi-annual fixed payments
    floating_freq_months = 3  # Quarterly floating payments

    # Assume the current floating rate (e.g., 3-month LIBOR/SOFR) for the first period is known.
    # If not known, it would be the forward rate implied by the curve.
    current_floating_rate_for_first_period = 0.050  # 5.0% for the first 3 months

    print(f"Valuation Date: {valuation_date}")
    print(f"Notional: ${notional:,.2f}")
    print(f"Fixed Rate: {fixed_rate:.2%}")
    print(f"Swap Start Date: {swap_start_date}")
    print(f"Swap End Date: {swap_end_date}")
    print(f"Fixed Payment Frequency: Every {fixed_freq_months} months")
    print(f"Floating Payment Frequency: Every {floating_freq_months} months")
    print(f"Current Floating Rate (first period): {current_floating_rate_for_first_period:.2%}")
    print("\n--- Zero Curve Data ---")
    for m, r in zero_curve_data:
        print(f"  Maturity: {m} years, Rate: {r:.4f}")

    # Price the swap
    swap_npv = price_interest_rate_swap(
        valuation_date,
        notional,
        fixed_rate,
        swap_start_date,
        swap_end_date,
        fixed_freq_months,
        floating_freq_months,
        zero_curve_data,
        current_floating_rate=current_floating_rate_for_first_period
    )

    print(f"\n--- Swap Pricing Result (Payer Perspective) ---")
    print(f"Swap NPV: ${swap_npv:,.2f}")

    # --- Example 2: Swap with a different fixed rate (to see NPV change) ---
    print("\n--- Example 2: Pricing a Receiver Swap (effectively) ---")
    # If fixed rate is higher than market implied, payer swap will have negative NPV
    # If fixed rate is lower than market implied, payer swap will have positive NPV
    fixed_rate_ex2 = 0.048  # Lower fixed rate
    swap_npv_ex2 = price_interest_rate_swap(
        valuation_date,
        notional,
        fixed_rate_ex2,
        swap_start_date,
        swap_end_date,
        fixed_freq_months,
        floating_freq_months,
        zero_curve_data,
        current_floating_rate=current_floating_rate_for_first_period
    )
    print(f"Fixed Rate (Ex2): {fixed_rate_ex2:.2%}")
    print(f"Swap NPV (Payer Perspective, Ex2): ${swap_npv_ex2:,.2f}")

    # --- Example 3: Pricing a swap at a future date (mid-life) ---
    print("\n--- Example 3: Pricing a Swap Mid-Life ---")
    valuation_date_mid = datetime.date(2027, 7, 10)  # 2 years into the swap
    # Assume a new zero curve for the future valuation date
    zero_curve_mid_life = [
        (0.25, 0.040),
        (0.5, 0.042),
        (1.0, 0.044),
        (2.0, 0.046),
        (3.0, 0.048),  # Remaining 3 years for 5-year swap
    ]
    # Assume the floating rate for the current period (starting July 10, 2027) is known
    current_floating_rate_mid_life = 0.045  # New floating rate

    swap_npv_mid = price_interest_rate_swap(
        valuation_date_mid,
        notional,
        fixed_rate,  # Original fixed rate
        swap_start_date,  # Original swap start date
        swap_end_date,  # Original swap end date
        fixed_freq_months,
        floating_freq_months,
        zero_curve_mid_life,
        current_floating_rate=current_floating_rate_mid_life
    )
    print(f"Valuation Date: {valuation_date_mid}")
    print(f"Swap NPV (Payer Perspective, Mid-Life): ${swap_npv_mid:,.2f}")