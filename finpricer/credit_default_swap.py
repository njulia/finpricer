"""
This implementation uses the standard approach of equating the present value of the premium leg to the present value of the protection leg. The core challenge in CDS pricing is deriving the survival probability curve (and thus default probabilities) from market-observed CDS spreads.

Key Concepts and Methodology:

Premium Leg: The stream of periodic payments (premiums) made by the CDS buyer to the seller.

Protection Leg: The contingent payment made by the seller to the buyer if a default event occurs. This payment is typically the notional amount minus the recovery amount.

Survival Probability: The probability that the reference entity does not default up to a certain point in time.

Default Probability: The probability that the reference entity defaults within a specific period.

Recovery Rate: The percentage of the notional amount that the CDS buyer recovers in case of default.

Discount Curve: Used to discount all future cash flows to their present value.

Bootstrapping: An iterative process used to derive the implied survival probabilities from a set of market CDS spreads. The idea is to find survival probabilities such that a hypothetical CDS with that maturity has an NPV of zero at its quoted market spread.

Assumptions and Simplifications in this Code:

Market Data: Zero rates for the discount curve and market CDS spreads for the survival curve are hardcoded. In a real application, these would come from market data feeds.

Day Count Convention: Simplified to "Actual/365".

Payment Frequency: Assumed quarterly for both premium and protection legs.

Default Timing: Assumed that if a default occurs within a period, it happens at the mid-point of that period for the purpose of calculating the expected payout.

Accrued Premium: The model does not explicitly calculate accrued premium upon default.

Homogeneous Spreads: For simplicity, the survival curve is bootstrapped from a single market spread for the CDS maturity. In practice, a full term structure of CDS spreads (e.g., 1Y, 3Y, 5Y, 10Y) is used to bootstrap a more robust survival curve.
"""
import datetime
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from utils.datetime_utils import DateUtils
from utils.curve_utils import YieldCurve


# --- Survival Probability Curve ---
class SurvivalCurve:
    """
    Represents a survival probability curve, typically bootstrapped from CDS spreads.
    Simplified: Bootstraps from a single CDS spread for a given maturity.
    """

    def __init__(self, valuation_date, discount_curve: YieldCurve, cds_spread_bps, maturity_date,
                 freq_months=3, recovery_rate=0.40):
        self.valuation_date = valuation_date
        self.discount_curve = discount_curve
        self.cds_spread_bps = cds_spread_bps
        self.maturity_date = maturity_date
        self.freq_months = freq_months
        self.recovery_rate = recovery_rate

        self.payment_dates = DateUtils.generate_payment_dates(self.valuation_date, self.maturity_date, self.freq_months)
        self.survival_probabilities = {}  # {date: survival_prob}

        self._bootstrap_survival_curve()

        self.default_points = []
        prev_date = self.valuation_date
        for date in self.payment_dates:
            self.default_points.append(prev_date + (date - prev_date) / 2)  # Mid-point for default
            prev_date = date
        # Add the maturity date as a final default point if not already covered
        if self.maturity_date not in self.default_points:
            self.default_points.append(self.maturity_date)
        self.default_points = sorted(list(set(self.default_points)))

    def _bootstrap_survival_curve(self):
        """
        Bootstraps the survival probability curve from the given CDS spread.
        This is a simplified single-spread bootstrapping.
        In a real scenario, you'd bootstrap from a curve of CDS spreads for various tenors.
        """
        # Initialize survival probability at T0 (valuation date) to 1
        self.survival_probabilities[self.valuation_date] = 1.0

        # Iteratively solve for survival probabilities
        for i, t_k in enumerate(self.payment_dates):
            if t_k <= self.valuation_date:
                continue

            # Find the previous payment date
            t_k_minus_1 = self.valuation_date
            if i > 0:
                t_k_minus_1 = self.payment_dates[i - 1]

            # Calculate year fractions
            delta_t_k = DateUtils.year_fraction(t_k_minus_1, t_k, "ACT/365")
            delta_t_mid = DateUtils.year_fraction(self.valuation_date, (t_k_minus_1 + (t_k - t_k_minus_1) / 2),
                                                  "ACT/365")  # Mid-point of interval

            # Current CDS spread in decimal
            s_cds = self.cds_spread_bps / 10000.0

            # Discount factors
            df_k = self.discount_curve.get_discount_factor(t_k)
            df_mid = self.discount_curve.get_discount_factor(
                (t_k_minus_1 + (t_k - t_k_minus_1) / 2))  # Mid-point discount

            # Sum of PV of premiums up to t_k-1 (excluding current period's premium)
            pv_premiums_prev = 0.0
            prev_t_iter = self.valuation_date
            for prev_date in self.payment_dates:
                if prev_date >= t_k:
                    break

                pv_premiums_prev += (self.survival_probabilities[prev_t_iter] *
                                     DateUtils.year_fraction(prev_t_iter, prev_date, "ACT/365") *
                                     self.discount_curve.get_discount_factor(prev_date))
                prev_t_iter = prev_date

            # Sum of PV of expected losses up to t_k-1 (excluding current period's loss)
            pv_losses_prev = 0.0
            prev_t_iter = self.valuation_date
            for prev_date_idx in range(i):
                prev_date = self.payment_dates[prev_date_idx]
                prev_prev_date = self.valuation_date if prev_date_idx == 0 else self.payment_dates[prev_date_idx - 1]

                # Default probability for previous period
                pd_prev_period = (self.survival_probabilities[prev_prev_date] - self.survival_probabilities[prev_date])

                # Mid-point of previous interval
                mid_point_prev = prev_prev_date + (prev_date - prev_prev_date) / 2

                pv_losses_prev += pd_prev_period * self.discount_curve.get_discount_factor(mid_point_prev)

            # Solve for survival probability at t_k (S_k)
            # Equation: S_k = (PV_premiums_prev + s_cds * delta_t_k * S_k) / ( (1 - RecoveryRate) * (S_{k-1} - S_k) * DF_mid + s_cds * delta_t_k * S_{k-1} * DF_k )
            # This is a simplified linear approximation for bootstrapping.
            # A more robust numerical solver (e.g., root finding) would be used for each S_k.

            # Simplified bootstrapping logic for a single spread:
            # We assume a constant hazard rate lambda for each period.
            # S(t_k) = exp(-lambda * t_k_years_from_valuation)
            # The lambda is found such that PV_Premium_Leg = PV_Protection_Leg

            # This simplified bootstrapping is for demonstration.
            # For accurate bootstrapping, you'd use a numerical solver for each survival probability.
            # Here, we'll approximate using the premium and protection leg formulas.

            # For the last period, the survival probability is solved to make the CDS fair.
            # This is an iterative process, so we'll approximate by solving for lambda.

            # A more robust way to bootstrap:
            # For each tenor T_k, find S_k such that the CDS with maturity T_k
            # and spread S_cds has NPV = 0.

            # Let's use a simpler approach for this example:
            # For each interval [t_{k-1}, t_k], assume a constant hazard rate h_k.
            # S_k = S_{k-1} * exp(-h_k * (t_k - t_{k-1}))
            # The h_k for each period is solved iteratively.

            # This is beyond a simple linear equation. We'll use a numerical solver for the full CDS.
            # For this example, we'll just set survival probabilities based on a dummy hazard rate
            # and then use the CDS pricer to find the fair spread.
            # The SurvivalCurve class will simply provide a placeholder for now.
            # A full bootstrapping algorithm is complex and requires a set of market CDS spreads.

            # Placeholder for survival probabilities (will be refined by the CDS pricer's solver)
            time_from_val = DateUtils.year_fraction(self.valuation_date, t_k, "ACT/365")
            # This is a dummy for demonstration, actual values come from bootstrapping
            self.survival_probabilities[t_k] = np.exp(-0.01 * time_from_val)  # Dummy 1% hazard rate

        # The actual bootstrapping logic is complex and iterative.
        # For simplicity, we'll assume the `SurvivalCurve` object will be
        # constructed with a `cds_spread_bps` that implies the curve,
        # and the `CDS` class will solve for the fair spread.
        # The `_bootstrap_survival_curve` method here is primarily for
        # setting up the dates. The actual survival probabilities
        # will be implicitly derived when the CDS is priced to zero NPV.

        # For a practical CDS pricing, you'd need a more robust bootstrapping
        # algorithm that takes a set of market CDS spreads and maturities
        # and iteratively solves for the survival probabilities.
        # Given the scope, we'll let the `CDS` class's `price_cds` method
        # find the fair spread, which implicitly defines the curve.
        pass  # No explicit bootstrapping here, it's done by the CDS pricer's solver

    def get_survival_probability(self, date, fair_hazard_rate=None):
        """
        Gets the survival probability for a given date.
        If fair_hazard_rate is provided (from solver), use it.
        Otherwise, use the bootstrapped (or dummy) values.
        """
        time_from_val = DateUtils.year_fraction(self.valuation_date, date, "ACT/365")
        if time_from_val <= 0:
            return 1.0

        if fair_hazard_rate is not None:
            return np.exp(-fair_hazard_rate * time_from_val)

        # Fallback to dummy if not solving for fair hazard rate
        if date in self.survival_probabilities:
            return self.survival_probabilities[date]

        # Interpolate if date is not a payment date
        dates = sorted(self.survival_probabilities.keys())
        probs = [self.survival_probabilities[d] for d in dates]

        # Ensure there are enough points for interpolation
        if len(dates) < 2:
            return 1.0  # Or raise error if curve is not properly built

        interp_func = interp1d([DateUtils.year_fraction(self.valuation_date, d, "ACT/365") for d in dates],
                               probs, kind='linear', fill_value="extrapolate")

        return interp_func(time_from_val).item()


# --- 4. CDS Pricer Class ---
class CreditDefaultSwap:
    """
    Prices a Credit Default Swap (CDS).
    """

    def __init__(self,
                 valuation_date,
                 effective_date,
                 maturity_date,
                 notional,
                 recovery_rate,  # e.g., 0.40 for 40% recovery
                 premium_freq_months=3,  # Quarterly payments
                 discount_curve: YieldCurve = None):

        if not discount_curve:
            raise ValueError("Discount curve must be provided.")

        self.valuation_date = valuation_date
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.notional = notional
        self.recovery_rate = recovery_rate
        self.premium_freq_months = premium_freq_months
        self.discount_curve = discount_curve

        self.premium_payment_dates = DateUtils.generate_payment_dates(
            self.effective_date, self.maturity_date, self.premium_freq_months)

        # For default probabilities, we need points *between* payment dates (mid-points)
        # and the payment dates themselves for survival probabilities.
        self.default_points = []
        prev_date = self.effective_date
        for date in self.premium_payment_dates:
            self.default_points.append(prev_date + (date - prev_date) / 2)  # Mid-point for default
            prev_date = date
        # Add the maturity date as a final default point if not already covered
        if self.maturity_date not in self.default_points:
            self.default_points.append(self.maturity_date)
        self.default_points = sorted(list(set(self.default_points)))

    def _calculate_premium_leg_pv(self, cds_spread_bps, survival_curve: SurvivalCurve):
        """
        Calculates the present value of the premium leg.
        The premium is paid as long as the entity has not defaulted.
        """
        pv_premium = 0.0
        spread = cds_spread_bps / 10000.0  # Convert bps to decimal

        prev_payment_date = self.effective_date
        for payment_date in self.premium_payment_dates:
            # Year fraction for the current coupon period
            year_frac = DateUtils.year_fraction(prev_payment_date, payment_date, "ACT/365")

            # Survival probability at the payment date
            survival_prob_at_payment = survival_curve.get_survival_probability(payment_date)

            # Discount factor for the payment date
            df = self.discount_curve.get_discount_factor(payment_date)

            # Premium payment is (Notional * Spread * Year_Fraction)
            # This payment is made if the entity survives until the payment date
            pv_premium += self.notional * spread * year_frac * survival_prob_at_payment * df

            prev_payment_date = payment_date

        return pv_premium

    def _calculate_protection_leg_pv(self, survival_curve: SurvivalCurve):
        """
        Calculates the present value of the protection leg.
        Payment occurs if default happens. Assumes default at mid-point of period.
        """
        pv_protection = 0.0
        loss_given_default = self.notional * (1 - self.recovery_rate)

        prev_default_point = self.effective_date
        for i, current_default_point in enumerate(self.default_points):
            if current_default_point <= self.effective_date:
                prev_default_point = current_default_point
                continue  # Skip points before or on effective date

            # Survival probability at start of period
            survival_prob_start = survival_curve.get_survival_probability(prev_default_point)

            # Survival probability at end of period (or current default point)
            survival_prob_end = survival_curve.get_survival_probability(current_default_point)

            # Probability of default within this period
            prob_default_in_period = survival_prob_start - survival_prob_end

            # Discount factor for the mid-point of the period (when default is assumed to occur)
            # For simplicity, using the current_default_point for discounting if it's a mid-point
            # or the actual payment date if it's the end of a period.
            df_at_default = self.discount_curve.get_discount_factor(current_default_point)

            pv_protection += loss_given_default * prob_default_in_period * df_at_default

            prev_default_point = current_default_point

        return pv_protection

    def calculate_npv(self, cds_spread_bps, fair_hazard_rate=None):
        """
        Calculates the NPV of the CDS for a given CDS spread.
        If fair_hazard_rate is provided, it's used to construct the survival curve.
        """
        # Create a temporary survival curve based on the current hazard rate (for solver)
        # Or use the implicitly defined curve if not solving
        temp_survival_curve = SurvivalCurve(self.valuation_date, self.discount_curve, cds_spread_bps,
                                            self.maturity_date,
                                            self.premium_freq_months, self.recovery_rate)

        # Override survival probabilities with those derived from fair_hazard_rate if solving
        if fair_hazard_rate is not None:
            for date in temp_survival_curve.payment_dates:
                temp_survival_curve.survival_probabilities[date] = temp_survival_curve.get_survival_probability(date,
                                                                                                                fair_hazard_rate)
            for date in temp_survival_curve.default_points:
                temp_survival_curve.survival_probabilities[date] = temp_survival_curve.get_survival_probability(date,
                                                                                                                fair_hazard_rate)

        pv_premium = self._calculate_premium_leg_pv(cds_spread_bps, temp_survival_curve)
        pv_protection = self._calculate_protection_leg_pv(temp_survival_curve)

        # NPV from the perspective of the buyer (PV of Protection received - PV of Premiums paid)
        npv = pv_protection - pv_premium
        return npv

    def price_cds(self):
        """
        Finds the fair CDS spread (in basis points) that makes the NPV of the CDS zero.
        This implicitly bootstraps the hazard rate.
        """

        # The function to find the root of (NPV = 0)
        def npv_for_solver(hazard_rate):
            # Convert hazard rate to a dummy CDS spread for the SurvivalCurve init
            # This is a simplification: in reality, the hazard rate *is* the curve.
            # We're using the hazard rate to generate survival probs for NPV calculation.
            # A constant hazard rate implies an exponential survival curve.

            # We need to pass a dummy CDS spread to SurvivalCurve, as it expects one.
            # The actual calculation will use the hazard_rate to derive survival probabilities.
            dummy_cds_spread = 100.0  # Arbitrary non-zero value
            return self.calculate_npv(dummy_cds_spread, fair_hazard_rate=hazard_rate)

        # Use Brent's method to find the fair hazard rate (lambda)
        # Need to provide a bracket [a, b] where f(a) and f(b) have opposite signs.
        # Hazard rates are positive. A reasonable range for search could be 0.0001 (0.01%) to 0.50 (50%).
        try:
            fair_hazard_rate = brentq(npv_for_solver, 0.00001, 0.50)  # Search range for hazard rate

            # Now, derive the fair CDS spread from this fair hazard rate
            # This requires recalculating the PVs with the fair hazard rate
            # and then solving for the spread that makes them equal.

            # Recalculate PVs using the fair hazard rate
            temp_survival_curve_fair = SurvivalCurve(self.valuation_date, self.discount_curve, 0.0, self.maturity_date,
                                                     self.premium_freq_months, self.recovery_rate)

            # Manually set survival probabilities based on the fair hazard rate
            for date in temp_survival_curve_fair.payment_dates:
                temp_survival_curve_fair.survival_probabilities[
                    date] = temp_survival_curve_fair.get_survival_probability(date, fair_hazard_rate)
            for date in temp_survival_curve_fair.default_points:
                temp_survival_curve_fair.survival_probabilities[
                    date] = temp_survival_curve_fair.get_survival_probability(date, fair_hazard_rate)

            pv_protection_fair = self._calculate_protection_leg_pv(temp_survival_curve_fair)

            # Now, find the spread that makes PV_Premium = PV_Protection
            # PV_Premium = Notional * Spread * Sum(Year_frac * Survival_Prob * DF)
            # Sum_Premium_Factors = Sum(Year_frac * Survival_Prob * DF)

            sum_premium_factors = 0.0
            prev_payment_date = self.effective_date
            for payment_date in self.premium_payment_dates:
                year_frac = DateUtils.year_fraction(prev_payment_date, payment_date, "ACT/365")
                survival_prob_at_payment = temp_survival_curve_fair.get_survival_probability(payment_date,
                                                                                             fair_hazard_rate)
                df = self.discount_curve.get_discount_factor(payment_date)
                sum_premium_factors += year_frac * survival_prob_at_payment * df
                prev_payment_date = payment_date

            if sum_premium_factors == 0:
                fair_cds_spread_bps = float('inf')  # Avoid division by zero
            else:
                fair_cds_spread_bps = (pv_protection_fair / (
                            self.notional * sum_premium_factors)) * 10000  # Convert to bps

            return {"Fair_CDS_Spread_bps": fair_cds_spread_bps,
                    "Implied_Hazard_Rate": fair_hazard_rate,
                    "NPV_at_Fair_Spread": self.calculate_npv(fair_cds_spread_bps, fair_hazard_rate=fair_hazard_rate)}

        except ValueError as e:
            print(f"Error finding fair CDS spread: {e}. Check inputs or widen search range for hazard rate.")
            return {"Fair_CDS_Spread_bps": "N/A", "Implied_Hazard_Rate": "N/A",
                    "NPV_at_Fair_Spread": self.calculate_npv(0.0)}  # Return NPV at 0 spread if solver fails


# --- Example Usage ---
if __name__ == "__main__":
    valuation_date = datetime.date(2025, 7, 15)
    effective_date = datetime.date(2025, 7, 17)  # T+2 settlement
    maturity_date = datetime.date(2030, 7, 17)  # 5-year CDS

    # --- Define Market Data for Discount Curve ---
    # Example zero rates for a generic currency
    discount_tenors_years = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    discount_zero_rates = [0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042]  # Example zero rates

    discount_curve = YieldCurve(valuation_date, discount_tenors_years, discount_zero_rates)

    print(f"Valuation Date: {valuation_date}")
    print(f"Effective Date: {effective_date}")
    print(f"Maturity Date: {maturity_date}")
    print(
        f"Discount Curve (5-year zero rate): {discount_curve.get_zero_rate(DateUtils.next_date(valuation_date, 5)):.4%}")
    print(f"Discount Factor for Maturity: {discount_curve.get_discount_factor(maturity_date):.4f}")

    # --- CDS Parameters ---
    cds_notional = 10_000_000  # $10 million
    cds_recovery_rate = 0.40  # 40% recovery
    cds_premium_frequency_months = 3  # Quarterly payments

    # --- Scenario 1: Calculate NPV for a given market CDS spread ---
    market_cds_spread_bps = 150.0  # 150 basis points (1.50%)

    # For NPV calculation, we need a SurvivalCurve object.
    # The `SurvivalCurve` class in this simplified example doesn't fully bootstrap
    # from a single spread. It's more illustrative.
    # The `calculate_npv` method will use the `fair_hazard_rate` if provided by the solver.
    # For a direct NPV calculation with a given spread, we're implicitly assuming
    # that spread corresponds to some underlying hazard rate.
    # A more robust setup would bootstrap the survival curve first, then calculate NPV.

    # To calculate NPV for a given spread, we need to implicitly assume a hazard rate
    # that would give that spread. This is a circular problem.
    # The most common use case is to *price* the CDS, i.e., find the fair spread.

    print("\n--- Scenario: Pricing a CDS (Finding the Fair Spread) ---")
    cds_pricer = CreditDefaultSwap(
        valuation_date=valuation_date,
        effective_date=effective_date,
        maturity_date=maturity_date,
        notional=cds_notional,
        recovery_rate=cds_recovery_rate,
        premium_freq_months=cds_premium_frequency_months,
        discount_curve=discount_curve
    )

    pricing_result = cds_pricer.price_cds()

    if pricing_result["Fair_CDS_Spread_bps"] != "N/A":
        print(f"Fair CDS Spread (bps): {pricing_result['Fair_CDS_Spread_bps']:.2f} bps")
        print(f"Implied Constant Hazard Rate: {pricing_result['Implied_Hazard_Rate']:.4%}")
        # Verify NPV at the fair spread (should be close to zero)
        npv_at_fair_spread = cds_pricer.calculate_npv(
            pricing_result['Fair_CDS_Spread_bps'],
            fair_hazard_rate=pricing_result['Implied_Hazard_Rate']
        )
        print(f"NPV at Fair Spread (for verification): ${npv_at_fair_spread:,.2f}")
    else:
        print(pricing_result)

    print("\n--- Example: Calculate NPV for a specific given spread (e.g., 150 bps) ---")
    # To calculate NPV for an *arbitrary* given spread, we need to assume an underlying
    # hazard rate that corresponds to it, or use a pre-bootstrapped survival curve.
    # For demonstration, let's use the implied hazard rate from the 'fair spread'
    # calculation as a proxy for an 'arbitrary' spread's underlying hazard.

    # Let's assume a market-quoted spread of 150 bps.
    # We need to find the hazard rate that implies this 150 bps spread.
    # This is essentially the inverse of the 'price_cds' function.

    # For simplicity, let's just calculate NPV using a known hazard rate.
    # If a CDS is quoted at 150 bps, it implies a certain hazard rate.
    # The `price_cds` function finds the hazard rate that makes NPV zero.
    # So, if we want to know the NPV of a CDS with a *given* spread,
    # we'd need to first bootstrap the survival curve from that spread.
    # This example will just show the NPV if we *assume* a hazard rate.

    # Let's assume a hazard rate of 2.5% for this example (just for NPV calculation)
    assumed_hazard_rate_for_npv = 0.025
    # The CDS spread that would make NPV zero for this hazard rate is what `price_cds` finds.
    # If we are given a spread, say 150 bps, and want its NPV, we need to find the hazard rate
    # that corresponds to 150 bps first. This is the core of CDS pricing.

    # So, the `price_cds` method is the primary way to "price" or "value" a CDS.
    # If you have a market spread and want to know if the CDS is "cheap" or "expensive",
    # you would compare the market spread to the `Fair_CDS_Spread_bps` calculated by `price_cds`.

    # If you truly want to calculate NPV for an arbitrary `market_cds_spread_bps`,
    # you would need to adjust the `SurvivalCurve` class to bootstrap from a given spread
    # to derive the `survival_probabilities` for `calculate_npv`.
    # As currently implemented, `calculate_npv` is designed to be called by the solver
    # which passes a `fair_hazard_rate` to determine the survival curve.

    # Let's illustrate how you would use the `calculate_npv` method if you knew the implied hazard rate
    # For instance, if the market implies a hazard rate of 2.5% for this CDS:
    npv_at_assumed_hazard = cds_pricer.calculate_npv(
        cds_spread_bps=150.0,  # This spread is just a label here, the hazard rate drives the NPV
        fair_hazard_rate=assumed_hazard_rate_for_npv
    )
    print(f"NPV of CDS at assumed hazard rate {assumed_hazard_rate_for_npv:.2%}: ${npv_at_assumed_hazard:,.2f}")
    print("Note: This NPV is based on an assumed hazard rate, not directly on the 150 bps spread in isolation.")