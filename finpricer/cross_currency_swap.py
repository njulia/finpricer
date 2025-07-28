"""
This cross-currency interest rate swap pricing model calculates the Net Present Value (NPV) from the perspective of the domestic party (paying fixed USD, receiving floating EUR). The total NPV is the sum of the present values of all cash flows, including both interest payments and principal exchanges, all converted to the domestic currency (USD).

1. **Interest Payments - Fixed-Rate Leg (USD - Paying):**

* For each semi-annual period over the 5-year maturity, a fixed interest payment is calculated as: `Fixed Rate * Domestic Notional / Payments per Year`.

* Each of these future payments is discounted back to today's present value using the `Domestic Curve` as the discount rate. The discount factor for a period `t` is `1 / (1 + Domestic_Curve / Payments_per_Year)^t`.

* Since the domestic party is *paying* these amounts, their present value contributes negatively to the total swap NPV.

2. **Interest Payments - Floating-Rate Leg (EUR - Receiving):**

* For each semi-annual period, a floating interest payment is calculated. Given the flat curve assumption, the forward rate for all periods is simply the `Foreign Curve` rate. So, the payment is: `(Foreign Curve + Floating Spread) * Foreign Notional / Payments per Year`.

* Each of these future payments (in EUR) is discounted back to today's present value using the `Foreign Curve` as the discount rate.

* The total present value of these EUR interest receipts is then converted to USD using the `Spot FX Rate` (`EUR_PV * Spot_FX_Rate`).

* Since the domestic party is *receiving* these amounts, their present value contributes positively to the total swap NPV.

3. **Principal Exchanges:**

* **Initial Exchange (at T=0):** At the start of the swap, the domestic party *pays* the `Domestic Notional` in USD and *receives* the `Foreign Notional` in EUR. The received EUR notional is immediately converted to USD using the `Spot FX Rate`. The net effect is `(-Domestic Notional) + (Foreign Notional * Spot FX Rate)`. This contribution is at time zero, so no discounting is applied to this component itself.

* **Final Exchange (at Maturity):** At the end of the swap (5 years), the domestic party *receives* the `Domestic Notional` in USD and *pays* the `Foreign Notional` in EUR. Both of these amounts are discounted back to today's present value using their respective curves (`Domestic Curve` for USD, `Foreign Curve` for EUR). The discounted EUR notional paid is then converted to USD using the `Spot FX Rate`. The net effect is `(Discounted Domestic Notional) - (Discounted Foreign Notional * Spot FX Rate)`.

4. **Total NPV Calculation:**

* The total NPV of the cross-currency swap in USD is the sum of the initial principal exchange, the negative PV of the fixed leg interest payments, the positive PV of the floating leg interest payments (converted to USD), and the net PV of the final principal exchange (converted to USD).
"""
import datetime
from scipy.optimize import brentq  # For finding the fair fixed rate or basis spread
from utils.datetime_utils import DateUtils
from utils.curve_utils import YieldCurve


# --- 3. Cross-Currency Swap Pricer ---
class CrossCurrencySwap:
    """
    Prices a Cross-Currency Interest Rate Swap (CCIRS).
    Supports Fixed-for-Floating and Floating-for-Floating (Basis Swap).
    """

    def __init__(self,
                 valuation_date,
                 effective_date,
                 maturity_date,
                 currency1_notional,
                 currency2_notional,
                 currency1_fixed_rate=None,  # If fixed leg
                 currency2_fixed_rate=None,  # If fixed leg
                 currency1_floating_index=None,  # If floating leg (e.g., 'SOFR')
                 currency2_floating_index=None,  # If floating leg (e.g., 'EURIBOR')
                 currency1_freq_months=6,  # Payment frequency in months
                 currency2_freq_months=6,
                 spot_fx_rate=1.0,  # C1 per unit of C2 (e.g., USD/EUR = 1.10 means 1.10 USD per 1 EUR)
                 currency1_curve: YieldCurve = None,
                 currency2_curve: YieldCurve = None,
                 currency2_basis_spread_bps=0.0,  # Basis spread in basis points, applied to C2 floating leg
                 day_count_convention="ACT/365"):

        if not (currency1_curve and currency2_curve):
            raise ValueError("Both currency1_curve and currency2_curve must be provided.")

        self.valuation_date = valuation_date
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.currency1_notional = currency1_notional
        self.currency2_notional = currency2_notional
        self.currency1_fixed_rate = currency1_fixed_rate
        self.currency2_fixed_rate = currency2_fixed_rate
        self.currency1_floating_index = currency1_floating_index
        self.currency2_floating_index = currency2_floating_index
        self.currency1_freq_months = currency1_freq_months
        self.currency2_freq_months = currency2_freq_months
        self.spot_fx_rate = spot_fx_rate  # C1 per unit of C2
        self.currency1_curve = currency1_curve
        self.currency2_curve = currency2_curve
        self.currency2_basis_spread_bps = currency2_basis_spread_bps
        self.day_count_convention = day_count_convention

        # Determine swap type
        self.is_fixed_fixed = (currency1_fixed_rate is not None and currency2_fixed_rate is not None)
        self.is_fixed_floating = (currency1_fixed_rate is not None and currency2_floating_index is not None) or \
                                 (currency2_fixed_rate is not None and currency1_floating_index is not None)
        self.is_floating_floating = (currency1_floating_index is not None and currency2_floating_index is not None)

        if not (self.is_fixed_fixed or self.is_fixed_floating or self.is_floating_floating):
            raise ValueError("Swap type (fixed-fixed, fixed-floating, floating-floating) not clearly defined.")

        # Generate payment schedules
        self.c1_payment_dates = DateUtils.generate_payment_dates(self.effective_date, self.maturity_date,
                                                                 self.currency1_freq_months)
        self.c2_payment_dates = DateUtils.generate_payment_dates(self.effective_date, self.maturity_date,
                                                                 self.currency2_freq_months)

    def _calculate_fixed_leg_pv(self, notional, fixed_rate, payment_dates, curve: YieldCurve):
        """Calculates the present value of a fixed leg."""
        pv = 0.0
        prev_date = self.effective_date
        for date in payment_dates:
            year_frac = DateUtils.year_fraction(prev_date, date, self.day_count_convention)
            coupon = notional * fixed_rate * year_frac
            pv += coupon * curve.get_discount_factor(date)
            prev_date = date
        # Add present value of final principal exchange
        pv += notional * curve.get_discount_factor(self.maturity_date)
        return pv

    def _calculate_floating_leg_pv(self, notional, payment_dates, curve: YieldCurve, basis_spread_bps=0.0):
        """Calculates the present value of a floating leg."""
        pv = 0.0
        prev_date = self.effective_date
        basis_spread = basis_spread_bps / 10000.0  # Convert bps to decimal

        for date in payment_dates:
            year_frac = DateUtils.year_fraction(prev_date, date, self.day_count_convention)
            # Project forward rate
            # If the payment date is today, the rate is often the current spot rate, not a forward
            # For simplicity, we use forward rate from curve for all periods
            projected_rate = curve.get_forward_rate(prev_date, date, self.day_count_convention) + basis_spread

            coupon = notional * projected_rate * year_frac
            pv += coupon * curve.get_discount_factor(date)
            prev_date = date
        # Add present value of final principal exchange
        pv += notional * curve.get_discount_factor(self.maturity_date)
        return pv

    def calculate_npv(self, currency1_fixed_rate_override=None, currency2_basis_spread_bps_override=None):
        """
        Calculates the Net Present Value (NPV) of the Cross-Currency Swap.
        Allows overriding fixed rate or basis spread for fair value calculations.
        NPV is from the perspective of receiving C2 cash flows and paying C1 cash flows.
        """
        c1_rate = currency1_fixed_rate_override if currency1_fixed_rate_override is not None else self.currency1_fixed_rate
        c2_basis = currency2_basis_spread_bps_override if currency2_basis_spread_bps_override is not None else self.currency2_basis_spread_bps

        # Calculate PV of Currency 1 Leg (paying leg from perspective of receiving C2)
        if self.currency1_fixed_rate is not None:
            pv_c1_leg = self._calculate_fixed_leg_pv(self.currency1_notional, c1_rate,
                                                     self.c1_payment_dates, self.currency1_curve)
        elif self.currency1_floating_index is not None:
            # For simplicity, assuming C1 floating leg does NOT have basis spread applied
            pv_c1_leg = self._calculate_floating_leg_pv(self.currency1_notional,
                                                        self.c1_payment_dates, self.currency1_curve,
                                                        basis_spread_bps=0.0)
        else:
            raise ValueError("Currency 1 leg type not defined.")

        # Calculate PV of Currency 2 Leg (receiving leg)
        if self.currency2_fixed_rate is not None:
            pv_c2_leg = self._calculate_fixed_leg_pv(self.currency2_notional, self.currency2_fixed_rate,
                                                     self.c2_payment_dates, self.currency2_curve)
        elif self.currency2_floating_index is not None:
            # Apply basis spread to C2 floating leg
            pv_c2_leg = self._calculate_floating_leg_pv(self.currency2_notional,
                                                        self.c2_payment_dates, self.currency2_curve,
                                                        basis_spread_bps=c2_basis)
        else:
            raise ValueError("Currency 2 leg type not defined.")

        # Initial principal exchange adjustment (if notional amounts are not already aligned)
        # Assuming notionals are already set at the spot rate for a fair swap at inception
        # If the swap is off-market, then the initial exchange might be adjusted.
        # For fair pricing, PV of C1 leg should equal PV of C2 leg * Spot FX.
        # NPV = PV(Receive C2) - PV(Pay C1)
        # Convert PV of C2 leg to C1 currency using spot FX
        npv = (pv_c2_leg * self.spot_fx_rate) - pv_c1_leg
        return npv

    def price_swap(self):
        """
        Prices the swap by finding the fair fixed rate or basis spread that results in NPV = 0.
        """
        if self.is_fixed_floating:
            # Assume C1 is fixed, C2 is floating with basis spread
            if self.currency1_fixed_rate is not None and self.currency2_floating_index is not None:
                # Find the fair fixed rate for C1 that makes NPV zero
                def npv_func(fixed_rate):
                    return self.calculate_npv(currency1_fixed_rate_override=fixed_rate)

                # Use Brent's method to find the root (fair fixed rate)
                # Need to provide a bracket [a, b] where f(a) and f(b) have opposite signs
                # This often requires some intuition or a wider search.
                # For typical rates, 0.0% to 10.0% should cover most cases.
                try:
                    fair_fixed_rate = brentq(npv_func, -0.10, 0.10)  # Search range for fixed rate (-10% to 10%)
                    return {"Fair_Fixed_Rate_C1": fair_fixed_rate, "NPV_at_Fair_Rate": 0.0}
                except ValueError:
                    print(
                        "Warning: Could not find a fair fixed rate within the given range. Check inputs or widen range.")
                    return {"Fair_Fixed_Rate_C1": "N/A", "NPV_at_Fair_Rate": self.calculate_npv()}

            # Assume C2 is fixed, C1 is floating (less common for basis spread application)
            elif self.currency2_fixed_rate is not None and self.currency1_floating_index is not None:
                # This scenario would typically involve a basis spread on C1, but our model applies to C2
                # For simplicity, we'll assume the fixed rate for C2 is given and we're just calculating NPV
                # Or, if we were to solve for C2 fixed rate, it would be similar to above.
                print("Pricing for C2 fixed leg is not implemented as a solver target in this model.")
                return {"NPV": self.calculate_npv()}

        elif self.is_floating_floating:
            # Find the fair basis spread for C2 that makes NPV zero
            def npv_func(basis_spread_bps):
                return self.calculate_npv(currency2_basis_spread_bps_override=basis_spread_bps)

            try:
                # Search range for basis spread (e.g., -100 bps to +100 bps)
                fair_basis_spread_bps = brentq(npv_func, -100.0, 100.0)
                return {"Fair_Basis_Spread_C2_bps": fair_basis_spread_bps, "NPV_at_Fair_Spread": 0.0}
            except ValueError:
                print(
                    "Warning: Could not find a fair basis spread within the given range. Check inputs or widen range.")
                return {"Fair_Basis_Spread_C2_bps": "N/A", "NPV_at_Fair_Spread": self.calculate_npv()}

        elif self.is_fixed_fixed:
            # For fixed-fixed, the rates are usually given. We just calculate NPV.
            # If one fixed rate is unknown, it would be solved for.
            print("Fixed-Fixed swap: Calculating NPV with given rates.")
            return {"NPV": self.calculate_npv()}
        else:
            raise ValueError("Invalid swap type for pricing.")


# --- Example Usage ---
if __name__ == "__main__":
    valuation_date = datetime.date(2025, 7, 15)
    effective_date = datetime.date(2025, 7, 17)  # T+2 settlement
    maturity_date = datetime.date(2030, 7, 17)  # 5-year swap

    # --- Define Market Data ---
    # Currency 1 (e.g., USD) Yield Curve Data
    usd_tenors_years = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    usd_zero_rates = [0.050, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.060]  # Example zero rates

    # Currency 2 (e.g., EUR) Yield Curve Data
    eur_tenors_years = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    eur_zero_rates = [0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040]  # Example zero rates

    # Spot FX Rate (USD per EUR, e.g., 1.08 USD for 1 EUR)
    usd_eur_spot_fx = 1.08

    # Create Yield Curve objects
    usd_curve = YieldCurve(valuation_date, usd_tenors_years, usd_zero_rates)
    eur_curve = YieldCurve(valuation_date, eur_tenors_years, eur_zero_rates)

    print(f"Valuation Date: {valuation_date}")
    print(f"Effective Date: {effective_date}")
    print(f"Maturity Date: {maturity_date}")
    print(f"Spot USD/EUR FX Rate: {usd_eur_spot_fx}")
    print("\n--- Example Yield Curve Data ---")
    print(f"USD 5-year zero rate: {usd_curve.get_zero_rate(DateUtils.next_date(valuation_date, 5)):.4f}")
    print(f"EUR 5-year zero rate: {eur_curve.get_zero_rate(DateUtils.next_date(valuation_date, 5)):.4f}")
    print(f"USD DF for maturity: {usd_curve.get_discount_factor(maturity_date):.4f}")
    print(f"EUR DF for maturity: {eur_curve.get_discount_factor(maturity_date):.4f}")

    # --- Scenario 1: Fixed-for-Floating Cross-Currency Swap ---
    # Party A (USD payer, EUR receiver) pays fixed USD, receives floating EUR
    print("\n--- Scenario 1: Fixed-for-Floating (Pay Fixed USD, Receive Floating EUR) ---")
    usd_notional = 10_000_000  # USD 10 million
    eur_notional = usd_notional / usd_eur_spot_fx  # Equivalent EUR notional at spot

    # Assume a market fixed rate for USD (e.g., 4.5%)
    # And a basis spread for EUR (e.g., -10 bps, meaning EUR funding is cheaper than implied by interest rates)
    given_usd_fixed_rate = 0.045
    given_eur_basis_bps = -10.0

    ccirs_fixed_floating = CrossCurrencySwap(
        valuation_date=valuation_date,
        effective_date=effective_date,
        maturity_date=maturity_date,
        currency1_notional=usd_notional,
        currency2_notional=eur_notional,
        currency1_fixed_rate=given_usd_fixed_rate,  # USD leg is fixed
        currency2_floating_index='EURIBOR',  # EUR leg is floating
        currency1_freq_months=6,
        currency2_freq_months=3,  # EURIBOR typically 3-month
        spot_fx_rate=usd_eur_spot_fx,
        currency1_curve=usd_curve,
        currency2_curve=eur_curve,
        currency2_basis_spread_bps=given_eur_basis_bps
    )

    npv_fixed_floating = ccirs_fixed_floating.calculate_npv()
    print(f"NPV of Fixed-for-Floating Swap (given rates): ${npv_fixed_floating:,.2f} (in USD)")

    # Price the swap: find the fair fixed rate for USD that makes NPV zero
    pricing_result_fixed_floating = ccirs_fixed_floating.price_swap()
    if "Fair_Fixed_Rate_C1" in pricing_result_fixed_floating:
        print(
            f"Fair Fixed Rate for USD Leg (to make NPV zero): {pricing_result_fixed_floating['Fair_Fixed_Rate_C1']:.4%} (NPV: ${pricing_result_fixed_floating['NPV_at_Fair_Rate']:,.2f})")
    else:
        print(pricing_result_fixed_floating)

    # --- Scenario 2: Floating-for-Floating Cross-Currency Swap (Basis Swap) ---
    # Party A (USD payer, EUR receiver) pays floating USD, receives floating EUR with basis
    print("\n--- Scenario 2: Floating-for-Floating (Pay Floating USD, Receive Floating EUR with Basis) ---")
    usd_notional_basis = 5_000_000
    eur_notional_basis = usd_notional_basis / usd_eur_spot_fx

    # Assume a market basis spread for EUR (e.g., -15 bps)
    given_eur_basis_bps_basis = -15.0

    ccirs_floating_floating = CrossCurrencySwap(
        valuation_date=valuation_date,
        effective_date=effective_date,
        maturity_date=maturity_date,
        currency1_notional=usd_notional_basis,
        currency2_notional=eur_notional_basis,
        currency1_floating_index='SOFR',  # USD leg is floating
        currency2_floating_index='EURIBOR',  # EUR leg is floating
        currency1_freq_months=3,
        currency2_freq_months=3,
        spot_fx_rate=usd_eur_spot_fx,
        currency1_curve=usd_curve,
        currency2_curve=eur_curve,
        currency2_basis_spread_bps=given_eur_basis_bps_basis  # This is the basis applied to EUR leg
    )

    npv_floating_floating = ccirs_floating_floating.calculate_npv()
    print(f"NPV of Floating-for-Floating Swap (given basis): ${npv_floating_floating:,.2f} (in USD)")

    # Price the swap: find the fair basis spread for EUR that makes NPV zero
    pricing_result_floating_floating = ccirs_floating_floating.price_swap()
    if "Fair_Basis_Spread_C2_bps" in pricing_result_floating_floating:
        print(
            f"Fair Basis Spread for EUR Leg (to make NPV zero): {pricing_result_floating_floating['Fair_Basis_Spread_C2_bps']:.2f} bps (NPV: ${pricing_result_floating_floating['NPV_at_Fair_Spread']:,.2f})")
    else:
        print(pricing_result_floating_floating)

    # --- Scenario 3: Fixed-for-Fixed Cross-Currency Swap ---
    # Party A (USD payer, EUR receiver) pays fixed USD, receives fixed EUR
    print("\n--- Scenario 3: Fixed-for-Fixed (Pay Fixed USD, Receive Fixed EUR) ---")
    usd_notional_fixed_fixed = 7_500_000
    eur_notional_fixed_fixed = usd_notional_fixed_fixed / usd_eur_spot_fx

    # Given fixed rates for both legs
    given_usd_fixed_rate_ff = 0.040
    given_eur_fixed_rate_ff = 0.025

    ccirs_fixed_fixed = CrossCurrencySwap(
        valuation_date=valuation_date,
        effective_date=effective_date,
        maturity_date=maturity_date,
        currency1_notional=usd_notional_fixed_fixed,
        currency2_notional=eur_notional_fixed_fixed,
        currency1_fixed_rate=given_usd_fixed_rate_ff,
        currency2_fixed_rate=given_eur_fixed_rate_ff,
        currency1_freq_months=6,
        currency2_freq_months=12,
        spot_fx_rate=usd_eur_spot_fx,
        currency1_curve=usd_curve,
        currency2_curve=eur_curve
    )

    npv_fixed_fixed = ccirs_fixed_fixed.calculate_npv()
    print(f"NPV of Fixed-for-Fixed Swap (given rates): ${npv_fixed_fixed:,.2f} (in USD)")
    # For fixed-fixed, we typically calculate NPV given rates, not solve for a fair rate/spread
    # unless one of the fixed rates is left as unknown.
    pricing_result_fixed_fixed = ccirs_fixed_fixed.price_swap()
    print(pricing_result_fixed_fixed)