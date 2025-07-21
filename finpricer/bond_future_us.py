import datetime
import numpy as np


# --- Helper Functions ---

def day_count_actual_actual(start_date, end_date):
    """
    Calculates the day count fraction using Actual/Actual (ISDA) convention.
    This is a simplified version; real Actual/Actual can be complex.
    For US Treasuries, Actual/Actual (ICMA) is common.
    """
    return (end_date - start_date).days / 365.25  # Approximation for simplicity


def calculate_accrued_interest(issue_date, settlement_date, coupon_rate, face_value, coupon_frequency_years=0.5):
    """
    Calculates accrued interest for a bond.
    Assumes semi-annual coupons (coupon_frequency_years = 0.5).
    """
    if settlement_date <= issue_date:
        return 0.0

    # Find the last coupon payment date before or on settlement_date
    # This is a simplification. A robust implementation needs to track actual coupon dates.
    # For simplicity, we'll assume coupons are paid on the same day of the month as issue date,
    # or closest valid date, every `coupon_frequency_years` interval.

    # Find the most recent coupon date prior to or on the settlement_date
    last_coupon_date = issue_date
    while last_coupon_date < settlement_date:
        next_possible_coupon_date = last_coupon_date + datetime.timedelta(days=int(coupon_frequency_years * 365.25))
        if next_possible_coupon_date > settlement_date:
            break
        last_coupon_date = next_possible_coupon_date

    # If no coupon date found before settlement, use issue_date as previous coupon date
    if last_coupon_date == issue_date and settlement_date > issue_date and (settlement_date - issue_date).days < (
            coupon_frequency_years * 365.25):
        # This is the first coupon period
        days_since_last_coupon = (settlement_date - issue_date).days
        days_in_coupon_period = (
                    last_coupon_date + datetime.timedelta(days=int(coupon_frequency_years * 365.25)) - issue_date).days
    else:
        days_since_last_coupon = (settlement_date - last_coupon_date).days
        # Find the next coupon date to determine full period days
        next_coupon_date = last_coupon_date
        while next_coupon_date < settlement_date:
            next_coupon_date += datetime.timedelta(days=int(coupon_frequency_years * 365.25))
        days_in_coupon_period = (next_coupon_date - last_coupon_date).days

    if days_in_coupon_period == 0:  # Avoid division by zero
        return 0.0

    annual_coupon_amount = face_value * coupon_rate
    accrued_interest = annual_coupon_amount * (days_since_last_coupon / days_in_coupon_period)
    return accrued_interest


def calculate_bond_price_from_yield(face_value, coupon_rate, maturity_date, current_date, yield_to_maturity,
                                    coupon_frequency_years=0.5):
    """
    Calculates the clean price of a bond given its yield to maturity.
    Assumes semi-annual compounding for yield.
    """
    num_periods = int((maturity_date - current_date).days / (coupon_frequency_years * 365.25))
    if num_periods < 0:
        return 0.0  # Bond has matured

    coupon_payment = (face_value * coupon_rate) * coupon_frequency_years
    ytm_per_period = yield_to_maturity * coupon_frequency_years

    price = 0.0
    for i in range(1, num_periods + 1):
        price += coupon_payment / ((1 + ytm_per_period) ** i)
    price += face_value / ((1 + ytm_per_period) ** num_periods)

    return price


def calculate_conversion_factor(bond_details, futures_delivery_date, notional_yield=0.06):
    """
    Calculates the conversion factor for a bond for a US Treasury futures contract.
    Based on the bond's price yielding a notional_yield (e.g., 6% for US Treasuries).
    This is a simplified calculation. Actual CME conversion factor calculation is very specific
    about rounding maturities, coupon rates, and day counts.
    """
    face_value = bond_details['face_value']
    coupon_rate = bond_details['coupon_rate']
    maturity_date = bond_details['maturity_date']
    issue_date = bond_details['issue_date']  # Needed for accrued interest

    # Calculate the price of the bond if it yielded the notional_yield
    # We need the clean price at the futures delivery date
    clean_price_at_notional_yield = calculate_bond_price_from_yield(
        face_value, coupon_rate, maturity_date, futures_delivery_date, notional_yield
    )

    # Accrued interest at the futures delivery date
    accrued_at_delivery = calculate_accrued_interest(
        issue_date, futures_delivery_date, coupon_rate, face_value
    )

    # Conversion Factor = (Dirty Price at Notional Yield - Accrued Interest) / Face Value
    # The conversion factor is essentially the clean price of $1 par value of the bond
    # if it yielded 6%.
    # So, (Clean Price at Notional Yield) / Face Value
    conversion_factor = clean_price_at_notional_yield / face_value
    return round(conversion_factor, 4)  # Typically rounded to 4 decimal places


# --- Main Pricing Function ---

def price_us_bond_future(
        valuation_date,
        futures_delivery_date,
        repo_rate,  # Annualized financing rate (e.g., 0.02 for 2%)
        eligible_bonds  # List of dictionaries, each describing a deliverable bond
):
    """
    Prices a US bond future based on the Cheapest-to-Deliver (CTD) bond
    and the Cost-of-Carry model.

    Args:
        valuation_date (datetime.date): The current date for valuation.
        futures_delivery_date (datetime.date): The delivery date of the futures contract.
        repo_rate (float): The annualized financing rate (repo rate).
        eligible_bonds (list): A list of dictionaries, each representing an eligible bond.
                                Each dict must have:
                                'id': str (e.g., 'BondA')
                                'face_value': float (e.g., 1000)
                                'coupon_rate': float (e.g., 0.04 for 4%)
                                'maturity_date': datetime.date
                                'issue_date': datetime.date (for accrued interest)
                                'current_market_price': float (clean price)

    Returns:
        dict: A dictionary containing the theoretical futures price, CTD bond details,
              and intermediate calculations.
    """
    if valuation_date >= futures_delivery_date:
        raise ValueError("Valuation date must be before futures delivery date.")

    cheapest_to_deliver_bond = None
    min_cost_of_delivery = float('inf')

    # Time to delivery in years (Actual/360 for repo)
    time_to_delivery_years = (futures_delivery_date - valuation_date).days / 360.0

    for bond in eligible_bonds:
        bond_id = bond['id']
        face_value = bond['face_value']
        coupon_rate = bond['coupon_rate']
        maturity_date = bond['maturity_date']
        issue_date = bond['issue_date']
        current_market_price_clean = bond['current_market_price']  # This is the clean price

        # 1. Calculate Accrued Interest (AI) on the spot bond
        accrued_interest_spot = calculate_accrued_interest(
            issue_date, valuation_date, coupon_rate, face_value
        )
        current_market_price_dirty = current_market_price_clean + accrued_interest_spot

        # 2. Calculate Conversion Factor (CF) for the bond
        conversion_factor = calculate_conversion_factor(
            bond, futures_delivery_date
        )

        # 3. Calculate Present Value of Coupons (PVCI) between valuation and delivery dates
        # This is a simplification. A robust model would list all coupons.
        # For simplicity, we assume coupons are paid semi-annually.
        # We need to find coupons between valuation_date and futures_delivery_date
        pv_coupons_during_futures_life = 0.0

        # Determine next coupon date after valuation_date
        next_coupon_date = issue_date
        while next_coupon_date <= valuation_date:
            next_coupon_date += datetime.timedelta(days=int(0.5 * 365.25))  # Semi-annual

        # Iterate through coupons until maturity or futures delivery date
        current_coupon_date = next_coupon_date
        while current_coupon_date <= futures_delivery_date and current_coupon_date <= maturity_date:
            coupon_amount = (face_value * coupon_rate) * 0.5  # Semi-annual coupon

            # Discount back to valuation_date
            time_to_coupon_payment_years = (current_coupon_date - valuation_date).days / 360.0
            discount_factor = np.exp(-repo_rate * time_to_coupon_payment_years)  # Using repo rate for discounting

            pv_coupons_during_futures_life += coupon_amount * discount_factor
            current_coupon_date += datetime.timedelta(days=int(0.5 * 365.25))  # Move to next coupon date

        # 4. Calculate Accrued Interest (AI) at the futures delivery date
        accrued_interest_at_delivery = calculate_accrued_interest(
            issue_date, futures_delivery_date, coupon_rate, face_value
        )

        # 5. Calculate the "Cost of Carry" for the bond
        # Cost of Carry = (Dirty Spot Price + Financing Cost) - Coupon Income
        # Financing Cost = Dirty Spot Price * Repo Rate * (Time to Delivery)
        # Coupon Income = PV of Coupons between now and delivery

        # Forward Price of the Bond (Dirty) = Current Dirty Price * exp(repo_rate * time_to_delivery_years) - FVCI (Future Value of Coupons)
        # For discrete compounding: Forward Price = Current Dirty Price * (1 + repo_rate * time_to_delivery_years) - FVCI

        # Let's use the standard arbitrage-free pricing formula for bond futures:
        # F = (S_0 + C - I) / CF
        # Where S_0 = current dirty price, C = PV of coupons received, I = financing cost
        # Or, more commonly, F = (S_0 * (1+r)^(T) - FVCI) / CF
        # Where FVCI is Future Value of Coupons received during futures life.

        # Let's use the standard formula for the theoretical futures price:
        # Futures Price = (Spot Price of CTD + Accrued Interest at Spot - PV of Coupons during futures life) * (1 + Repo Rate * Time to Delivery) / Conversion Factor
        # This is a common simplification, ignoring the exact compounding.

        # More robust: Futures Price = (Dirty Spot Price * (1 + Repo Rate)^(T) - FV of Coupons) / Conversion Factor
        # Where T is time to delivery in years, and FV of Coupons are coupons received and reinvested at repo rate.

        # Let's use the formula: Futures Price = (Forward Price of CTD Clean) / Conversion Factor
        # Forward Price (Dirty) = (Current Dirty Price * (1 + Repo Rate * Days / 360)) - FV of Coupons
        # FV of Coupons = Sum of (Coupon Amount * (1 + Repo Rate * Days_from_coupon_to_delivery / 360))

        # Re-evaluating the cost of delivery for the CTD
        # This is the net cost to the short if they buy the bond today and deliver it.
        # Cost = Current Dirty Price - PV of Coupons during futures life + Financing Cost
        # Or, more practically, find the implied repo rate.

        # The theoretical futures price (F) is such that:
        # F * CF + AI_delivery = (Dirty Spot Price) * (1 + Repo Rate * (Days to Delivery / 360)) - Sum(FV of Coupons during futures life)
        # Where FV of Coupons are future valued to the delivery date.

        # Let's use the standard simplified "arbitrage-free" price:
        # F_0 = (S_0 * (1 + r_repo * T_delivery) - FV_coupons) / CF
        # Where S_0 is the dirty spot price, r_repo is the repo rate, T_delivery is time to delivery
        # FV_coupons is the future value of coupons received during the futures contract's life.

        # Calculate Future Value of Coupons (FVCI)
        fv_coupons_during_futures_life = 0.0

        current_coupon_date_fv = next_coupon_date  # Already calculated from above
        while current_coupon_date_fv <= futures_delivery_date and current_coupon_date_fv <= maturity_date:
            coupon_amount = (face_value * coupon_rate) * 0.5  # Semi-annual coupon

            # Time from coupon payment to futures delivery date
            time_from_coupon_to_delivery_years = (futures_delivery_date - current_coupon_date_fv).days / 360.0

            # Future value of coupon
            fv_coupons_during_futures_life += coupon_amount * (1 + repo_rate * time_from_coupon_to_delivery_years)
            current_coupon_date_fv += datetime.timedelta(days=int(0.5 * 365.25))

        # Theoretical Futures Price (clean price)
        # F_theoretical = (Dirty Spot Price * (1 + r_repo * T_delivery) - FV_coupons) / CF - Accrued Interest at Delivery

        # This is the standard formula for the theoretical futures price:
        # Futures Price (Clean) = [ (Dirty Spot Price + Cost of Carry) - Accrued Interest at Delivery ] / Conversion Factor
        # Cost of Carry = Dirty Spot Price * Repo Rate * (Days to Delivery / 360) - PV of Coupons during futures life
        # This is getting complicated with PV/FV. Let's use the most direct arbitrage formula:

        # Invoice Price = Futures Price * Conversion Factor + Accrued Interest at Delivery
        # At delivery, the short should be indifferent between:
        # 1. Buying the bond in spot market and delivering it.
        # 2. Holding the bond from valuation date until delivery.

        # The theoretical futures price (clean) is such that the cost of buying the bond today,
        # holding it, and delivering it is minimized.

        # Cost to deliver = (Dirty Spot Price + Financing Cost - Coupon Income)
        # Financing Cost = Dirty Spot Price * Repo Rate * (Days from Valuation to Delivery / 360)
        # Coupon Income = Sum of (Coupons * (1 + Repo Rate * Days from Coupon to Delivery / 360))

        # Let's use the formula for the theoretical futures price (clean):
        # F = (S_dirty * (1 + R * T) - Sum(C_i * (1 + R * (T - t_i)))) / CF - AI_T
        # Where:
        # S_dirty = current dirty price of the bond
        # R = repo rate
        # T = time to delivery (years)
        # C_i = i-th coupon payment
        # t_i = time from valuation date to i-th coupon payment
        # AI_T = accrued interest at delivery date

        # This is the most common textbook formula for theoretical futures price:
        # Futures Price (Clean) = (Dirty Spot Price * (1 + Repo Rate * (Days to Delivery / 360)) - Sum of Future Value of Coupons) / Conversion Factor - Accrued Interest at Delivery

        # Let's re-calculate FV_coupons_during_futures_life more robustly
        fv_coupons_during_futures_life_recalc = 0.0

        # Find all coupon dates between valuation_date and futures_delivery_date (inclusive of delivery date if it's a coupon date)
        coupon_dates_in_period = []
        temp_date = issue_date
        while temp_date <= maturity_date:
            if temp_date > valuation_date and temp_date <= futures_delivery_date:
                coupon_dates_in_period.append(temp_date)
            temp_date += datetime.timedelta(days=int(0.5 * 365.25))

        for c_date in coupon_dates_in_period:
            coupon_amount = (face_value * coupon_rate) * 0.5
            time_from_coupon_to_delivery_years_fv = (futures_delivery_date - c_date).days / 360.0
            fv_coupons_during_futures_life_recalc += coupon_amount * (
                        1 + repo_rate * time_from_coupon_to_delivery_years_fv)

        # Theoretical Futures Price (Clean)
        theoretical_futures_price_clean = (
                                                  (current_market_price_dirty * (
                                                              1 + repo_rate * time_to_delivery_years)) - fv_coupons_during_futures_life_recalc
                                          ) / conversion_factor - accrued_interest_at_delivery

        # The cost of delivery for the short (normalized by conversion factor)
        # This is the value that needs to be minimized.
        # Cost = (Dirty Price - PV of Coupons) * (1 + Repo Rate * T) / CF
        # This is often simplified to:
        # (Dirty Price + Carry Cost - Accrued Interest at Delivery) / Conversion Factor
        # Where Carry Cost = Dirty Price * Repo Rate * (Days to Delivery / 360) - Sum of Future Value of Coupons

        # Let's use the common "implied repo rate" method in reverse to find futures price.
        # The futures price is such that the implied repo rate on the CTD is the highest.
        # Or, directly, the theoretical futures price is:
        # F = (S_dirty - PVCI) * (1 + r_repo * T) / CF - AI_T
        # This is also a common form.
        # Let's stick to the simplest cost-of-carry:

        # Cost of carrying the bond to delivery date (per unit of notional)
        # This is the "basis" in some contexts.
        # Cost = (Dirty Spot Price - PV of Coupons) * (1 + Repo Rate * Time to Delivery)
        # Then, Futures Price = (Cost - Accrued Interest at Delivery) / Conversion Factor

        # Let's use the most straightforward formula for the theoretical futures price (clean):
        # F = [ (S_dirty * (1 + r * T)) - FV_coupons ] / CF - AI_T
        # Where T is time to delivery from valuation date.

        # Calculate the "cost of delivery" for this bond
        # This is the amount the short would effectively pay to deliver this bond.
        # It's (Dirty Spot Price - PV of coupons received) * (1 + repo_rate * time_to_delivery)
        # Then, divide by conversion factor and subtract accrued interest at delivery.

        # Let's use the standard formula where the futures price is the forward price
        # of the CTD bond, adjusted by its conversion factor.

        # Forward Price (Dirty) = Spot_Dirty * exp(repo_rate * time_to_delivery_years) - FV_coupons_during_futures_life_recalc
        # Or with discrete compounding:
        forward_price_dirty = current_market_price_dirty * (
                    1 + repo_rate * time_to_delivery_years) - fv_coupons_during_futures_life_recalc

        theoretical_futures_price_clean = (forward_price_dirty - accrued_interest_at_delivery) / conversion_factor

        # We are looking for the CTD, which minimizes the "Implied Repo Rate"
        # Or, equivalently, minimizes the "Cost of Delivery"
        # Cost of Delivery = (Dirty Spot Price - PVCI) / Conversion Factor
        # This is more for finding CTD.

        # For pricing the futures contract, we assume the market prices the CTD.
        # The theoretical futures price is such that the implied repo rate is equal to the market repo rate.
        # Or, more simply, it's the forward price of the CTD divided by its CF.

        # Let's calculate the "Implied Repo Rate" for each bond
        # Implied Repo Rate = ( (Futures Price * CF + AI_delivery) / Dirty Spot Price )^(1/T) - 1
        # This is what you'd typically solve for to find the CTD.

        # To find the futures price, we need to find the CTD first.
        # The CTD is the bond that minimizes:
        # (Dirty Price - Accrued Interest at Delivery) / Conversion Factor
        # This is the "clean price" of the bond adjusted for delivery.

        # The actual cost of delivering a bond is:
        # Invoice Price - (Dirty Spot Price + Cost of Carry)
        # Where Cost of Carry = Dirty Spot Price * (Repo Rate * Time to Delivery) - FV of Coupons

        # The short will deliver the bond that maximizes:
        # (Invoice Price - Dirty Spot Price - Carry Cost)
        # Or minimizes:
        # Dirty Spot Price + Carry Cost - Invoice Price

        # Let's calculate the "Net Basis" for each bond:
        # Net Basis = Dirty Spot Price - (Futures Price * Conversion Factor) - Accrued Interest at Delivery
        # The CTD is the bond with the lowest (most negative) net basis.

        # For theoretical pricing, we assume the futures price is such that the implied repo rate
        # of the CTD bond equals the market repo rate.

        # Let's calculate the "Gross Basis" for each bond:
        # Gross Basis = Current Dirty Price - (Futures Price * Conversion Factor)
        # The CTD is the one with the lowest Gross Basis.

        # For simplicity, we will calculate the "Implied Futures Price" for each bond
        # based on the cost of carry, and the lowest one (after adjusting for CF) will be the CTD.

        # Cost of carrying the bond to delivery (including coupons)
        # This is the "all-in" cost if you buy the bond today and hold it to delivery.
        # All-in Cost = Dirty Spot Price * (1 + repo_rate * time_to_delivery_years) - fv_coupons_during_futures_life_recalc

        # The theoretical clean futures price for *this specific bond* if it were the CTD
        # This is the price that would make the implied repo rate equal to the market repo rate.
        theoretical_clean_futures_price_for_this_bond = (
                                                                (current_market_price_dirty * (
                                                                            1 + repo_rate * time_to_delivery_years))
                                                                - fv_coupons_during_futures_life_recalc
                                                        ) / conversion_factor - accrued_interest_at_delivery

        # We are looking for the bond that gives the *highest* implied repo rate
        # or, more directly, the one that minimizes the cost of delivery.
        # The cost of delivery is (Dirty Spot Price - PV of Coupons) / Conversion Factor
        # This is the "adjusted" clean price.

        # The short will deliver the bond that minimizes:
        # (Dirty Price - Accrued Interest at Delivery) / Conversion Factor
        # Let's call this the "Adjusted Clean Price"
        adjusted_clean_price = (current_market_price_dirty - accrued_interest_at_delivery) / conversion_factor

        # The futures price should be the minimum of these adjusted clean prices.
        # This is a common simplification for the theoretical futures price.

        # Let's use the more standard approach for CTD:
        # CTD is the bond that maximizes its "Implied Repo Rate"
        # Implied Repo Rate = [ (Invoice Price / Dirty Spot Price) ^ (1/T) ] - 1
        # Where Invoice Price = Futures Price * CF + AI_delivery

        # Since we are trying to *find* the Futures Price, we can't use it directly in the CTD calculation.
        # Instead, we find the bond that minimizes the "net cost of delivery"
        # Net Cost = Dirty Spot Price - PV of (Futures Price * CF + AI_delivery)
        # This is a circular problem.

        # The most common simplified approach to finding the theoretical futures price is:
        # 1. For each deliverable bond, calculate its "implied futures price"
        #    using the cost-of-carry model with the market repo rate.
        # 2. The *lowest* of these implied futures prices will be the theoretical futures price.
        #    This is because the short will always deliver the bond that makes the futures contract
        #    cheapest for them.

        # Re-calculating implied futures price for each bond:
        # F = (Dirty Spot Price - PV of Coupons) * (1 + Repo Rate * Time to Delivery) / Conversion Factor - Accrued Interest at Delivery
        # This is a very common formula.

        # Let's refine the PV of coupons (PVCI) part for accuracy
        pv_coupons_during_futures_life_refined = 0.0

        # Find all coupon dates between valuation_date and futures_delivery_date
        coupon_dates_between = []
        temp_date = issue_date
        while temp_date <= maturity_date:
            if temp_date > valuation_date and temp_date <= futures_delivery_date:
                coupon_dates_between.append(temp_date)
            temp_date += datetime.timedelta(days=int(0.5 * 365.25))

        for c_date in coupon_dates_between:
            coupon_amount = (face_value * coupon_rate) * 0.5
            time_to_coupon_payment_years_pv = (c_date - valuation_date).days / 360.0
            discount_factor_pv = np.exp(-repo_rate * time_to_coupon_payment_years_pv)
            pv_coupons_during_futures_life_refined += coupon_amount * discount_factor_pv

        # Calculate the theoretical futures price for *this specific bond*
        # based on the cost-of-carry model.
        # F = (S_dirty - PVCI) * (1 + r * T) / CF - AI_T
        # Where S_dirty is current dirty price, PVCI is PV of coupons during futures life
        # r is repo rate, T is time to delivery, CF is conversion factor, AI_T is accrued interest at delivery.

        # Let's use the simplified formula where the futures price is the forward price of the CTD bond, adjusted by its conversion factor.
        # Forward Price (Clean) = (Dirty Spot Price * (1 + Repo Rate * T)) - FV_Coupons - Accrued Interest at Delivery
        # Futures Price = Forward Price (Clean) / Conversion Factor

        # The most common conceptual formula for theoretical futures price:
        # F = (S_0 + C - I) / CF
        # Where S_0 = current dirty price, C = PV of coupons, I = financing cost
        # This is equivalent to:
        # F = (S_0 * (1 + r_repo * T) - FV_coupons) / CF - AI_delivery

        # Let's stick to the formula that directly calculates the theoretical futures price
        # for each bond, and then we take the minimum.

        # Calculate the theoretical futures price (clean) for this bond
        # This is the price that would make the implied repo rate equal to the market repo rate.

        # Calculate the "all-in" cost of holding the bond until delivery, then normalize.
        all_in_cost_dirty = current_market_price_dirty * (1 + repo_rate * time_to_delivery_years)

        # Subtract future value of coupons received
        fv_coupons_in_period = 0.0
        temp_date_fv = issue_date
        while temp_date_fv <= maturity_date:
            if temp_date_fv > valuation_date and temp_date_fv <= futures_delivery_date:
                coupon_amount = (face_value * coupon_rate) * 0.5
                time_to_fv_years = (futures_delivery_date - temp_date_fv).days / 360.0
                fv_coupons_in_period += coupon_amount * (1 + repo_rate * time_to_fv_years)
            temp_date_fv += datetime.timedelta(days=int(0.5 * 365.25))

        all_in_cost_dirty -= fv_coupons_in_period

        # Convert to clean price at delivery
        all_in_cost_clean_at_delivery = all_in_cost_dirty - accrued_interest_at_delivery

        # Divide by conversion factor to get implied futures price
        implied_futures_price_for_this_bond = all_in_cost_clean_at_delivery / conversion_factor

        # Compare with current minimum cost
        if implied_futures_price_for_this_bond < min_cost_of_delivery:
            min_cost_of_delivery = implied_futures_price_for_this_bond
            cheapest_to_deliver_bond = {
                'id': bond_id,
                'current_market_price_clean': current_market_price_clean,
                'accrued_interest_spot': accrued_interest_spot,
                'conversion_factor': conversion_factor,
                'accrued_interest_at_delivery': accrued_interest_at_delivery,
                'implied_futures_price': implied_futures_price_for_this_bond
            }

    if cheapest_to_deliver_bond is None:
        return {"error": "No eligible bonds found or valid for delivery."}

    return {
        "theoretical_futures_price_clean": min_cost_of_delivery,
        "cheapest_to_deliver_bond": cheapest_to_deliver_bond,
        "valuation_date": valuation_date,
        "futures_delivery_date": futures_delivery_date,
        "repo_rate": repo_rate
    }


# --- Example Usage ---
if __name__ == "__main__":
    # Parameters from the user
    valuation_date = datetime.date(2025, 7, 10)
    futures_delivery_date = datetime.date(2025, 9, 20)  # Example delivery date for Sept futures
    repo_rate = 0.025  # 2.5% annual repo rate

    # Define eligible bonds (simplified data)
    # In reality, you'd get this from CME or a data provider.
    # 'current_market_price' is the clean price.
    eligible_bonds_data = [
        {
            'id': 'UST_2.5_2045',
            'face_value': 1000.0,
            'coupon_rate': 0.025,  # 2.5% annual coupon
            'maturity_date': datetime.date(2045, 5, 15),
            'issue_date': datetime.date(2015, 5, 15),
            'current_market_price': 850.0  # Example clean price
        },
        {
            'id': 'UST_3.0_2047',
            'face_value': 1000.0,
            'coupon_rate': 0.030,  # 3.0% annual coupon
            'maturity_date': datetime.date(2047, 8, 15),
            'issue_date': datetime.date(2017, 8, 15),
            'current_market_price': 920.0  # Example clean price
        },
        {
            'id': 'UST_2.0_2040',
            'face_value': 1000.0,
            'coupon_rate': 0.020,  # 2.0% annual coupon
            'maturity_date': datetime.date(2040, 2, 15),
            'issue_date': datetime.date(2010, 2, 15),
            'current_market_price': 800.0  # Example clean price
        }
    ]

    print(f"Valuation Date: {valuation_date}")
    print(f"Futures Delivery Date: {futures_delivery_date}")
    print(f"Repo Rate: {repo_rate:.2%}")
    print("\n--- Eligible Bonds Data ---")
    for bond in eligible_bonds_data:
        print(
            f"  ID: {bond['id']}, Coupon: {bond['coupon_rate']:.1%}, Maturity: {bond['maturity_date']}, Price: ${bond['current_market_price']:.2f}")

    pricing_result = price_us_bond_future(
        valuation_date,
        futures_delivery_date,
        repo_rate,
        eligible_bonds_data
    )

    if "error" in pricing_result:
        print(f"\nError: {pricing_result['error']}")
    else:
        print("\n--- Futures Pricing Result ---")
        print(f"Theoretical Futures Price (Clean): ${pricing_result['theoretical_futures_price_clean']:.4f}")
        print("\nCheapest-to-Deliver (CTD) Bond Details:")
        ctd = pricing_result['cheapest_to_deliver_bond']
        print(f"  ID: {ctd['id']}")
        print(f"  Current Market Price (Clean): ${ctd['current_market_price_clean']:.2f}")
        print(f"  Accrued Interest (Spot): ${ctd['accrued_interest_spot']:.2f}")
        print(f"  Conversion Factor: {ctd['conversion_factor']:.4f}")
        print(f"  Accrued Interest (at Delivery): ${ctd['accrued_interest_at_delivery']:.2f}")
        print(f"  Implied Futures Price (from CTD): ${ctd['implied_futures_price']:.4f}")

    # --- Demonstrate Accrued Interest and Conversion Factor for one bond ---
    print("\n--- Detailed Calculation for UST_2.5_2045 ---")
    bond_a = eligible_bonds_data[0]
    ai_spot_a = calculate_accrued_interest(bond_a['issue_date'], valuation_date, bond_a['coupon_rate'],
                                           bond_a['face_value'])
    print(f"Accrued Interest (Spot) for {bond_a['id']}: ${ai_spot_a:.2f}")

    cf_a = calculate_conversion_factor(bond_a, futures_delivery_date)
    print(f"Conversion Factor for {bond_a['id']}: {cf_a:.4f}")

    ai_delivery_a = calculate_accrued_interest(bond_a['issue_date'], futures_delivery_date, bond_a['coupon_rate'],
                                               bond_a['face_value'])
    print(f"Accrued Interest (at Delivery) for {bond_a['id']}: ${ai_delivery_a:.2f}")