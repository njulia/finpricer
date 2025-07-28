import datetime


class DateUtils:
    """Utility functions for date calculations."""

    @staticmethod
    def next_date(current_date, frequency_months):
        """Adds years to a date, handling leap years for maturity dates."""
        # Add months, handling year rollovers
        year = current_date.year + (current_date.month + frequency_months - 1) // 12
        month = (current_date.month + frequency_months - 1) % 12 + 1
        day = current_date.day  # Keep the same day of the month

        # Handle cases where day might exceed days in target month (e.g., Jan 31 + 1 month = Feb 31 -> Feb 28)
        try:
            dt = datetime.date(year, month, day)
        except ValueError:
            # If day is too high for the month, set to last day of the month
            dt = datetime.date(year, month, 1) + datetime.timedelta(days=-1)

        return dt
    @staticmethod
    def year_fraction(start_date, end_date, convention="ACT/365"):
        """
        Calculates year fraction between two dates.
        Simplified conventions.
        """
        delta = (end_date - start_date).days
        if convention == "ACT/365":
            return delta / 365.0
        elif convention == "ACT/360":
            return delta / 360.0
        elif convention == "30/360":
            # Simplified 30/360
            d1 = min(start_date.day, 30)
            d2 = min(end_date.day, 30) if start_date.day <= 30 else min(end_date.day, 30) if end_date.day != 31 else 30
            m1 = start_date.month
            m2 = end_date.month
            y1 = start_date.year
            y2 = end_date.year
            return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0
        else:
            raise ValueError("Unsupported day count convention")

    @staticmethod
    def generate_payment_dates(start_date, end_date, frequency_months):
        """Generates a list of payment dates."""
        dates = []
        current_date = start_date
        while current_date < end_date:
            current_date = DateUtils.next_date(current_date, frequency_months)
            if current_date > start_date:  # Exclude the start date itself if it's not a payment date
                dates.append(current_date)
        # Ensure the last payment date is not beyond maturity, or include maturity if it aligns
        if dates and dates[-1] > end_date:
            dates[-1] = end_date  # Adjust last date to maturity if it overshoots slightly
        elif not dates or dates[-1] < end_date:
            # Add maturity date if it's not already included and is a logical payment point
            if (end_date - start_date).days > 0 and (
            end_date - dates[-1] if dates else end_date - start_date).days < (
                    frequency_months / 12.0 * 365 * 1.5):  # Within 1.5 of a period
                dates.append(end_date)

        # Filter out dates before or on start_date
        dates = [d for d in dates if d > start_date]

        return sorted(list(set(dates)))  # Use set to remove duplicates, then sort

