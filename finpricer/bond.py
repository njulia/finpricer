"""
Imagine you want to borrow money, perhaps to buy a new car or expand your business. Instead of going to a bank, you could ask many different people to lend you small amounts. A **bond** is essentially an "IOU" (I Owe You) that works similarly.

*   When you **buy a bond**, you are lending money to an entity – usually a corporation (a company) or a government (like the U.S. Treasury or a city).
*   In return for your loan, the issuer (the one who borrowed the money) promises to do two things:
    1.  Pay you regular interest payments, called **coupon payments**, over a set period.
    2.  Repay your original loan amount (the **face value** or **par value**) when the bond reaches its **maturity date**.

So, a bond is a way for organizations to borrow money from investors, and for investors to earn a predictable income stream.

### The Price of a Bond

You mentioned your bond has a Face Value of $1000, but its calculated price is $1081.11. This immediately tells us that a bond's price isn't always its face value. So, why does a bond's price change?

The main reason is changes in **market interest rates** (often called the **Yield to Maturity** or YTM for bonds). Think of it like this:

*   **Your bond offers a fixed coupon rate:** Your bond promises to pay you 5% of its face value ($1000 * 5% = $50) every year for 10 years. This payment is set in stone when the bond is issued.
*   **The market offers a changing interest rate:** The current market interest rate (Yield to Maturity) is like the going rate for new investments with similar risk. In your case, it's 4%.

Now, let's compare them:

*   **Coupon Rate (5%) vs. Market Rate (4%)**
    *   Your bond's coupon rate (5%) is **higher** than the current market interest rate (4%).
    *   This means your bond is offering a *better* annual payment ($50) than what new similar investments in the market are currently offering (if a new $1000 bond were issued today at 4%, it would pay only $40 per year).
    *   Because your bond's fixed payments are more attractive, people are willing to pay *more than its face value* to own it. They are essentially paying a premium to get those higher coupon payments.
    *   This is why your bond is trading at a **premium**: Its price ($1081.11) is greater than its face value ($1000).

**In summary:**

*   If the **Coupon Rate > Market Interest Rate**, the bond trades at a **Premium** (Price > Face Value). (Your case!)
*   If the **Coupon Rate < Market Interest Rate**, the bond trades at a **Discount** (Price < Face Value). People would only buy it for less than face value because its payments are less attractive than what the market offers.
*   If the **Coupon Rate = Market Interest Rate**, the bond trades **at Par** (Price = Face Value).

### The Calculation Explained

The price of a bond is essentially the "present value" of all the money you expect to receive from it in the future.

#### What is "Present Value"?

Money you receive in the future is worth less than the same amount of money you have today. Why?
1.  **Inflation:** Over time, money tends to buy less.
2.  **Opportunity Cost:** If you have money today, you could invest it and earn interest, making it grow. If you have to wait for it, you miss out on that earning potential.

So, "present value" is the process of figuring out what a future payment is worth in today's dollars, considering the current market interest rate (which acts as our "discount rate").

The bond pricing formula breaks down into two main parts:

1.  **The Present Value of All Future Coupon Payments (An Annuity)**
    *   You will receive $50 (5% of $1000 face value) every year for 10 years. This series of equal, regular payments is called an **annuity**.
    *   Each of these $50 payments, received at different points in the future, needs to be discounted back to its value *today*.
    *   The further in the future a payment is, the more heavily it's discounted.
    *   This part of the calculation sums up the present value of all ten $50 payments.

2.  **The Present Value of the Face Value (A Lump Sum at Maturity)**
    *   At the end of 10 years, you will receive the bond's face value of $1000 back.
    *   This is a single, "lump sum" payment far in the future.
    *   This $1000 also needs to be discounted back to its value *today*.

**The bond's total price is simply the sum of these two present values.** The higher the discount rate (market interest rate), the lower the present value, and thus the lower the bond's price. Conversely, a lower discount rate leads to a higher present value and a higher bond price (as seen in your example, where the 4% market rate is lower than the 5% coupon, leading to a premium price).

"""
def price_bond(
        face_value: float,
        coupon_rate: float,
        years_to_maturity: float,
        interest_rate: float):
    """
    Calculates the present value of a bond.

    Args:
        face_value (float): The face value or par value is the principal amount.
        coupon_rate (float): The coupon rate in %.
        years_to_maturity (float): Number of periods (years).
        interest_rate (float): Market interest rate (yield to maturity) in %.

    Returns:
        float: The Present Value (PV) of a bond.
    """
    # Annual coupon payment
    c = face_value * (coupon_rate / 100)

    # Market interest rate (yield to maturity) as a decimal
    r = interest_rate / 100

    # Number of periods (years)
    n = years_to_maturity

    # Present value of the stream of coupon payments (annuity)
    pv_coupons = c * (1 - (1 + r) ** -n) / r

    # Present value of the face value paid at maturity
    pv_face_value = face_value / ((1 + r) ** n)

    # The bond's price is the sum of the present values
    bond_price = pv_coupons + pv_face_value
    return bond_price


# --- Example Usage ---
if __name__ == "__main__":
    # Parameters from the user
    face_value = 1000
    coupon_rate = 5  # in percent
    years_to_maturity = 10
    interest_rate = 4  # in percent

    # Calculate and print the price
    price = price_bond(face_value, coupon_rate, years_to_maturity, interest_rate)
    print(f"The calculated price of the bond is: ${price:.2f}")  # 1081.11


