from finpricer import bond
# Bond parameters from the user
face_value = 1000
coupon_rate = 5  # in percent
years_to_maturity = 10
interest_rate = 4  # in percent

# Calculate and print the bond price
price = bond.calculate_bond_price(face_value, coupon_rate, years_to_maturity, interest_rate)
print(f"The calculated price of the bond is: ${price:.2f}") # 1081.11