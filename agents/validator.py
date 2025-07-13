from typing import List
import pandas as pd
from langchain.tools import tool

@tool("validate_financial_csv", return_direct=True)
def validate_financial_csv(csv_path: str) -> List[str]:
    """
    Validates a financial CSV for missing values, negative Net_Income, negative Shareholder_Equity, and Interest_Expense exceeding Interest_Income.
    Args:
        csv_path (str): Path to the CSV file.
    Returns:
        List[str]: List of validation messages (empty if all checks pass).
    """
    messages = []
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return [f"❌ Error loading CSV: {e}"]

    key_columns = [
        "Interest_Income", "Interest_Expense", "Average_Earning_Assets", "Net_Income",
        "Total_Assets", "Shareholder_Equity", "Operating_Expenses", "Operating_Income"
    ]
    for col in key_columns:
        if col not in df.columns:
            messages.append(f"❌ Missing column: {col}")
        elif df[col].isnull().any():
            messages.append(f"❌ Missing values in column: {col}")

    # Check for negative Net_Income
    if "Net_Income" in df.columns:
        if (df["Net_Income"] < 0).any():
            messages.append("❌ Net_Income is negative for at least one row.")

    # Check for negative Shareholder_Equity
    if "Shareholder_Equity" in df.columns:
        if (df["Shareholder_Equity"] < 0).any():
            messages.append("❌ Shareholder_Equity is negative for at least one row.")

    # Check if Interest_Expense exceeds Interest_Income
    if "Interest_Income" in df.columns and "Interest_Expense" in df.columns:
        if (df["Interest_Expense"] > df["Interest_Income"]).any():
            messages.append("❌ Interest_Expense exceeds Interest_Income for at least one row.")

    if not messages:
        messages.append("✅ File passes all validation checks.")
    return messages
