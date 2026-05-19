from pydantic import BaseModel


class Transaction(BaseModel):

    user_id: str
    transaction_amount: float
    merchant_category: str
    payment_method: str
    network_latency: float
    upi_app: str
    failed_transaction_attempts: int
    transaction_gap_minutes: float
    is_recurring: int
    income_level: str
    user_profile: str
    behavior_personality: str
    is_weekend: int
    hour_of_day: int