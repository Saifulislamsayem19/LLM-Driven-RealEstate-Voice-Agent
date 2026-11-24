"""
Unit tests for the data_manager service.
Run with: pytest tests/
"""
import pytest


def test_convert_numpy_types_int():
    import numpy as np
    from app.utils.data_manager import convert_numpy_types

    result = convert_numpy_types(np.int64(42))
    assert result == 42
    assert isinstance(result, int)


def test_convert_numpy_types_float():
    import numpy as np
    from app.utils.data_manager import convert_numpy_types

    result = convert_numpy_types(np.float64(3.14))
    assert abs(result - 3.14) < 1e-9
    assert isinstance(result, float)


def test_convert_numpy_types_nested():
    import numpy as np
    from app.utils.data_manager import convert_numpy_types

    data = {"price": np.int64(500000), "scores": [np.float64(4.5), np.float64(3.0)]}
    result = convert_numpy_types(data)
    assert result == {"price": 500000, "scores": [4.5, 3.0]}


def test_convert_numpy_types_nan_becomes_none():
    import numpy as np
    from app.utils.data_manager import convert_numpy_types

    data = {"a": np.float64("nan"), "b": float("nan"), "c": np.float64("inf"), "d": 1.5}
    assert convert_numpy_types(data) == {"a": None, "b": None, "c": None, "d": 1.5}


@pytest.mark.parametrize("text, expected", [
    ("I want a 3 bedroom house in Seattle, budget around 600000", None),
    ("budget $12000000 please", None),
    ("around 15000000 dollars", None),
    ("my whatsapp is 01305354567", "01305354567"),
    ("call me at +1 (206) 555-0123 thanks", "+1 (206) 555-0123"),
    ("budget 500000, whatsapp +8801305354567", "+8801305354567"),
])
def test_extract_phone_number_ignores_budgets(text, expected):
    from app.services.agent import extract_phone_number

    assert extract_phone_number(text) == expected


def test_requirements_saved_prevents_duplicate_leads_on_subsequent_turns(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from unittest.mock import MagicMock, patch
    from langchain_core.messages import AIMessage
    from app.services.agent import RealEstateAgent
    from app.services.query_processor import process_query
    from app.core.config import settings

    csv = tmp_path / "user_requirements.csv"
    monkeypatch.setattr(type(settings), "USER_REQUIREMENTS_FILE", property(lambda self: csv))

    agent = RealEstateAgent()
    agent.df = pd.DataFrame([{
        "property_id": 1, "price": 500000.0, "city": "Seattle",
        "bedrooms": 3, "bathrooms": 2, "sqft_living": 1500
    }])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='{"budget": 500000, "location": "Seattle"}')

    with patch("app.services.agent.get_chat_model", return_value=mock_llm):
        agent.setup_chain()

    # Turn 1: User provides requirements and contact info
    res1 = process_query(agent, "budget 500k in Seattle, email jane@example.com", [])
    assert res1["system_state"]["requirements_saved"] is True
    assert len(pd.read_csv(csv)) == 1

    # Turn 2: User asks a follow-up question in the same chat
    history = [
        {"role": "user", "content": "budget 500k in Seattle, email jane@example.com"},
        {"role": "assistant", "content": res1["answer"]},
        {"role": "system", "content": json.dumps(res1["system_state"])},
    ]
    res2 = process_query(agent, "tell me more about property 1", history)
    assert res2["system_state"]["requirements_saved"] is True
    assert len(pd.read_csv(csv)) == 1

