from generator import GroundedGenerator
from rag import contextual_query, load_faqs, select_language


def test_sample_faq_data_is_generic_and_loads():
    faqs = load_faqs("data/sample_faq.json")
    assert len(faqs) == 3
    assert faqs[0].id == "account-access"


def test_language_selection_and_short_follow_up_context():
    assert select_language("میں مدد چاہتا ہوں", None) == "urdu"
    history = [{"role": "user", "content": "I want to update my profile"}]
    query = contextual_query(history, "where?", "english")
    assert "update my profile" in query and query.endswith("where?")


def test_generator_uses_verified_faq_when_no_provider_is_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    faq = load_faqs("data/sample_faq.json")[0]
    assert GroundedGenerator().answer("How do I sign in?", faq, "english") == faq.answer_en
