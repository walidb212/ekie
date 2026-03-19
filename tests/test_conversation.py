"""Tests for the conversation flow."""

from unittest.mock import MagicMock, patch

from api.conversation import (
    _conversations,
    _is_meaningful_user_message,
    _should_generate_brief,
    process_message,
)


class TestConversation:
    """Test suite for conversation safeguards."""

    def setup_method(self):
        """Reset in-memory conversations between tests."""
        _conversations.clear()

    def test_ignores_non_substantive_message(self):
        """Greeting-only messages should not enter the history."""
        result = process_message(
            conversation_id=None,
            user_message="Helloo",
            gemini_client=MagicMock(),
        )

        state = _conversations[result["conversation_id"]]

        assert result["action"] == "question"
        assert "contexte" in result["message"]
        assert state["messages"] == []
        assert state["questions_asked"] == []

    def test_keeps_short_legal_signal_message(self):
        """Short but legally meaningful messages should still count."""
        assert _is_meaningful_user_message("vol")

    @patch("api.conversation._generate_next_question")
    @patch("api.conversation.embed_query")
    @patch("api.conversation.classify_question")
    def test_short_follow_up_answers_are_accepted(
        self,
        mock_classify,
        mock_embed,
        mock_next_question,
    ):
        """Short answers like 'non' or 'Paris' should not be rejected mid-conversation."""
        mock_classify.return_value = {
            "domaine": "penal",
            "domaine_secondaire": None,
            "sous_domaine": "vol",
            "confiance": 0.93,
        }
        mock_embed.return_value = [0.1] * 768
        mock_next_question.side_effect = [
            "Avez-vous depose plainte ?",
            "Ou le vol a-t-il eu lieu ?",
            "Quel est le montant estime ?",
        ]

        first = process_message(None, "Je me suis fait voler des bijoux avant-hier.", MagicMock())
        second = process_message(first["conversation_id"], "non", MagicMock())
        third = process_message(second["conversation_id"], "Paris", MagicMock())

        assert first["message"] == "Avez-vous depose plainte ?"
        assert second["message"] == "Ou le vol a-t-il eu lieu ?"
        assert third["message"] == "Quel est le montant estime ?"

        state = _conversations[first["conversation_id"]]
        user_messages = [message["content"] for message in state["messages"] if message["role"] == "user"]
        assert user_messages == [
            "Je me suis fait voler des bijoux avant-hier.",
            "non",
            "Paris",
        ]

    @patch("api.conversation._generate_next_question")
    @patch("api.conversation.embed_query")
    @patch("api.conversation.classify_question")
    def test_greeting_reply_is_still_ignored_mid_conversation(
        self,
        mock_classify,
        mock_embed,
        mock_next_question,
    ):
        """A follow-up greeting should still be treated as noise."""
        mock_classify.return_value = {
            "domaine": "penal",
            "domaine_secondaire": None,
            "sous_domaine": "vol",
            "confiance": 0.93,
        }
        mock_embed.return_value = [0.1] * 768
        mock_next_question.return_value = "Avez-vous depose plainte ?"

        first = process_message(None, "Je me suis fait voler des bijoux avant-hier.", MagicMock())
        second = process_message(first["conversation_id"], "hello", MagicMock())

        assert second["action"] == "question"
        assert "contexte" in second["message"]

    def test_requires_four_substantive_user_messages_for_brief(self):
        """Three short substantive messages should not trigger the brief yet."""
        state = {
            "messages": [
                {"role": "user", "content": "Je me suis fait arnaquer."},
                {"role": "assistant", "content": "Quand cela s'est-il passe ?"},
                {"role": "user", "content": "Hier soir en rentrant."},
                {"role": "assistant", "content": "Quel est le montant du prejudice ?"},
                {"role": "user", "content": "On m'a vole des bijoux."},
            ],
            "confiance": 0.92,
            "questions_asked": ["Quand cela s'est-il passe ?", "Quel est le montant du prejudice ?"],
        }

        assert not _should_generate_brief(state)

    @patch("api.conversation._generate_next_question")
    @patch("api.conversation.embed_query")
    @patch("api.conversation.classify_question")
    def test_fourth_substantive_message_can_trigger_brief(
        self,
        mock_classify,
        mock_embed,
        mock_next_question,
    ):
        """The fourth substantive user message should allow the brief path."""
        mock_classify.return_value = {
            "domaine": "penal",
            "domaine_secondaire": None,
            "sous_domaine": "vol",
            "confiance": 0.93,
        }
        mock_embed.return_value = [0.1] * 768
        mock_next_question.return_value = "Question de suivi"

        first = process_message(None, "Je me suis fait arnaquer.", MagicMock())
        second = process_message(first["conversation_id"], "Hier soir en rentrant.", MagicMock())
        third = process_message(second["conversation_id"], "On m'a vole des bijoux.", MagicMock())

        assert third["action"] == "question"

        mock_qdrant = MagicMock()
        mock_mistral = MagicMock()
        mock_brief = MagicMock()

        with patch("api.conversation.retrieve_legal_context", return_value=[{"source": "Code penal", "extrait": "...", "pertinence": 0.91}]) as mock_retrieve:
            with patch("api.conversation.generate_brief", return_value=mock_brief) as mock_generate:
                result = process_message(
                    third["conversation_id"],
                    "La valeur est d'environ 8 000 euros.",
                    MagicMock(),
                    qdrant_client=mock_qdrant,
                    mistral_client=mock_mistral,
                )

        assert result["action"] == "brief"
        assert result["brief"] is mock_brief
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()
