"""Tests for the Mistral brief generator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from api.brief_generator import generate_brief
from api.models import BriefAvocat


VALID_BRIEF_JSON = {
    "domaine": "travail",
    "sous_domaine": "licenciement",
    "urgence": "haute",
    "resume_situation": "Le salarié a été licencié sans respect du préavis légal.",
    "points_cles": [
        "Licenciement potentiellement abusif",
        "Non-respect du préavis de 2 mois",
        "Indemnités de licenciement possiblement dues",
    ],
    "questions_a_creuser": [
        "Motif exact invoqué par l'employeur ?",
        "Ancienneté du salarié dans l'entreprise ?",
        "Convention collective applicable ?",
    ],
    "references_legales": [
        {
            "source": "Code du travail - Art. L1234-1",
            "extrait": "Le salarié licencié a droit à un préavis...",
            "pertinence": 0.95,
        }
    ],
    "delais_importants": "12 mois pour contester devant le CPH",
    "recommandation_immediate": "Saisir le Conseil de Prud'hommes dans les plus brefs délais.",
}

SAMPLE_CONTEXT = [
    {"source": "Code du travail - Art. L1234-1", "extrait": "Le salarié a droit à un préavis...", "pertinence": 0.95},
    {"source": "Code du travail - Art. L1234-5", "extrait": "L'indemnité compensatrice...", "pertinence": 0.88},
]


class TestGenerateBrief:
    """Test suite for generate_brief."""

    def test_returns_valid_brief(self):
        """Test that a valid BriefAvocat is returned."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(VALID_BRIEF_JSON)
        mock_client.chat.complete.return_value = mock_response

        brief = generate_brief(
            question="Mon employeur m'a licencié sans préavis",
            domaine="travail",
            sous_domaine="licenciement",
            context_docs=SAMPLE_CONTEXT,
            mistral_client=mock_client,
        )

        assert isinstance(brief, BriefAvocat)
        assert brief.domaine == "travail"
        assert brief.urgence == "haute"
        assert len(brief.points_cles) == 3
        assert len(brief.references_legales) >= 1

    def test_retries_on_invalid_json(self):
        """Test retry mechanism when first response is invalid JSON."""
        mock_client = MagicMock()

        # First call: invalid JSON, second call: valid
        response_invalid = MagicMock()
        response_invalid.choices = [MagicMock()]
        response_invalid.choices[0].message.content = "Not valid JSON"

        response_valid = MagicMock()
        response_valid.choices = [MagicMock()]
        response_valid.choices[0].message.content = json.dumps(VALID_BRIEF_JSON)

        mock_client.chat.complete.side_effect = [response_invalid, response_valid]

        brief = generate_brief(
            question="Test question",
            domaine="travail",
            sous_domaine="licenciement",
            context_docs=SAMPLE_CONTEXT,
            mistral_client=mock_client,
        )

        assert isinstance(brief, BriefAvocat)
        assert mock_client.chat.complete.call_count == 2

    def test_raises_after_max_retries(self):
        """Test that ValueError is raised after 2 failed attempts."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "invalid json"
        mock_client.chat.complete.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to generate valid brief"):
            generate_brief(
                question="Test question",
                domaine="travail",
                sous_domaine="test",
                context_docs=SAMPLE_CONTEXT,
                mistral_client=mock_client,
            )

    def test_strips_markdown_fences(self):
        """Test that markdown code fences are stripped."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```json\n" + json.dumps(VALID_BRIEF_JSON) + "\n```"
        mock_client.chat.complete.return_value = mock_response

        brief = generate_brief(
            question="Test",
            domaine="travail",
            sous_domaine="licenciement",
            context_docs=SAMPLE_CONTEXT,
            mistral_client=mock_client,
        )

        assert isinstance(brief, BriefAvocat)

    def test_all_urgence_levels(self):
        """Test all urgence levels are accepted."""
        for urgence in ["faible", "normale", "haute", "critique"]:
            mock_client = MagicMock()
            brief_data = {**VALID_BRIEF_JSON, "urgence": urgence}
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(brief_data)
            mock_client.chat.complete.return_value = mock_response

            brief = generate_brief(
                question="Test",
                domaine="travail",
                sous_domaine="test",
                context_docs=SAMPLE_CONTEXT,
                mistral_client=mock_client,
            )

            assert brief.urgence == urgence
