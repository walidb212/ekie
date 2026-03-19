"""Tests for the legal domain classifier."""

import json
from unittest.mock import MagicMock, patch

from api.classifier import classify_question


class TestClassifyQuestion:
    """Test suite for classify_question."""

    @patch("api.classifier.genai")
    def test_returns_valid_domain(self, mock_genai):
        """Test that a valid classification is returned."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "domaine": "travail",
            "sous_domaine": "licenciement",
            "confiance": 0.94,
        })
        mock_client.models.generate_content.return_value = mock_response

        result = classify_question(
            "Mon employeur m'a licencié sans préavis",
            gemini_client=mock_client,
        )

        assert result["domaine"] == "travail"
        assert result["sous_domaine"] == "licenciement"
        assert result["confiance"] == 0.94

    @patch("api.classifier.genai")
    def test_falls_back_on_invalid_domain(self, mock_genai):
        """Test fallback to 'autre' for unknown domain."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "domaine": "unknown_domain",
            "sous_domaine": "test",
            "confiance": 0.8,
        })
        mock_client.models.generate_content.return_value = mock_response

        result = classify_question("Question test", gemini_client=mock_client)

        assert result["domaine"] == "autre"

    @patch("api.classifier.genai")
    def test_falls_back_on_json_error(self, mock_genai):
        """Test fallback on malformed JSON response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not JSON"
        mock_client.models.generate_content.return_value = mock_response

        result = classify_question("Question test", gemini_client=mock_client)

        assert result["domaine"] == "autre"
        assert result["confiance"] == 0.0

    @patch("api.classifier.genai")
    def test_falls_back_on_api_exception(self, mock_genai):
        """Test fallback when Gemini API raises an exception."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API down")

        result = classify_question("Question test", gemini_client=mock_client)

        assert result["domaine"] == "autre"
        assert result["confiance"] == 0.0

    @patch("api.classifier.genai")
    def test_strips_markdown_fences(self, mock_genai):
        """Test that markdown code fences are stripped before parsing."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '```json\n{"domaine": "fiscal", "sous_domaine": "impôts", "confiance": 0.88}\n```'
        mock_client.models.generate_content.return_value = mock_response

        result = classify_question("Question sur les impôts", gemini_client=mock_client)

        assert result["domaine"] == "fiscal"
        assert result["confiance"] == 0.88

    @patch("api.classifier.genai")
    def test_all_valid_domains(self, mock_genai):
        """Test that all valid domains are accepted."""
        valid_domains = ["travail", "famille", "immobilier", "consommation", "fiscal", "penal", "societe", "autre"]
        mock_client = MagicMock()

        for domain in valid_domains:
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "domaine": domain,
                "sous_domaine": "test",
                "confiance": 0.9,
            })
            mock_client.models.generate_content.return_value = mock_response

            result = classify_question("Test question", gemini_client=mock_client)
            assert result["domaine"] == domain
