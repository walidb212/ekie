"""Tests for the Qdrant retriever."""

from unittest.mock import MagicMock, patch

from api.retriever import retrieve_legal_context


def _make_scored_point(point_id, score, payload):
    """Create a mock scored point."""
    point = MagicMock()
    point.id = point_id
    point.score = score
    point.payload = payload
    return point


class TestRetrieveLegalContext:
    """Test suite for retrieve_legal_context."""

    @patch("api.retriever.np")
    def test_returns_top_3(self, mock_np):
        """Test that top 3 results are returned."""
        mock_np.array.return_value = MagicMock()
        mock_np.linalg.norm.return_value = 1.0
        mock_np.array.return_value.__truediv__ = MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=[0.1] * 768)))

        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()

        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        mock_embed_result = MagicMock()
        mock_embed_result.embeddings = [mock_embedding]
        mock_gemini.models.embed_content.return_value = mock_embed_result

        # Mock Qdrant response
        mock_search_result = MagicMock()
        mock_search_result.points = [
            _make_scored_point("1", 0.95, {"text": "Article L1234-1", "source": "Code du travail", "domaine": "travail"}),
            _make_scored_point("2", 0.88, {"text": "Article L1234-2", "source": "Code du travail", "domaine": "travail"}),
            _make_scored_point("3", 0.82, {"text": "Article 1240", "source": "Code civil", "domaine": "famille"}),
        ]
        mock_qdrant.query_points.return_value = mock_search_result

        results = retrieve_legal_context(
            question="Mon employeur m'a licencié",
            domaine="travail",
            confiance=0.9,
            n=3,
            qdrant_client=mock_qdrant,
            gemini_client=mock_gemini,
        )

        assert len(results) == 3
        assert results[0]["pertinence"] == 0.95
        assert results[0]["source"] == "Code du travail"

    @patch("api.retriever.np")
    def test_returns_empty_on_no_results(self, mock_np):
        """Test empty list when Qdrant returns no results."""
        mock_np.array.return_value = MagicMock()
        mock_np.linalg.norm.return_value = 1.0
        mock_np.array.return_value.__truediv__ = MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=[0.1] * 768)))

        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        mock_embed_result = MagicMock()
        mock_embed_result.embeddings = [mock_embedding]
        mock_gemini.models.embed_content.return_value = mock_embed_result

        mock_search_result = MagicMock()
        mock_search_result.points = []
        mock_qdrant.query_points.return_value = mock_search_result

        results = retrieve_legal_context(
            question="Question sans résultat",
            domaine="autre",
            confiance=0.3,
            qdrant_client=mock_qdrant,
            gemini_client=mock_gemini,
        )

        assert len(results) == 0

    @patch("api.retriever.np")
    def test_filters_by_domain_when_confident(self, mock_np):
        """Test that domain filter is applied when confidence > 0.7."""
        mock_np.array.return_value = MagicMock()
        mock_np.linalg.norm.return_value = 1.0
        mock_np.array.return_value.__truediv__ = MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=[0.1] * 768)))

        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        mock_embed_result = MagicMock()
        mock_embed_result.embeddings = [mock_embedding]
        mock_gemini.models.embed_content.return_value = mock_embed_result

        mock_search_result = MagicMock()
        mock_search_result.points = []
        mock_qdrant.query_points.return_value = mock_search_result

        retrieve_legal_context(
            question="Test",
            domaine="travail",
            confiance=0.9,
            qdrant_client=mock_qdrant,
            gemini_client=mock_gemini,
        )

        call_kwargs = mock_qdrant.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None

    @patch("api.retriever.np")
    def test_no_filter_when_low_confidence(self, mock_np):
        """Test that no domain filter when confidence <= 0.7."""
        mock_np.array.return_value = MagicMock()
        mock_np.linalg.norm.return_value = 1.0
        mock_np.array.return_value.__truediv__ = MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=[0.1] * 768)))

        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        mock_embed_result = MagicMock()
        mock_embed_result.embeddings = [mock_embedding]
        mock_gemini.models.embed_content.return_value = mock_embed_result

        mock_search_result = MagicMock()
        mock_search_result.points = []
        mock_qdrant.query_points.return_value = mock_search_result

        retrieve_legal_context(
            question="Test",
            domaine="travail",
            confiance=0.5,
            qdrant_client=mock_qdrant,
            gemini_client=mock_gemini,
        )

        call_kwargs = mock_qdrant.query_points.call_args[1]
        assert call_kwargs["query_filter"] is None
