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
    def test_filters_low_pertinence_results(self, mock_np):
        """Test that weak references are dropped before returning results."""
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
        mock_search_result.points = [
            _make_scored_point("1", 0.91, {"text": "Article pertinent", "source": "Source fiable"}),
            _make_scored_point("2", 0.72, {"text": "Decision pertinente", "source": "Decision solide"}),
            _make_scored_point("3", 0.60, {"text": "Decision hors sujet", "source": "Reference douteuse"}),
        ]
        mock_qdrant.query_points.return_value = mock_search_result

        results = retrieve_legal_context(
            question="On m'a vole des bijoux",
            domaine="travail",
            confiance=0.9,
            n=3,
            qdrant_client=mock_qdrant,
            gemini_client=mock_gemini,
        )

        assert len(results) == 2
        assert all(result["pertinence"] >= 0.72 for result in results)
        assert all(result["source"] != "Reference douteuse" for result in results)

    @patch("api.retriever.np")
    def test_penal_uses_lower_pertinence_threshold(self, mock_np):
        """Penal retrieval should keep moderately relevant references."""
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
        mock_search_result.points = [
            _make_scored_point("1", 0.61, {"text": "Article 311-1", "source": "Code penal"}),
            _make_scored_point("2", 0.59, {"text": "Decision faible", "source": "Reference faible"}),
        ]
        mock_qdrant.query_points.return_value = mock_search_result

        results = retrieve_legal_context(
            question="On m'a vole mon sac",
            domaine="penal",
            confiance=0.9,
            n=3,
            qdrant_client=mock_qdrant,
            gemini_client=mock_gemini,
        )

        assert len(results) == 1
        assert results[0]["source"] == "Code penal"
        assert results[0]["pertinence"] == 0.61

    @patch("api.retriever.np")
    def test_retries_without_domain_filter_when_no_refs_survive(self, mock_np):
        """If the filtered search yields nothing usable, fallback without domain filter."""
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

        first_search = MagicMock()
        first_search.points = [
            _make_scored_point("1", 0.58, {"text": "Reference trop faible", "source": "Faible"}),
        ]
        second_search = MagicMock()
        second_search.points = [
            _make_scored_point("2", 0.75, {"text": "Article utile", "source": "Code penal"}),
        ]
        mock_qdrant.query_points.side_effect = [first_search, second_search]

        results = retrieve_legal_context(
            question="On m'a vole mon sac",
            domaine="travail",
            confiance=0.9,
            n=3,
            qdrant_client=mock_qdrant,
            gemini_client=mock_gemini,
        )

        assert len(results) == 1
        assert results[0]["source"] == "Code penal"
        assert mock_qdrant.query_points.call_count == 2
        assert mock_qdrant.query_points.call_args_list[0][1]["query_filter"] is not None
        assert mock_qdrant.query_points.call_args_list[1][1]["query_filter"] is None

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

        call_kwargs = mock_qdrant.query_points.call_args_list[0][1]
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
