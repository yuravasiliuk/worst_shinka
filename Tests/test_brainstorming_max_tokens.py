from unittest.mock import Mock, patch

from worst_shinka.brainstorming_system.brainstorm import BrainstormingPipeline


def test_brainstorming_sets_safe_max_tokens_for_llm_calls():
    pipeline = BrainstormingPipeline(
        model_a="model-a",
        model_b="model-b",
        config_path="/tmp",
        max_tokens=8192,
    )

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="ok"))]
    mock_response.usage = {"cost": 0.0}

    with patch("worst_shinka.brainstorming_system.brainstorm.client.chat.completions.create") as create:
        create.return_value = mock_response
        result = pipeline._call_llm("model-a", "system", "prompt")

    assert result == "ok"
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "model-a"
    assert kwargs["max_tokens"] == 8192
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"
