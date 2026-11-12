from philo_mcts.models import chat_token_ids


class FakeTokenizer:
    def __init__(self, result):
        self.result = result

    def apply_chat_template(self, *args, **kwargs):
        return self.result


def test_chat_token_ids_supports_transformers_4_and_5():
    messages = [{"role": "user", "content": "test"}]
    assert chat_token_ids(FakeTokenizer([1, 2]), messages) == [1, 2]
    assert chat_token_ids(FakeTokenizer({"input_ids": [1, 2]}), messages) == [1, 2]
