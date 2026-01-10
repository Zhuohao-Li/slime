from transformers import AutoTokenizer

from slime.utils.types import MultimodalTypes


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        self.tokenizer = tokenizer
        self.system_message_length, self.gen_token_length = self.get_system_message_length()
        self.tokenizer_type = tokenizer_type

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def _normalize_messages_for_loss_mask(self, messages: list[dict]) -> list[dict]:
        normalized = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "text":
                            parts.append(item.get("text", ""))
                        elif item_type in ("image", "video", "audio"):
                            mt = MultimodalTypes.get(item_type)
                            parts.append(mt.placeholder if mt else f"<{item_type}>")
                    elif isinstance(item, str):
                        parts.append(item)
                new_msg = dict(msg)
                new_msg["content"] = "".join(parts)
                normalized.append(new_msg)
            else:
                normalized.append(msg)
        return normalized

    def _get_multimodal_token_ids(self) -> set[int]:
        keywords = ("image", "video", "audio", "vision")
        ids: set[int] = set()

        added_vocab = self.tokenizer.get_added_vocab()
        for token, token_id in added_vocab.items():
            if any(k in token.lower() for k in keywords):
                ids.add(token_id)

        for token in getattr(self.tokenizer, "additional_special_tokens", []) or []:
            if not any(k in token.lower() for k in keywords):
                continue
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                continue
            if self.tokenizer.unk_token_id is not None and token_id == self.tokenizer.unk_token_id:
                continue
            ids.add(token_id)

        for placeholder in [m.placeholder for m in MultimodalTypes.all()]:
            token_id = self.tokenizer.convert_tokens_to_ids(placeholder)
            if token_id is None:
                continue
            if self.tokenizer.unk_token_id is not None and token_id == self.tokenizer.unk_token_id:
                continue
            ids.add(token_id)

        return ids

    def _align_loss_mask_to_input_ids(
        self, input_ids: list[int], token_ids: list[int], loss_mask: list[int]
    ) -> list[int]:
        multimodal_token_ids = self._get_multimodal_token_ids()
        aligned_mask = []
        text_idx = 0

        for tok in input_ids:
            if tok in multimodal_token_ids:
                aligned_mask.append(0)
                continue

            while text_idx < len(token_ids) and token_ids[text_idx] in multimodal_token_ids:
                text_idx += 1

            if text_idx >= len(token_ids):
                aligned_mask.append(0)
                continue

            if tok != token_ids[text_idx]:
                raise ValueError(
                    "Multimodal alignment failed: token mismatch between processor input_ids and loss mask tokens. "
                    "Please check that the chat template and processor tokenization are consistent."
                )

            aligned_mask.append(loss_mask[text_idx])
            text_idx += 1

        while text_idx < len(token_ids) and token_ids[text_idx] in multimodal_token_ids:
            text_idx += 1

        if text_idx != len(token_ids):
            raise ValueError(
                "Multimodal alignment failed: unused loss mask tokens remain after alignment. "
                "Please check that the chat template and processor tokenization are consistent."
            )

        return aligned_mask

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, tools=tools)
            else:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True)

        for i, message in enumerate(messages):
            if i == 0:
                tailed_message_ids = self.tokenizer.apply_chat_template(
                    [message, prefix_message], tokenize=True, tools=tools
                )
                message_ids = tailed_message_ids[: -len(prefix_token_ids)]
            else:
                prefixed_message_ids = self.tokenizer.apply_chat_template([prefix_message, message], tokenize=True)
                message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        normalized_messages = self._normalize_messages_for_loss_mask(messages)
        token_ids, loss_mask_text = self.get_loss_mask(normalized_messages, tools=tools)
        loss_mask = self._align_loss_mask_to_input_ids(input_ids, token_ids, loss_mask_text)
        return input_ids, loss_mask

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
