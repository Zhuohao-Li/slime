from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    return [mask.count(1) if 1 in mask else 0 for mask in loss_masks]


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

    def get_loss_mask(
        self, messages: list[dict], tools: list[dict] = None, input_ids: list[int] = None
    ) -> tuple[list[int], list[int]]:
        """Get loss mask for SFT training.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of tool definitions.
            input_ids: Optional input_ids from processor for multimodal inputs.
                       If provided, the loss_mask will be aligned to handle image tokens.
                       
        Returns:
            Tuple of (token_ids, loss_mask). If input_ids is provided, returns
            (input_ids, aligned_loss_mask) instead.
        """
        # First get text-only token_ids and loss_mask based on tokenizer type
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                text_token_ids, loss_mask_text = self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
            else:
                text_token_ids, loss_mask_text = self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            text_token_ids, loss_mask_text = self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            text_token_ids, loss_mask_text = self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")
        
        # If input_ids is provided, align loss_mask for multimodal inputs
        # Uses greedy matching - works for any template regardless of image token position
        if input_ids is not None:
            loss_mask = self._align_multimodal_loss_mask(input_ids, text_token_ids, loss_mask_text)
            assert len(loss_mask) == len(input_ids), (
                f"Aligned loss_mask length ({len(loss_mask)}) != input_ids length ({len(input_ids)})"
            )
            return input_ids, loss_mask
        
        return text_token_ids, loss_mask_text

    def _align_multimodal_loss_mask(
        self, input_ids: list[int], text_token_ids: list[int], loss_mask_text: list[int]
    ) -> list[int]:
        """Align loss mask for multimodal inputs.
        
        For Qwen3-VL style models: image tokens are enclosed between <|vision_start|>
        and <|vision_end|>. The text_token_ids contains <|image_pad|> as placeholder,
        while input_ids contains the actual image tokens.
        
        Args:
            input_ids: Full input_ids from processor (includes image tokens).
            text_token_ids: Token ids from tokenizer (with <|image_pad|> placeholder).
            loss_mask_text: Loss mask for text_token_ids.
            
        Returns:
            Aligned loss_mask with same length as input_ids.
        """
        vocab = self.tokenizer.get_added_vocab()
        vision_start_id = vocab.get("<|vision_start|>")
        vision_end_id = vocab.get("<|vision_end|>")
        image_pad_id = vocab.get("<|image_pad|>")
        
        loss_mask = []
        text_idx = 0
        input_idx = 0
        in_vision = False
        
        while input_idx < len(input_ids):
            token = input_ids[input_idx]
            
            if vision_start_id is not None and token == vision_start_id:
                # <|vision_start|> - match with text and enter vision region
                in_vision = True
                if text_idx < len(text_token_ids) and text_token_ids[text_idx] == vision_start_id:
                    loss_mask.append(loss_mask_text[text_idx])
                    text_idx += 1
                else:
                    loss_mask.append(0)
                input_idx += 1
            elif vision_end_id is not None and token == vision_end_id:
                # <|vision_end|> - match with text and exit vision region
                in_vision = False
                # Skip <|image_pad|> in text_token_ids if present
                while text_idx < len(text_token_ids) and image_pad_id is not None and text_token_ids[text_idx] == image_pad_id:
                    text_idx += 1
                if text_idx < len(text_token_ids) and text_token_ids[text_idx] == vision_end_id:
                    loss_mask.append(loss_mask_text[text_idx])
                    text_idx += 1
                else:
                    loss_mask.append(0)
                input_idx += 1
            elif in_vision:
                # Inside vision region - these are image tokens, no loss
                loss_mask.append(0)
                input_idx += 1
            else:
                # Regular text token - should match
                if text_idx < len(text_token_ids) and token == text_token_ids[text_idx]:
                    loss_mask.append(loss_mask_text[text_idx])
                    text_idx += 1
                    input_idx += 1
                else:
                    raise ValueError(
                        f"Token mismatch at text_idx={text_idx}, input_idx={input_idx}. "
                        f"Expected {text_token_ids[text_idx] if text_idx < len(text_token_ids) else 'EOF'}, "
                        f"got {token}."
                    )
        
        return loss_mask

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
