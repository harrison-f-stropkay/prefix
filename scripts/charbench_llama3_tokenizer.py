from prefix.config import LLAMA3_HF_ID
from prefix.objectives import load_tokenizer


def main() -> None:
    tokenizer = load_tokenizer(LLAMA3_HF_ID)
    normal_vocab = tokenizer.vocab

    # Ensure that all 11 mult choice answers exist as tokens
    questionable_tokens = [str(i) for i in range(11)]
    for t in questionable_tokens:
        assert t in normal_vocab

    byte_vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(normal_vocab))]
    for vocab in [normal_vocab, byte_vocab]:
        # Ensure that no token has a prefix of a number, then something afterward
        # This shows that all non-"reserved_special_token" tokens that contain a number ONLY contain that number and nothing else; follows from the tokenizer regex
        # So, we're not worried about a model assigning mass to "4\n" when we're looking for the mass it puts on "4"
        digits = [str(i) for i in range(10)]
        for word in vocab:
            if any(d in word for d in digits):
                assert (
                    not any(char not in digits for char in word) or "reserved_special_token" in word
                )

    print("all ok")


if __name__ == "__main__":
    main()
