## Tokenizing

### HF:

The Llama 3 HF pipeline:

- `normalizer`: null--don't normalize anything
- `pre_tokenizer`:
  - Split: since the regex options under the `|` operators cover all cases and since we're using `Isolated`, the split boils down to a ranking of regex matches in this order:
    1. **Contraction Suffixes**: e.g., `'s`, `'re`
    2. **Words (Letter runs)**: one or more letters (in any language), plus an optional additional single character prefix (which must be non-letter and non-number and non-newline), allowing as one common case the prefix space, e.g., `" the"`
    3. **Numbers (Numeral runs)**: 1-3 unicode numbers (e.g., 512, ½, Ⅻ)
    4. **Symbols (non-Letter non-Number runs)**: one or more symbols with an optional space (" ") prefix and zero or more newlines
    5. **Breaks (newline runs)**: whitespace followed by at least one newline (so, the newline is mandatory)
    6. **Whitespace**: whitespace _not_ followed by non-whitespace (next char is end of string or whitespace, leaving room for the next char)
    7. **Misc Whitespace**: one or more whitespace

Source below from `/Users/strophf1/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/tokenizer.json`

```json
"normalizer": null,
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": false
      }
    ]
  },
  "post_processor": {
    "type": "Sequence",
    "processors": [
      {
        "type": "ByteLevel",
        "add_prefix_space": true,
        "trim_offsets": false,
        "use_regex": true
      },
      {
        "type": "TemplateProcessing",
        // ...I redacted for brevity...
        "special_tokens": {
          "<|begin_of_text|>": {
            "id": "<|begin_of_text|>",
            "ids": [
              128000
            ],
            "tokens": [
              "<|begin_of_text|>"
            ]
          }
        }
      }
    ]
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": true,
    "vocab": {
    // ...I redacted for brevity...
```

### NFC:

For example, the distinct Unicode strings "U+212B" (the angstrom sign "Å") and "U+00C5" (the Swedish letter "Å") are both expanded by NFD (or NFKD) into the sequence "U+0041 U+030A" (Latin letter "A" and combining ring above "°") which is then reduced by NFC (or NFKC) to "U+00C5" (the Swedish letter "Å").

- So we shouldn't normalize to NFC; perhaps "U+212B" is used in physics contexts more frequently, and "U+00C5" is used in Swedish literature more frequently
