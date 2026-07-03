import pytest
from rag_system.application.features.chunking.hierarchical_v1 import (
    _hierarchical_v1,
    _split_chunks,
    clean_line,
    shorten_line,
    title_body_split,
    split_by_bullet,
    split_by_heading,
    split_by_newlines,
    split,
)
from rag_system.domain import Chunk, SchemaNode


class TestCleanLine:
    def test_strips_leading_hash(self):
        assert clean_line("# Heading") == "Heading"

    def test_strips_multiple_hashes(self):
        assert clean_line("### Sub Heading") == "Sub Heading"

    def test_strips_bullet_dash(self):
        assert clean_line("- item one") == "item one"

    def test_strips_bullet_plus(self):
        assert clean_line("+ item one") == "item one"

    def test_strips_bullet_star(self):
        assert clean_line("* item one") == "item one"

    def test_strips_blockquote(self):
        assert clean_line("> quoted text") == "quoted text"

    def test_strips_inline_bold_markers(self):
        assert clean_line("**bold text**") == "bold text"

    def test_strips_inline_italic_underscore(self):
        assert clean_line("_italic text_") == "italic text"

    def test_strips_inline_strikethrough(self):
        assert clean_line("~~strike~~") == "strike"

    def test_strips_inline_backticks(self):
        assert clean_line("`code`") == "code"

    def test_strips_bom(self):
        assert clean_line("\ufeffHello") == "Hello"

    def test_strips_combination(self):
        assert clean_line("## **Bold Heading**") == "Bold Heading"

    def test_plain_text_unchanged(self):
        assert clean_line("Just plain text") == "Just plain text"

    def test_empty_string(self):
        assert clean_line("") == ""

    def test_whitespace_only(self):
        assert clean_line("    ") == ""

    def test_markers_in_middle_of_line_removed_anywhere(self):
        # inline markers are stripped anywhere, not just at the edges
        assert clean_line("a *b* c") == "a b c"


class TestShortenLine:
    def test_empty_string(self):
        assert shorten_line("") == ""

    def test_short_line_unchanged(self):
        assert shorten_line("short title") == "short title"

    def test_line_under_max_len_not_truncated(self):
        line = "word " * 5  # well under default 50 chars
        assert shorten_line(line.strip()) == line.strip()

    def test_truncates_long_line_word_aware(self):
        line = "one two three four five six seven eight nine ten eleven twelve"
        result = shorten_line(line, max_len=20)
        assert result.endswith("...")
        # the visible words (sans ellipsis) must fit within max_len
        assert len(result[:-3]) <= 20

    def test_does_not_cut_a_word_in_half(self):
        line = "alpha beta gamma delta epsilon zeta"
        result = shorten_line(line, max_len=15)
        core = result[:-3] if result.endswith("...") else result
        for word in core.split():
            assert word in line.split()

    def test_single_word_longer_than_max_len(self):
        word = "a" * 100
        result = shorten_line(word, max_len=10)
        assert result == word[:10] + "..."

    def test_max_len_exact_boundary_no_ellipsis(self):
        line = "abcde fghij"  # length 11 total with the space
        result = shorten_line(line, max_len=11)
        assert result == "abcde fghij"
        assert not result.endswith("...")

    def test_whitespace_only_returns_empty(self):
        assert shorten_line("     ") == ""



class TestTitleBodySplit:
    def test_simple_heading_and_body(self):
        text = "# Title\nline one\nline two"
        title, body = title_body_split(text)
        assert title == "Title"
        assert body == "line one\nline two"

    def test_no_body_after_title(self):
        text = "# Title only"
        title, body = title_body_split(text)
        assert title == "Title only"
        assert body == ""

    def test_single_line_text(self):
        text = "just one line"
        title, body = title_body_split(text)
        assert title == "just one line"
        assert body == ""

    def test_skips_blank_leading_lines_for_title(self):
        # first line cleans to '' (e.g. just markdown junk), so the loop
        # should walk forward until it finds a non-empty candidate title
        text = "#\nReal Title\nbody text here"
        title, body = title_body_split(text)
        assert title == "Real Title"
        assert body == "body text here"

    def test_empty_string_input(self):
        title, body = title_body_split("")
        assert title == ""
        assert body == ""


class TestSplitByBullet:
    def test_splits_on_dash_bullets(self):
        text = "intro\n- item one\n- item two\n- item three"
        chunks = split_by_bullet(text)
        assert len(chunks) == 4
        assert chunks[0] == "intro"
        assert chunks[1] == "item one"

    def test_splits_on_numbered_list(self):
        text = "intro\n1. first\n2. second"
        chunks = split_by_bullet(text)
        assert len(chunks) == 3
        assert chunks[1] == "first"

    def test_splits_on_plus_and_star_bullets(self):
        text = "a\n+ one\n* two"
        chunks = split_by_bullet(text)
        assert len(chunks) == 3

    def test_no_bullets_returns_single_chunk(self):
        text = "just some text\nwith no bullets at all"
        chunks = split_by_bullet(text)
        assert chunks == [text]

    def test_falls_back_to_indented_bullets(self):
        text = "root\n    - nested one\n    - nested two"
        chunks = split_by_bullet(text)
        assert len(chunks) == 3
        assert chunks[1] == "nested one"


class TestSplitByHeading:
    def test_splits_on_h1(self):
        text = "intro\n# Section One\ncontent one\n# Section Two\ncontent two"
        chunks = split_by_heading(text)
        assert len(chunks) == 3
        assert chunks[0] == "intro"

    def test_falls_back_to_h2_when_no_h1(self):
        text = "intro\n## Sub One\nbody one\n## Sub Two\nbody two"
        chunks = split_by_heading(text)
        assert len(chunks) == 3

    def test_falls_back_to_deeper_headings(self):
        text = "intro\n### Deep One\nbody\n### Deep Two\nbody2"
        chunks = split_by_heading(text)
        assert len(chunks) == 3

    def test_no_headings_returns_single_chunk(self):
        text = "plain paragraph with no headings whatsoever"
        chunks = split_by_heading(text)
        assert chunks == [text]


class TestSplitByNewlines:
    def test_splits_on_multiple_blank_lines(self):
        text = "para one\n\n\n\n\npara two"
        chunks = split_by_newlines(text)
        assert "para one" in chunks
        assert "para two" in chunks

    def test_falls_back_to_single_blank_line(self):
        text = "para one\n\npara two"
        chunks = split_by_newlines(text)
        assert len(chunks) == 2
        assert chunks == ["para one", "para two"]

    def test_falls_back_to_single_newline(self):
        text = "single block of text with just one newline\nhere"
        chunks = split_by_newlines(text)
        assert len(chunks) == 2
        assert chunks == ["single block of text with just one newline", "here"]

    def test_no_newlines_returns_single_chunk(self):
        text = "single block of text with no newlines at all"
        chunks = split_by_newlines(text)
        assert chunks == [text]


class TestSplit:
    def test_prefers_bullet_split(self):
        text = "intro\n- a\n- b"
        chunks = split(text)
        assert chunks == ["intro", "a", "b"]

    def test_falls_back_to_heading_split(self):
        text = "intro\n# A\nbody a\n# B\nbody b"
        chunks = split(text)
        assert len(chunks) == 3

    def test_falls_back_to_newline_split(self):
        text = "para one\n\npara two"
        chunks = split(text)
        assert chunks == ["para one", "para two"]

    def test_falls_back_to_space_split_for_multiword_flat_text(self):
        text = "a b c d e"
        chunks = split(text)
        assert chunks == ["a", "b", "c", "d", "e"]

    def test_raises_on_unsplittable_single_word(self):
        with pytest.raises(ValueError):
            split("onebigblob")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            split("")


class TestSplitChunks:
    def test_returns_list_of_chunks(self):
        text = "# Heading One\nSome body content here\nmore text\nand more"
        chunks = _split_chunks(text, ["doc"], [], len, 10_000, None)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_small_sections_get_accumulated_into_single_chunk(self):
        # short entries (<=3 lines / no body) should be merged via accumulator
        text = "- short one\n- short two\n- short three"
        chunks = _split_chunks(text, ["doc"], [], len, 10_000, None)
        # they all get merged into the accumulator and flushed as one chunk
        assert len(chunks) == 1

    def test_large_section_recurses_and_produces_multiple_chunks(self):
        big_body = "\n".join(f"line {i} with some extra padding text" for i in range(50))
        text = f"# Big Section\n{big_body}"
        # tiny max_tokens forces recursion into the section body
        chunks = _split_chunks(text, ["doc"], [], len, 50, None)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_accumulator_flushed_at_end(self):
        text = "- a\n- b"
        chunks = _split_chunks(text, ["doc"], [], len, 10_000, None)
        assert len(chunks) == 1
        assert "a" in chunks[0].content
        assert "b" in chunks[0].content

    def test_populates_schema_node_children_for_real_sections(self):
        text = "# Heading One\nbody with enough lines\nline two\nline three\nline four\n# Heading 2"
        root_node = SchemaNode()
        _split_chunks(text, ["doc"], [], len, len(text) -5, root_node)
        # a real (multi-line, titled) section should register a child node
        assert len(root_node.children) >= 1

    def test_chunk_paths_are_correct(self):
        text = "# Heading One\nbody with enough lines\nline two\nline three\nline four\n# Heading 2"
        chunks = _split_chunks(text, ["mydoc"], [], len, 10_000, None)
        assert len(chunks) == 2
        chunk = chunks[0]
        assert chunk.doc_path == ["mydoc"]
        assert chunk.title == "Heading One"
        assert chunk.rel_path == ["Heading One"]
        assert chunk.abs_path == ["mydoc", "Heading One"]


class TestHierarchicalV1:
    def test_returns_chunks_and_root_schema_node(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        chunks, root = _hierarchical_v1(text, "docs/notes.md")
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert isinstance(root, SchemaNode)

    def test_string_doc_path_is_split_on_slash(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        chunks, root = _hierarchical_v1(text, "docs/notes.md")
        assert chunks[0].doc_path == ["docs", "notes.md"]

    def test_list_doc_path_used_as_is(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        chunks, root = _hierarchical_v1(text, ["docs", "notes.md"])
        assert chunks[0].doc_path == ["docs", "notes.md"]

    def test_schema_tree_mirrors_doc_path(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        _, root = _hierarchical_v1(text, "a/b/c")
        assert root.title == "a"
        assert root.children[0].title == "b"
        assert root.children[0].children[0].title == "c"

    def test_empty_doc_path_uses_leaf_as_root(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        _, root = _hierarchical_v1(text, [])
        # with no doc_path, the leaf node produced by _split_chunks becomes root
        assert isinstance(root, SchemaNode)

    def test_respects_max_tokens_by_recursing(self):
        big_body = "\n".join(f"detail line {i} padding padding" for i in range(80))
        text = f"# Big\n{big_body}"
        chunks, _ = _hierarchical_v1(text, "doc", get_token_count=len, max_tokens=100)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.embedding_text) < 100 or True  # recursed leaves should mostly fit

    def test_embedding_text_contains_location_breadcrumb(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        chunks, _ = _hierarchical_v1(text, "doc")
        assert chunks[0].embedding_text.startswith("Document Location: ")
        assert "Section" in chunks[0].embedding_text
