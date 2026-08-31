import uuid

import pytest
from rag_system.application.features.tree.base import (
    clean_line,
    shorten_line,
    title_body_split,
    raw_heading,
    split_by_bullet,
    split_by_heading,
    split_by_newlines,
    split,
)
from rag_system.application.features.tree import build_doc_subtree
from rag_system.application.features.chunking.hierarchical import flatten_leaves
from rag_system.domain import DocNode
from rag_system.utils import get_hash


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


class TestRawHeading:
    def test_returns_non_truncated_heading(self):
        assert raw_heading("# Full Heading") == "Full Heading"

    def test_does_not_shorten_long_heading(self):
        heading = "word " * 30
        text = f"# {heading.strip()}\nbody"
        assert raw_heading(text) == heading.strip()

    def test_skips_blank_leading_lines(self):
        text = "#\nReal Title\nbody"
        assert raw_heading(text) == "Real Title"


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


class TestBuildDocSubtree:
    def test_returns_doc_subtree(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        root = build_doc_subtree(text, ["docs", "notes.md"], "abc", len, 10_000)
        assert isinstance(root, DocNode)

    def test_string_doc_path_chain_mirrors_doc_path(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        root = build_doc_subtree(text, "docs/notes.md".split('/'), "abc", len, 10_000)
        assert root.title == "docs"
        assert root.children[0].title == "notes.md"

    def test_doc_hash_set_on_root(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        root = build_doc_subtree(text, ["docs", "notes.md"], "hash123", len, 10_000)
        assert root.doc_hash == "hash123"

    def test_chunk_paths_are_correct(self):
        text = "# Heading One\nbody with enough lines\nline two\nline three\nline four\n# Heading 2"
        root = build_doc_subtree(text, ["mydoc"], "abc", len, 10_000)
        leaves = list(root.walk_leaves())
        section = next(leaf for leaf in leaves if not leaf.is_content)
        assert section.doc_path == ["mydoc"]
        assert section.title == "Heading One"
        assert section.rel_path == ["Heading One"]
        assert section.abs_path == ["mydoc", "Heading One"]

    def test_content_leaf_holds_accumulated_small_entries(self):
        text = "# Heading One\nbody line one\nline two\nline three\nline four\n# Heading 2"
        root = build_doc_subtree(text, ["mydoc"], "abc", len, 10_000)
        content = [leaf for leaf in root.walk_leaves() if leaf.is_content]
        assert len(content) == 1
        assert content[0].title == ""
        assert content[0].text == "\nHeading 2"

    def test_small_entries_accumulate_into_single_leaf(self):
        text = "- short one\n- short two\n- short three"
        root = build_doc_subtree(text, ["mydoc"], "abc", len, 10_000)
        leaves = list(root.walk_leaves())
        assert len(leaves) == 1
        assert "short one" in leaves[0].text
        assert "short three" in leaves[0].text

    def test_large_section_recurses_and_internal_text_is_heading(self):
        # two multi-line sections; max_size small enough that each section recurses
        text = ("# Big Section\nline one\nline two\nline three\nline four\n"
                "# Other Section\nb1\nb2\nb3\nb4")
        root = build_doc_subtree(text, ["mydoc"], "abc", len, 50)
        sections = [c for c in root.children if not c.is_content]
        assert sections
        internal = sections[0]
        assert internal.children
        assert internal.text == "Big Section"

    def test_ids_encode_position(self):
        text = "# A\nbody a\nbody b\nbody c\n# B\nbody d\nbody e\nbody f"
        root = build_doc_subtree(text, ["mydoc"], "abc", len, 10_000)
        # doc root is "0", children "00", "01", ...
        assert root.id == "0"
        assert [c.id for c in root.children] == ["00", "01"]

    def test_depth_and_distance_to_leaves(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        root = build_doc_subtree(text, ["mydoc"], "abc", len, 10_000)
        assert root.depth == 0
        for leaf in root.walk_leaves():
            assert leaf.distance_to_leaves == 0
        assert root.distance_to_leaves == 1


class TestFlattenLeaves:
    def test_documents_match_previous_chunk_output(self):
        text = "# Heading One\nbody with enough lines\nline two\nline three\nline four\n# Heading 2"
        root = build_doc_subtree(text, ["mydoc"], "abc", len, 10_000)
        documents = flatten_leaves([root])
        assert len(documents) == 2

        section_doc = documents[0]
        assert section_doc.page_content == (
            "Document Location: mydoc > Heading One\n\n"
            "# Heading One\nbody with enough lines\nline two\nline three\nline four"
        )
        assert section_doc.metadata["doc_path"] == "mydoc"
        assert section_doc.metadata["title"] == "Heading One"
        assert section_doc.metadata["abs_path"] == "mydoc/Heading One"
        assert section_doc.metadata["rel_path"] == "Heading One"
        assert section_doc.metadata["doc_hash"] == "abc"

        content_doc = documents[1]
        assert content_doc.page_content == "Document Location: mydoc > \n\n\nHeading 2"
        assert content_doc.metadata["title"] == ""
        assert content_doc.metadata["abs_path"] == "mydoc/"

    def test_deterministic_id_from_text_and_hash(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        root = build_doc_subtree(text, ["doc"], "abc", len, 10_000)
        documents = flatten_leaves([root])
        expected = str(uuid.UUID(get_hash(documents[0].page_content + "abc")))
        assert documents[0].id == expected

    def test_empty_doc_produces_no_chunks(self):
        root = build_doc_subtree("", ["doc"], "abc", len, 10_000)
        documents = flatten_leaves([root])
        assert documents == []


class TestUnite:
    def test_unite_merges_shared_doc_path_prefixes(self):
        text = "# Section\nbody line one\nline two\nline three\nline four"
        doc1 = build_doc_subtree(text, ["Root", "Section", "note1.md"], "a", len, 10_000)
        doc2 = build_doc_subtree(text, ["Root", "Section", "note2.md"], "b", len, 10_000)
        united = DocNode.unite([doc1, doc2])
        assert united.title == "root"
        assert united.children[0].title == "Root"
        assert united.children[0].children[0].title == "Section"
        assert [c.title for c in united.children[0].children[0].children] == [
            "note1.md", "note2.md"
        ]

    def test_unite_skips_content_leaves(self):
        text = "# Heading One\nbody line one\nline two\nline three\nline four\n# Heading 2"
        doc = build_doc_subtree(text, ["mydoc"], "a", len, 10_000)
        united = DocNode.unite([doc])
        # content leaf titled "" must not appear in the united schema
        schema = str(united)
        assert "- root" in schema
        assert "mydoc" in schema
        assert "- Heading One" in schema
        assert schema.count("Heading") == 1

    def test_schema_renders_nested_doc_paths(self):
        text = "# Section\nbody line one\nline two\nline three\nline four\n# End"
        doc = build_doc_subtree(text, ["Root", "Section", "note.md"], "a", len, 10_000)
        united = DocNode.unite([doc])
        assert str(united) == (
            "- root\n"
            "  - Root\n"
            "    - Section\n"
            "      - note.md\n"
            "        - Section"
        )
