from rag_system.application.etl.text_cleaners import clean_text


class TestPropertyTags:
    def test_removes_tag_pair_no_indent(self):
        result = clean_text("- [Status]();-[Draft]()")
        assert result.strip() == ""

    def test_removes_tag_pair_space_indented(self):
        result = clean_text("    - [Status]();-[Draft]()")
        assert result.strip() == ""

    def test_removes_tag_pair_tab_indented(self):
        result = clean_text("\t- [Status]();-[Draft]()")
        assert result.strip() == ""

    def test_removes_tag_pair_deep_indent(self):
        result = clean_text("                - [Color]();-[Green]()")
        assert result.strip() == ""

    def test_tag_pair_line_not_present_in_output(self):
        text = "- MLE\n    - [Status]();-[Draft]()\n    - ML engineering"
        result = clean_text(text)
        assert "[Status]" not in result
        assert "[Draft]" not in result

    def test_sibling_lines_preserved(self):
        text = "- MLE\n    - [Status]();-[Draft]()\n    - ML engineering"
        result = clean_text(text)
        assert "MLE" in result
        assert "ML engineering" in result

    def test_color_tag_deep_nested(self):
        text = "            - Docker\n                - [Color]();-[Green]()"
        result = clean_text(text)
        assert "[Color]" not in result
        assert "[Green]" not in result
        assert "Docker" in result


class TestEmptyBullets:
    def test_removes_empty_bullet_no_indent(self):
        result = clean_text("- ")
        assert result.strip() == ""

    def test_removes_empty_bullet_space_indented(self):
        result = clean_text("    - ")
        assert result.strip() == ""

    def test_removes_empty_bullet_tab_indented(self):
        result = clean_text("\t- ")
        assert result.strip() == ""

    def test_removes_multiple_empty_bullets(self):
        text = "    - \n    - \n    - \n  - MLOps principles"
        result = clean_text(text)
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 1
        assert "MLOps principles" in lines[0]

    def test_preserves_content_after_empty_bullets(self):
        text = "    - \n    - \n  - MLOps principles\n   - reproducibility"
        result = clean_text(text)
        assert "MLOps principles" in result
        assert "reproducibility" in result


class TestPortals:
    def test_removes_portal_header(self):
        text = "--------------------- Portal ---------------------\nSome content"
        result = clean_text(text)
        assert "Portal" not in result
        assert "Some content" in result


class TestLinks:
    def test_keeps_link_text(self):
        result = clean_text("[My Note](some/path)")
        print(result)
        assert "My Note" in result

    def test_removes_link_url(self):
        result = clean_text("[My Note](some/path)")
        assert "some/path" not in result

    def test_removes_images_entirely(self):
        result = clean_text("![alt text](image.png)")
        assert "image.png" not in result
        assert "alt text" not in result


class TestCheckboxes:
    def test_removes_empty_checkbox(self):
        result = clean_text("- [ ] task")
        assert "[ ]" not in result

    def test_removes_checked_checkbox(self):
        result = clean_text("- [x] done")
        assert "[x]" not in result

    def test_preserves_task_text(self):
        result = clean_text("- [ ] task")
        assert "task" in result


class TestBlankLines:
    def test_removes_blank_lines_between_content(self):
        result = clean_text("Docker\n\n\nKubernetes")
        assert "\n\n" not in result

    def test_full_example(self):
        text = (
            "    - \n"
            "    - \n"
            "    - \n"
            "  - MLOps concepts\n"
            "   - feature pipelines run offline\n"
            "\n"
            "    - [Status]();-[Draft]()\n"
            "- Docker\n"
            "    - Basics\n"
            "        - containers are isolated processes\n"
            "    - ## Commands\n"
            "        - docker compose up -d\n"
            "        - \n"
            "\n"
            "            - Vector databases\n"
            "                - [Color]();-[Green]()\n"
            "            - Qdrant\n"
            "                - [Color]();-[Green]()\n"
        )
        result = clean_text(text)
        assert "MLOps concepts" in result
        assert "feature pipelines run offline" in result
        assert "Docker" in result
        assert "Basics" in result
        assert "containers are isolated processes" in result
        assert "docker compose up -d" in result
        assert "Vector databases" in result
        assert "Qdrant" in result
        assert "[Status]" not in result
        assert "[Draft]" not in result
        assert "[Color]" not in result
        assert "[Green]" not in result
        assert "\n\n" not in result
