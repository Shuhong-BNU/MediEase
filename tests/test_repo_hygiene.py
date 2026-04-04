"""
仓库整洁度测试。

目标：
- 确保主目录里已经移除旧的作者署名横幅。
- 排除嵌套副本目录和缓存目录，避免误报。
"""

from __future__ import annotations

import unittest
from pathlib import Path


class RepoHygieneTests(unittest.TestCase):
    """扫描主目录代码与文档，确认历史署名已经清理。"""

    def test_author_banner_removed_from_main_tree(self) -> None:
        root = Path(__file__).resolve().parents[1]
        excluded_fragments = ["__pycache__", f"{root.name}\\{root.name}\\"]
        text_extensions = {
            ".py",
            ".js",
            ".html",
            ".css",
            ".md",
            ".txt",
            ".sql",
            ".xml",
            ".iml",
            ".gitignore",
            ".example",
        }
        legacy_banner = "作者：" + "小红书@人间清醒的李某人"

        offenders = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            path_str = str(path)
            if any(fragment in path_str for fragment in excluded_fragments):
                continue
            if path.name in {"PatientAgent_面试复习手册-1.0.md"}:
                continue
            if path.suffix.lower() not in text_extensions and path.name not in {
                "README.md",
                "requirements.txt",
                ".env.example",
            }:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if legacy_banner in content:
                offenders.append(path_str)

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
