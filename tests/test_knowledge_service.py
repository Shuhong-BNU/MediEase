"""
知识库混合检索测试。

覆盖点：
1. LangChain Retriever 继续可用。
2. 知识库搜索不仅支持关键词命中，也支持纯向量命中。
3. 向量链路命中时会返回 `vector` / `hybrid` 等检索标签，便于前端和调试展示。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import KnowledgeDocument
from app.services import knowledge_service, knowledge_vector_service


class KnowledgeServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "knowledge_test.db"
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(self.engine)
        self.db: Session = self.SessionLocal()

        self.db.add_all(
            [
                KnowledgeDocument(
                    id=1,
                    title="Alpha guidance",
                    category="general",
                    content="alpha only guidance",
                    keywords="ALPHA_GUIDE",
                    enabled=True,
                ),
                KnowledgeDocument(
                    id=2,
                    title="Blood pressure warning",
                    category="general",
                    content="beta pressure guidance",
                    keywords="BETA_GUIDE",
                    enabled=True,
                ),
            ]
        )
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_search_can_return_vector_only_hit(self) -> None:
        class FakeIndex:
            def __init__(self, dimensions: int) -> None:
                self.d = dimensions
                self._vectors: list[list[float]] = []

            def add(self, matrix) -> None:
                self._vectors = [list(row) for row in matrix]

            def search(self, query_vector, top_n: int):
                query = list(query_vector[0])
                ranked = sorted(
                    enumerate(
                        sum(left * right for left, right in zip(vector, query))
                        for vector in self._vectors
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
                scores = [float(score) for _, score in ranked[:top_n]]
                positions = [int(index) for index, _ in ranked[:top_n]]
                while len(scores) < top_n:
                    scores.append(float("-inf"))
                    positions.append(-1)
                return [scores], [positions]

        class FakeFaiss:
            def __init__(self) -> None:
                self.storage: dict[str, FakeIndex] = {}

            def IndexFlatIP(self, dimensions: int) -> FakeIndex:
                return FakeIndex(dimensions)

            def write_index(self, index: FakeIndex, path: str) -> None:
                self.storage[path] = index
                path_obj = Path(path)
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                path_obj.write_text("fake-index", encoding="utf-8")

            def read_index(self, path: str) -> FakeIndex:
                return self.storage[path]

        class FakeNumpy:
            @staticmethod
            def array(value, dtype=None):
                return value

        def fake_embed_documents(texts: list[str]) -> list[list[float]]:
            vectors = []
            for text in texts:
                lowered = text.lower()
                if "alpha" in lowered:
                    vectors.append([1.0, 0.0])
                elif "blood pressure" in lowered or "beta" in lowered:
                    vectors.append([0.0, 1.0])
                else:
                    vectors.append([0.5, 0.5])
            return vectors

        fake_faiss = FakeFaiss()
        with patch.object(knowledge_vector_service, "_embed_documents", side_effect=fake_embed_documents), patch.object(
            knowledge_vector_service, "_embed_query", return_value=[0.0, 1.0]
        ), patch.object(
            knowledge_vector_service, "_embedding_dimensions", return_value=2
        ), patch.object(
            knowledge_vector_service, "faiss", fake_faiss
        ), patch.object(
            knowledge_vector_service, "np", FakeNumpy()
        ):
            knowledge_vector_service.rebuild_knowledge_index(self.db)
            results = knowledge_service.search_knowledge_documents(
                self.db,
                query="hypertension followup",
                top_n=3,
            )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].title, "Blood pressure warning")
        self.assertEqual(results[0].retrieval_label, "vector")
        self.assertEqual(results[0].retrieval_sources, ["vector"])
        self.assertGreater(results[0].vector_score, 0.0)
        self.assertEqual(results[0].keyword_score, 0.0)


if __name__ == "__main__":
    unittest.main()
