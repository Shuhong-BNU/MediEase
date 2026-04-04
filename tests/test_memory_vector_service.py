"""
长期记忆向量检索的回归测试。
覆盖点：
- 当磁盘上的 FAISS 索引文件无法读取时，服务应退回到“内存重建索引”而不是抛出 500。
- 重建后的检索结果仍应可用，并尽量把新索引重新写回磁盘。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services import memory_vector_service


class _FakeIndex:
    def __init__(self, dimensions: int) -> None:
        self.d = dimensions
        self._vectors: list[list[float]] = []

    def add(self, matrix) -> None:
        for row in matrix:
            self._vectors.append([float(value) for value in row])

    def search(self, query_vector, top_k: int):
        if not self._vectors:
            return [[0.0]], [[-1]]
        query = [float(value) for value in query_vector[0]]
        scores = [
            sum(query_value * vector_value for query_value, vector_value in zip(query, vector))
            for vector in self._vectors
        ]
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        ordered_scores = [[scores[int(position)] for position in order]]
        ordered_positions = [[int(position) for position in order]]
        return ordered_scores, ordered_positions


class MemoryVectorServiceTests(unittest.TestCase):
    """验证索引异常时的降级与恢复逻辑。"""

    def test_search_rebuilds_in_memory_when_disk_index_cannot_be_read(self) -> None:
        """磁盘索引损坏或不可读时，应现场重建并继续检索。"""

        temp_dir = tempfile.TemporaryDirectory()
        try:
            vector_dir = Path(temp_dir.name)
            index_path = vector_dir / "patient_3.index"
            index_path.write_bytes(b"broken")

            fake_faiss = SimpleNamespace()
            fake_faiss.IndexFlatIP = lambda dimensions: _FakeIndex(dimensions)
            fake_faiss.read_index = lambda path: (_ for _ in ()).throw(
                RuntimeError("cannot open index")
            )
            write_calls: list[str] = []
            fake_np = SimpleNamespace(array=lambda data, dtype=None: data)

            def _write_index(index, path: str) -> None:
                write_calls.append(path)

            fake_faiss.write_index = _write_index

            with (
                patch.object(memory_vector_service, "VECTOR_DIR", vector_dir),
                patch.object(memory_vector_service, "faiss", fake_faiss),
                patch.object(memory_vector_service, "np", fake_np),
                patch.object(
                    memory_vector_service,
                    "_load_metadata",
                    return_value=[
                        {
                            "event_id": 11,
                            "patient_id": 3,
                            "event_type": "visit_record",
                            "event_time": "2026-04-04T10:00:00",
                            "source_type": "visit_record",
                            "source_id": "1",
                            "document": "心内科复诊 稳定",
                        }
                    ],
                ),
                patch.object(memory_vector_service, "_embedding_dimensions", return_value=2),
                patch.object(memory_vector_service, "_embed_documents", return_value=[[1.0, 0.0]]),
                patch.object(memory_vector_service, "_embed_query", return_value=[1.0, 0.0]),
            ):
                rows = memory_vector_service.search_memory_events(
                    patient_id=3,
                    query="最近一次心内科复诊怎么样",
                    top_n=1,
                )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], 11)
            self.assertGreater(rows[0]["vector_score"], 0.0)
            self.assertEqual(write_calls, [str(index_path)])
        finally:
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
