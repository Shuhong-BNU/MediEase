"""
报告服务测试。
覆盖点：
- 报告文本解读能产出摘要、异常项、风险等级和免责声明。
- 报告 CRUD 的最小链路可用。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Patient
from app.schemas.report import MedicalReportCreate
from app.services import report_service


class ReportServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "report_test.db"
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(self.engine)
        self.db = self.SessionLocal()

        patient = Patient(patient_code="P1001", full_name="测试患者")
        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)
        self.patient = patient

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_interpret_report_text_returns_abnormal_items_and_risk(self) -> None:
        result = report_service.interpret_report_text(
            "血常规：白细胞 12.6 偏高\nC 反应蛋白 36 异常\n建议结合感染症状判断。",
            title="血常规复查",
            report_type="检验",
        )

        self.assertIn("血常规复查", result.summary)
        self.assertGreaterEqual(len(result.abnormal_items), 2)
        self.assertEqual(result.risk_level, "medium")
        self.assertTrue(result.disclaimer)

    def test_create_and_fetch_medical_report(self) -> None:
        report = report_service.create_medical_report(
            self.db,
            MedicalReportCreate(
                patient_id=self.patient.id,
                report_code="R0001",
                report_type="检验",
                title="血常规",
                raw_text="白细胞 12.6 偏高",
            ),
        )

        fetched = report_service.get_medical_report(self.db, report.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.report_code, "R0001")


if __name__ == "__main__":
    unittest.main()
