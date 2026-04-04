"""
风控服务测试。
覆盖点：
- 高风险问法能命中告警。
- 回答后风险检查会对过度确定表达补充告警。
"""

from __future__ import annotations

import unittest

from app.services import safety_service


class SafetyServiceTests(unittest.TestCase):
    def test_high_risk_query_triggers_alerts_and_escalation(self) -> None:
        assessment = safety_service.analyze_user_query("我胸痛得厉害，可以现在把药停掉吗？")

        self.assertEqual(assessment["risk_level"], "high")
        self.assertGreaterEqual(len(assessment["alerts"]), 2)
        self.assertTrue(safety_service.should_escalate(assessment))

    def test_post_check_flags_overconfident_answer(self) -> None:
        assessment = safety_service.analyze_answer(
            "你可以立即停药，基本可以确诊。",
            "我是不是这个病，能不能停药？",
        )

        self.assertEqual(assessment["risk_level"], "high")
        self.assertTrue(
            any("过度确定" in item["message"] for item in assessment["alerts"])
        )


if __name__ == "__main__":
    unittest.main()
